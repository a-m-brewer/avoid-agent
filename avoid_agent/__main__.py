"""Main entry point for the Avoid Agent CLI."""

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

from avoid_agent.agent.context import ContextStrategy
from avoid_agent.agent.runtime import AgentRuntime, RuntimeEvent, _parse_structured_action
from avoid_agent.agent.tools.finder import find_available_tools
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from avoid_agent import providers
from avoid_agent.providers import list_available_models, get_saved_model, save_selected_model, load_user_config, save_user_config
from avoid_agent.permissions import load_allowed, save_allowed
from avoid_agent.session import delete_session, list_sessions, load_session, save_session
from avoid_agent.prompts import build_system_prompt, export_system_prompt_markdown
from avoid_agent.prompts.system_prompt import SystemPromptOptions
from avoid_agent.tui import TUI
from avoid_agent.tui.components.conversation import (
    AssistantItem,
    ConversationItem,
    StatusItem,
    ToolCallItem,
    ToolResultItem,
    UserItem,
)


def _count_backlog_totals(repo_root: Path) -> tuple[int, int]:
    """Return (completed, total) for markdown checklist items in backlog.md."""
    backlog_path = repo_root / "backlog.md"
    if not backlog_path.exists():
        return (0, 0)

    completed = 0
    total = 0
    with open(backlog_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if re.match(r"^- \[( |x|!)\] .+", line):
                total += 1
            if re.match(r"^- \[x\] .+", line):
                completed += 1
    return (completed, total)


def _stream_selfdev_headless_stderr(pipe, tui: TUI, tool_count_ref: list[int]) -> None:
    """Map headless stderr JSON events into rich TUI updates."""
    text_buffer = ""
    suppress_streaming = False

    for raw_line in pipe:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "turn_start":
            turn = event.get("turn", "?")
            tui.set_phase(f"turn {turn}")
            tui.push_item(StatusItem(text=f"turn {turn} started"))
            tui.set_spinner_message("thinking...")

        elif etype == "text_delta":
            delta = event.get("text", "")
            if delta:
                text_buffer += delta
                if not suppress_streaming:
                    if text_buffer.lstrip().startswith("{"):
                        suppress_streaming = True
                    else:
                        tui.append_chunk(delta)

        elif etype == "tool_call_detected":
            call_id = event.get("id")
            name = event.get("name", "tool")
            args = event.get("arguments", {})
            tui.push_item(ToolCallItem(id=call_id, name=name, arguments=args, status="pending"))
            tui.set_spinner_message(f"tool detected: {name}")

        elif etype == "tool_execution_start":
            text_buffer = ""
            suppress_streaming = False
            tool_count_ref[0] += 1
            call_id = event.get("id")
            name = event.get("name", "tool")
            args = event.get("arguments", {})
            tui.push_item(ToolCallItem(id=call_id, name=name, arguments=args, status="running"))
            if call_id:
                tui.update_tool_status(call_id, "running")
            tui.set_spinner_message(f"running tool: {name}")

        elif etype == "tool_result":
            call_id = event.get("id")
            name = event.get("name") or "tool"
            content = str(event.get("content") or "")
            is_error = bool(event.get("is_error", False))
            status = "failed" if is_error else "done"
            if call_id:
                tui.update_tool_status(call_id, status)
            tui.push_item(
                ToolResultItem(
                    id=call_id,
                    name=name,
                    content=content,
                    status=status,
                )
            )
            tui.reset_spinner_message()

        elif etype == "reasoning":
            item = event.get("item")
            summary = "reasoning"
            if isinstance(item, dict):
                value = item.get("summary")
                if isinstance(value, list):
                    summary = " ".join(str(x) for x in value if x)
                elif value:
                    summary = str(value)
            tui.push_item(StatusItem(text=f"reasoning: {summary}"))
            tui.set_spinner_message("reasoning...")

        elif etype == "status":
            message = str(event.get("message") or "")
            if message:
                tui.push_item(StatusItem(text=message))
                tui.set_spinner_message(message)

        elif etype == "structured_action":
            text_buffer = ""
            suppress_streaming = False
            message = event.get("message")
            if isinstance(message, str) and message.strip():
                tui.replace_last_assistant(message)

        elif etype == "context_trimmed":
            message = event.get("message")
            if isinstance(message, str) and message.strip():
                tui.push_item(StatusItem(text=message))

        elif etype in ("error", "fatal", "validation_error"):
            message = str(event.get("message") or etype)
            tui.report_error(message)
            tui.reset_spinner_message()

        elif etype == "turn_complete":
            tokens = event.get("input_tokens")
            if isinstance(tokens, int):
                tui.update_tokens(tokens)
            turn = event.get("turn", "?")
            tui.push_item(StatusItem(text=f"turn {turn} complete"))
            tui.set_phase("awaiting feedback")


def _run_selfdev_interactive(repo_root: Path, model: str | None, max_turns: int) -> int:
    """Run one selfdev cycle with live TUI streaming and optional user feedback turns."""
    from avoid_agent.selfdev import RESTART_EXIT_CODE
    from avoid_agent.selfdev.loop import (
        _commit_if_dirty,
        build_prompt_for_task,
        cleanup_worktree,
        create_worktree,
        detach_worktree,
        mark_backlog_item,
        merge_worktree,
        parse_backlog,
    )
    from avoid_agent.selfdev.validate import validate_worktree

    items = parse_backlog(repo_root)
    if not items:
        print("No unchecked backlog items. Nothing to do.")
        return 0

    item = items[0]
    done_count, total_count = _count_backlog_totals(repo_root)
    progress_current = done_count + 1

    worktree_path: Path | None = None
    branch_name = f"selfdev/{re.sub(r'[^a-z0-9]+', '-', item.text.lower())[:50].strip('-')}"

    prompts: queue.Queue[str] = queue.Queue()
    prompt_counter = [1]
    worker_done = threading.Event()
    worker_result: dict[str, object] = {"result": "error", "error": "unexpected"}

    active_model = model or os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4-6")
    tui = TUI(on_submit=lambda _text: None, model=active_model, auto_spinner_on_submit=False)
    tui.set_phase("preparing")
    tui.set_progress(progress_current, total_count)
    tui.push_item(StatusItem(text=f"task: {item.text}"))
    tui.push_item(StatusItem(text="Type feedback to send another turn, /done to finish and validate."))

    def on_submit(text: str) -> None:
        stripped = text.strip()
        if stripped == "/done":
            prompts.put("__SELFDEV_DONE__")
            tui.push_item(StatusItem(text="finishing current run and validating..."))
            return
        if stripped == "/abort":
            prompts.put("__SELFDEV_ABORT__")
            tui.push_item(StatusItem(text="aborting selfdev run..."))
            return
        prompts.put(text)
        tui.push_item(StatusItem(text=f"queued feedback turn #{prompt_counter[0]}"))
        prompt_counter[0] += 1

    tui.on_submit = on_submit

    def worker() -> None:
        nonlocal worktree_path
        proc = None
        stderr_thread = None
        try:
            tui.set_phase("creating worktree")
            worktree_path = create_worktree(repo_root, branch_name)

            prompt = build_prompt_for_task(item.text, repo_root, worktree_path)

            cmd = [
                sys.executable,
                "-m",
                "avoid_agent",
                "headless",
                "--auto-approve",
                "--no-session",
                "--max-turns",
                str(max_turns),
            ]
            if model:
                cmd.extend(["--model", model])

            env = os.environ.copy()
            env["PYTHONPATH"] = str(worktree_path)

            tui.set_phase("running agent")
            tui._start_spinner()  # pylint: disable=protected-access

            proc = subprocess.Popen(
                cmd,
                cwd=worktree_path,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            tool_count = [0]
            stderr_thread = threading.Thread(
                target=_stream_selfdev_headless_stderr,
                args=(proc.stderr, tui, tool_count),
                daemon=True,
            )
            stderr_thread.start()

            assert proc.stdin is not None
            assert proc.stdout is not None

            def send_prompt(prompt_text: str) -> None:
                proc.stdin.write(json.dumps({"prompt": prompt_text}) + "\n")
                proc.stdin.flush()

            send_prompt(prompt)

            successful_turns = 0

            while True:
                line = proc.stdout.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not result.get("success"):
                    worker_result["result"] = "failed"
                    worker_result["error"] = result.get("error") or "headless run failed"
                    break

                successful_turns += 1
                tui.push_item(StatusItem(text=f"turn {successful_turns} completed"))

                tui.push_item(StatusItem(text="waiting for your feedback (/done to continue pipeline)..."))
                next_prompt = prompts.get()

                if next_prompt is None or next_prompt == "__SELFDEV_DONE__":
                    worker_result["result"] = "success"
                    break

                if next_prompt == "__SELFDEV_ABORT__":
                    worker_result["result"] = "failed"
                    worker_result["error"] = "aborted by user"
                    break

                send_prompt(next_prompt)
                tui.set_phase("running feedback turn")

            if proc.stdin:
                try:
                    proc.stdin.write(json.dumps({"command": "quit"}) + "\n")
                    proc.stdin.flush()
                except Exception:  # pylint: disable=broad-except
                    pass

            proc.wait(timeout=10)

            if worker_result.get("result") != "success":
                error = str(worker_result.get("error") or "unknown error")
                tui.report_error(error)
                if worktree_path:
                    _commit_if_dirty(worktree_path, item.text)
                mark_backlog_item(repo_root, item, "failed", note=f"{error[:80]} | branch: {branch_name}")
                detach_worktree(repo_root, branch_name)
                return

            # Ensure there is a commit if agent only changed working tree files.
            diff_result = subprocess.run(
                ["git", "diff", "--stat", "main", "HEAD"],
                capture_output=True,
                text=True,
                cwd=worktree_path,
                check=False,
            )
            if not diff_result.stdout.strip() and not _commit_if_dirty(worktree_path, item.text):
                mark_backlog_item(repo_root, item, "failed", note="no changes produced")
                cleanup_worktree(repo_root, branch_name)
                worker_result["result"] = "failed"
                worker_result["error"] = "no changes produced"
                tui.report_error("No changes were produced.")
                return

            tui.set_phase("validating")
            validation = validate_worktree(repo_root, worktree_path)
            if not validation.passed:
                mark_backlog_item(repo_root, item, "failed", note=f"validation failed | branch: {branch_name}")
                detach_worktree(repo_root, branch_name)
                worker_result["result"] = "failed"
                worker_result["error"] = "validation failed"
                tui.report_error("Validation failed. Branch preserved for review.")
                tui.push_item(AssistantItem(text=validation.summary))
                return

            tui.set_phase("merging")
            if not merge_worktree(repo_root, branch_name):
                mark_backlog_item(repo_root, item, "failed", note=f"merge conflict | branch: {branch_name}")
                detach_worktree(repo_root, branch_name)
                worker_result["result"] = "failed"
                worker_result["error"] = "merge conflict"
                tui.report_error("Merge failed. Branch preserved for review.")
                return

            mark_backlog_item(repo_root, item, "done")
            cleanup_worktree(repo_root, branch_name)
            worker_result["result"] = "restart"
            tui.set_phase("done")
            tui.push_item(StatusItem(text=f"Task completed and merged: {item.text}"))

        except Exception as e:  # pylint: disable=broad-except
            worker_result["result"] = "error"
            worker_result["error"] = str(e)
            tui.report_error(str(e))
            if worktree_path:
                try:
                    _commit_if_dirty(worktree_path, item.text)
                    mark_backlog_item(repo_root, item, "failed", note=f"{str(e)[:80]} | branch: {branch_name}")
                    detach_worktree(repo_root, branch_name)
                except Exception:  # pylint: disable=broad-except
                    pass
        finally:
            if stderr_thread:
                stderr_thread.join(timeout=3)
            tui._stop_spinner()  # pylint: disable=protected-access
            worker_done.set()
            tui.stop()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    tui.run()
    worker_done.wait(timeout=5)

    result = str(worker_result.get("result") or "error")
    if result == "restart":
        return RESTART_EXIT_CODE
    if result == "success":
        return RESTART_EXIT_CODE
    if result in ("failed", "error"):
        return 1
    return 0


def gather_initial_context() -> tuple[str, str, str]:
    """Collect runtime context used by the system prompt and initial conversation."""
    cwd = os.getcwd()
    git_status = subprocess.run(
        "git status --short", shell=True, capture_output=True, text=True, cwd=cwd
    )
    git_output = (
        git_status.stdout.strip()
        if git_status.returncode == 0
        else "Not a git repository"
    )
    top_level_structure = subprocess.run(
        "find . -maxdepth 2 ! -path './.venv/*' ! -path './.git' ! -path './.git/*' ! -path '*/__pycache__/*'",
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return cwd, git_output, top_level_structure.stdout


def messages_to_items(messages: list[Message]) -> list[ConversationItem]:
    """Reconstruct TUI conversation items from a saved message list."""
    tool_name_map: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for tc in msg.tool_calls:
                tool_name_map[tc.id] = tc.name

    items: list[ConversationItem] = []
    for msg in messages:
        if isinstance(msg, UserMessage):
            items.append(UserItem(text=msg.text))
        elif isinstance(msg, AssistantMessage):
            if msg.text:
                items.append(AssistantItem(text=_display_text_for_assistant_message(msg)))
            for tc in msg.tool_calls:
                items.append(ToolCallItem(id=tc.id, name=tc.name, arguments=tc.arguments))
        elif isinstance(msg, ToolResultMessage):
            name = tool_name_map.get(msg.tool_call_id, "tool")
            items.append(ToolResultItem(id=msg.tool_call_id, name=name, content=msg.content))
    return items


def _display_text_for_assistant_message(message: AssistantMessage) -> str:
    structured = _parse_structured_action(message.text)
    if structured is None:
        return message.text or ""

    if structured.tool == "blocker":
        reason = structured.args.get("reason")
        if isinstance(reason, str) and reason.strip():
            return reason

    if structured.tool == "complete":
        summary = structured.args.get("summary")
        if isinstance(summary, str) and summary.strip():
            return summary

    return structured.plan


def _export_prompt_command(output: str) -> None:
    cwd, git_status, top_level_structure = gather_initial_context()
    written = export_system_prompt_markdown(
        output,
        options=SystemPromptOptions(
            working_directory=cwd,
            git_status=git_status,
            top_level_file_structure=top_level_structure,
        ),
    )
    print(f"Exported system prompt markdown to: {written}")


def _run_agent() -> None:
    load_dotenv()
    default_model = get_saved_model() or os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4-6")
    max_tokens = int(os.getenv("MAX_TOKENS", "8192"))
    tool_definitions = find_available_tools()

    cwd, git_status, top_level_structure = gather_initial_context()
    system = build_system_prompt(
        working_directory=cwd,
        git_status=git_status,
        top_level_file_structure=top_level_structure,
    )

    # Load persisted user config for thinking/effort
    user_config = load_user_config()
    thinking_enabled = bool(user_config.get("thinking", False))
    effort = user_config.get("effort", "high")
    if effort not in ("low", "medium", "high"):
        effort = "high"

    active_model = default_model
    tui = TUI(model=active_model, on_submit=lambda _: None)

    def build_provider(model: str):
        provider = providers.get_provider(
            model=model,
            system=system,
            max_tokens=max_tokens,
            thinking_enabled=thinking_enabled,
            effort=effort,
        )
        if thinking_enabled and not provider.supports_thinking:
            provider = providers.get_provider(
                model=model,
                system=system,
                max_tokens=max_tokens,
                thinking_enabled=False,
                effort=effort,
            )
            tui.set_warning("! thinking n/a")
        else:
            tui.set_warning(None)
        return provider

    provider = build_provider(active_model)

    valid_strategies: set[ContextStrategy] = {"window", "compact", "compact+window"}
    env_strategy = os.getenv("CONTEXT_STRATEGY", "compact+window")
    if env_strategy not in valid_strategies:
        env_strategy = "compact+window"
    context_strategy: ContextStrategy = env_strategy  # type: ignore[assignment]

    allowed_prefixes = load_allowed()
    active_session = "default"
    saved = load_session(cwd, active_session)
    if saved is not None:
        messages: list[Message] = saved
        restored = True
    else:
        messages: list[Message] = []
        restored = False

    # Initialize status bar with persisted settings
    tui.set_thinking_enabled(thinking_enabled)
    tui.set_effort(effort)

    def on_submit(text: str) -> None:
        nonlocal messages, active_session, context_strategy, provider, active_model, thinking_enabled, effort

        if text.strip() == "/strategy":
            tui.report_info(
                f"Current strategy: {context_strategy}\n"
                f"Options: {', '.join(sorted(valid_strategies))}\n"
                f"Use /strategy <name> to switch."
            )
            return

        # Toggle or show extended thinking mode
        if text.strip().startswith("/thinking"):
            parts = text.strip().split()
            if len(parts) == 1:
                # Toggle
                thinking_enabled = not thinking_enabled
            elif len(parts) == 2 and parts[1].lower() in {"on", "off"}:
                thinking_enabled = parts[1].lower() == "on"
            else:
                tui.report_error("Usage: /thinking [on|off]")
                return

            # Persist and apply
            cfg = load_user_config()
            cfg["thinking"] = thinking_enabled
            save_user_config(cfg)
            provider = build_provider(active_model)
            tui.set_thinking_enabled(thinking_enabled)
            tui.report_info(f"Thinking is now {'on' if thinking_enabled else 'off'}")
            return

        # Set or show reasoning effort level
        if text.strip().startswith("/effort"):
            parts = text.strip().split()
            if len(parts) == 1:
                tui.report_info("Current effort: " + effort + "\nOptions: low, medium, high\nUse /effort <level> to change.")
                return
            if len(parts) >= 2:
                new_effort = parts[1].lower()
                if new_effort not in {"low", "medium", "high"}:
                    tui.report_error("Invalid effort level. Use one of: low, medium, high")
                    return
                effort = new_effort
                cfg = load_user_config()
                cfg["effort"] = effort
                save_user_config(cfg)
                provider = build_provider(active_model)
                tui.set_effort(effort)
                tui.report_info(f"Effort set to: {effort}")
                return

        if text.strip().startswith("/strategy "):
            new_strategy = text.strip().split(maxsplit=1)[1]
            if new_strategy not in valid_strategies:
                tui.report_error(
                    f"Unknown strategy: {new_strategy}. "
                    f"Options: {', '.join(sorted(valid_strategies))}"
                )
                return
            context_strategy = new_strategy  # type: ignore[assignment]
            tui.report_info(f"Context strategy set to: {context_strategy}")
            return

        if text.strip().startswith("/model"):
            tui._stop_spinner()
            parts = text.strip().split()
            if len(parts) == 1:
                picked = tui.pick_from_list("Select model", list_available_models())
                tui._start_spinner()
                if picked is None:
                    tui.report_info("Model selection cancelled")
                    return
                new_model = picked
            elif len(parts) == 2:
                new_model = parts[1]
            elif len(parts) >= 3:
                new_model = f"{parts[1]}/{' '.join(parts[2:]).strip()}"
            else:
                tui.report_error("Usage: /model [provider/model] or /model <provider> <model>")
                return

            if "/" not in new_model:
                tui.report_error("Model must include provider prefix (example: anthropic/claude-sonnet-4-6)")
                return

            previous_model = active_model
            previous_provider = provider
            try:
                provider = build_provider(new_model)
                active_model = new_model
                tui.set_model(active_model)
                save_selected_model(active_model)
                tui.report_info(f"Switched model to: {active_model}")
            except Exception as e:  # pylint: disable=broad-except
                provider = previous_provider
                active_model = previous_model
                tui.report_error(f"Failed to switch model: {e}")
            return

        if text.strip() == "/clear":
            messages = []
            delete_session(cwd, active_session)
            tui.clear_conversation()
            return

        if text.strip() == "/resume":
            names = list_sessions(cwd)
            if not names:
                tui.report_info("No saved sessions for this repo.")
                return
            tui.report_info("Saved sessions: " + ", ".join(names) + "\nUse /resume <name>")
            return

        if text.strip().startswith("/resume "):
            name = text.strip().split(maxsplit=1)[1]
            restored_messages = load_session(cwd, name)
            if restored_messages is None:
                tui.report_error(f"Session not found: {name}")
                return
            active_session = name
            messages = restored_messages
            tui.clear_conversation()
            for item in messages_to_items(messages):
                tui.push_item(item)
            tui.report_info(f"Resumed session: {active_session}")
            return

        messages_checkpoint = messages[:]

        try:
            _text_buffer = ""
            _suppress_streaming = False

            def handle_runtime_event(event: RuntimeEvent) -> None:
                nonlocal _text_buffer, _suppress_streaming
                if event.type == "provider_event" and event.provider_event:
                    provider_event = event.provider_event
                    if provider_event.type == "text_delta" and provider_event.text:
                        _text_buffer += provider_event.text
                        if not _suppress_streaming:
                            if _text_buffer.lstrip().startswith("{"):
                                _suppress_streaming = True
                            else:
                                tui.append_chunk(provider_event.text)
                    elif provider_event.type == "tool_call_detected" and provider_event.tool_call:
                        tool_call = provider_event.tool_call
                        tui.push_item(
                            ToolCallItem(
                                id=tool_call.id,
                                name=tool_call.name,
                                arguments=tool_call.arguments,
                                status="pending",
                            )
                        )
                        tui.set_spinner_message(f"tool detected: {tool_call.name}")
                    elif provider_event.type == "reasoning_item" and provider_event.reasoning_item:
                        summary = provider_event.reasoning_item.get("summary")
                        if isinstance(summary, list):
                            summary_text = " ".join(str(x) for x in summary if x)
                        else:
                            summary_text = str(summary or "reasoning")
                        tui.push_item(StatusItem(text=f"reasoning: {summary_text}"))
                        tui.set_spinner_message("reasoning...")
                    elif provider_event.type == "status" and provider_event.status:
                        tui.push_item(StatusItem(text=provider_event.status))
                        tui.set_spinner_message(provider_event.status)
                    elif provider_event.type == "error" and provider_event.error:
                        tui.report_error(provider_event.error)
                        tui.reset_spinner_message()
                elif event.type == "tool_execution_start" and event.tool_call:
                    _text_buffer = ""
                    _suppress_streaming = False
                    tool_call = event.tool_call
                    tui.push_item(
                        ToolCallItem(
                            id=tool_call.id,
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            status="pending",
                        )
                    )
                    tui.update_tool_status(tool_call.id, "running")
                    tui.set_spinner_message(f"running tool: {tool_call.name}")
                elif event.type == "tool_result" and event.tool_result:
                    tool_result = event.tool_result
                    status = "failed" if tool_result.is_error else "done"
                    tui.update_tool_status(tool_result.tool_call_id, status)
                    tui.push_item(
                        ToolResultItem(
                            id=tool_result.tool_call_id,
                            name=tool_result.tool_name or "tool",
                            content=tool_result.content,
                            status=status,
                        )
                    )
                    tui.reset_spinner_message()
                elif event.type == "validation_error" and event.message:
                    tui.report_error(event.message)
                    tui.reset_spinner_message()
                elif event.type == "structured_action" and event.message:
                    _text_buffer = ""
                    _suppress_streaming = False
                    tui.replace_last_assistant(event.message)
                elif event.type == "context_trimmed" and event.message:
                    tui.report_info(event.message)

            runtime = AgentRuntime(
                provider=provider,
                tool_definitions=tool_definitions,
                allowed_prefixes=allowed_prefixes,
                request_permission=tui.ask_permission,
                save_allowed_prefixes=save_allowed,
                on_event=handle_runtime_event,
                context_strategy=context_strategy,
            )
            result = runtime.run_user_turn(messages, text)
            messages = result.messages
            if messages and isinstance(messages[-1], AssistantMessage) and messages[-1].text:
                display_text = _display_text_for_assistant_message(messages[-1])
                if display_text != messages[-1].text:
                    tui.replace_last_assistant(display_text)
            tui.update_tokens(result.input_tokens)
            save_session(cwd, messages, active_session)

        except Exception as e:  # pylint: disable=broad-except
            messages = messages_checkpoint
            tui.report_error(str(e))

    tui.on_submit = on_submit
    if restored:
        for item in messages_to_items(messages):
            tui.push_item(item)
    tui.run()


def _run_headless(args) -> None:
    """Run the agent in headless mode with structured JSON I/O."""
    load_dotenv()

    model = args.model or os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4-6")
    max_tokens = int(os.getenv("MAX_TOKENS", "8192"))
    max_turns = args.max_turns
    auto_approve = args.auto_approve
    session_name = args.session or f"headless-{os.getpid()}"
    no_session = args.no_session
    context_strategy: ContextStrategy = args.context_strategy

    valid_strategies: set[ContextStrategy] = {"window", "compact", "compact+window"}
    if context_strategy not in valid_strategies:
        _headless_fatal(f"Invalid context strategy: {context_strategy}")

    tool_definitions = find_available_tools()
    cwd, git_status, top_level_structure = gather_initial_context()
    system = build_system_prompt(
        working_directory=cwd,
        git_status=git_status,
        top_level_file_structure=top_level_structure,
    )
    provider = providers.get_provider(model=model, system=system, max_tokens=max_tokens)

    allowed_prefixes = load_allowed()

    def emit_event(event_dict: dict) -> None:
        sys.stderr.write(json.dumps(event_dict) + "\n")
        sys.stderr.flush()

    def headless_request_permission(command: str) -> str:
        if auto_approve:
            emit_event({"type": "permission_auto_approved", "command": command})
            return "allow"
        emit_event({"type": "permission_denied", "command": command})
        return "deny"

    if no_session:
        messages: list[Message] = []
    else:
        saved = load_session(cwd, session_name)
        messages = saved if saved is not None else []

    def handle_event(event: RuntimeEvent) -> None:
        if event.type == "provider_event" and event.provider_event:
            pe = event.provider_event
            if pe.type == "text_delta" and pe.text:
                emit_event({"type": "text_delta", "text": pe.text})
            elif pe.type == "tool_call_detected" and pe.tool_call:
                tc = pe.tool_call
                emit_event({"type": "tool_call_detected", "id": tc.id, "name": tc.name, "arguments": tc.arguments})
            elif pe.type == "reasoning_item" and pe.reasoning_item:
                emit_event({"type": "reasoning", "item": pe.reasoning_item})
            elif pe.type == "status" and pe.status:
                emit_event({"type": "status", "message": pe.status})
            elif pe.type == "error" and pe.error:
                emit_event({"type": "error", "message": pe.error})
        elif event.type == "tool_execution_start" and event.tool_call:
            tc = event.tool_call
            emit_event({"type": "tool_execution_start", "id": tc.id, "name": tc.name, "arguments": tc.arguments})
        elif event.type == "tool_result" and event.tool_result:
            tr = event.tool_result
            emit_event({
                "type": "tool_result",
                "id": tr.tool_call_id,
                "name": tr.tool_name,
                "content": tr.content[:2000],
                "is_error": tr.is_error,
            })
        elif event.type == "validation_error" and event.message:
            emit_event({"type": "validation_error", "message": event.message})
        elif event.type == "structured_action" and event.message:
            emit_event({"type": "structured_action", "message": event.message})
        elif event.type == "context_trimmed" and event.message:
            emit_event({"type": "context_trimmed", "message": event.message})

    def run_one_turn(prompt: str, turn_number: int) -> dict:
        nonlocal messages
        emit_event({"type": "turn_start", "turn": turn_number, "prompt": prompt[:500]})

        tool_calls_log: list[dict] = []

        def capturing_handler(event: RuntimeEvent) -> None:
            handle_event(event)
            if event.type == "tool_execution_start" and event.tool_call:
                tool_calls_log.append({
                    "id": event.tool_call.id,
                    "name": event.tool_call.name,
                    "arguments": event.tool_call.arguments,
                })
            if event.type == "tool_result" and event.tool_result:
                for tc in reversed(tool_calls_log):
                    if tc["id"] == event.tool_result.tool_call_id:
                        tc["result_preview"] = event.tool_result.content[:500]
                        tc["is_error"] = event.tool_result.is_error
                        break

        try:
            runtime = AgentRuntime(
                provider=provider,
                tool_definitions=tool_definitions,
                allowed_prefixes=allowed_prefixes,
                request_permission=headless_request_permission,
                save_allowed_prefixes=save_allowed if not no_session else None,
                on_event=capturing_handler,
                context_strategy=context_strategy,
            )
            result = runtime.run_user_turn(messages, prompt)
            messages = result.messages

            assistant_text = None
            if messages and isinstance(messages[-1], AssistantMessage):
                assistant_text = _display_text_for_assistant_message(messages[-1])

            if not no_session:
                save_session(cwd, messages, session_name)

            emit_event({"type": "turn_complete", "turn": turn_number, "input_tokens": result.input_tokens})

            return {
                "success": True,
                "turn": turn_number,
                "input_tokens": result.input_tokens,
                "assistant_text": assistant_text,
                "session": session_name if not no_session else None,
                "messages_count": len(messages),
                "tool_calls": tool_calls_log,
                "error": None,
            }
        except Exception as e:  # pylint: disable=broad-except
            emit_event({"type": "error", "message": str(e)})
            return {
                "success": False,
                "turn": turn_number,
                "input_tokens": 0,
                "assistant_text": None,
                "session": session_name if not no_session else None,
                "messages_count": len(messages),
                "tool_calls": tool_calls_log,
                "error": str(e),
            }

    # Single-turn mode
    if args.prompt is not None:
        result = run_one_turn(args.prompt, 1)
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
        sys.stdout.flush()
        sys.exit(0 if result["success"] else 1)

    # Multi-turn stdin mode
    if sys.stdin.isatty():
        sys.stderr.write(
            "Error: headless mode requires --prompt or piped stdin.\n"
            "Usage: python -m avoid_agent headless --prompt '...'\n"
            "   or: echo '{\"prompt\": \"...\"}' | python -m avoid_agent headless\n"
        )
        sys.exit(1)

    turn = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            emit_event({"type": "error", "message": f"Invalid JSON input: {e}"})
            continue

        if request.get("command") == "quit":
            emit_event({"type": "session_end", "turns": turn})
            break

        prompt = request.get("prompt")
        if not prompt or not isinstance(prompt, str):
            emit_event({"type": "error", "message": "Input must have a 'prompt' string field"})
            continue

        turn += 1
        if turn > max_turns:
            emit_event({"type": "error", "message": f"Max turns ({max_turns}) exceeded"})
            error_result = {"success": False, "error": f"Max turns ({max_turns}) exceeded", "turn": turn}
            sys.stdout.write(json.dumps(error_result) + "\n")
            sys.stdout.flush()
            break

        result = run_one_turn(prompt, turn)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()


def _headless_fatal(message: str) -> None:
    """Print a fatal error as JSON to stderr and exit."""
    sys.stderr.write(json.dumps({"type": "fatal", "message": message}) + "\n")
    sys.exit(1)


def _run_selfdev(args) -> None:
    """Run the self-improvement loop."""
    from dotenv import load_dotenv

    from avoid_agent.selfdev.loop import run_loop

    load_dotenv()
    repo_root = Path(__file__).resolve().parent.parent
    model = args.model or os.getenv("DEFAULT_MODEL")
    if getattr(args, "legacy", False):
        exit_code = run_loop(
            repo_root=repo_root,
            model=model,
            max_turns=args.max_turns,
            single=args.single,
        )
    else:
        exit_code = _run_selfdev_interactive(
            repo_root=repo_root,
            model=model,
            max_turns=args.max_turns,
        )
    sys.exit(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser(prog="avoid-agent")
    subparsers = parser.add_subparsers(dest="command")

    prompt_parser = subparsers.add_parser("prompt", help="Prompt development utilities")
    prompt_subparsers = prompt_parser.add_subparsers(dest="prompt_command")

    export_parser = prompt_subparsers.add_parser("export", help="Export system prompt to markdown")
    export_parser.add_argument(
        "--out",
        default="./system-prompt.md",
        help="Output markdown file path (default: ./system-prompt.md)",
    )

    headless_parser = subparsers.add_parser(
        "headless", help="Run agent in headless mode for programmatic use"
    )
    headless_parser.add_argument(
        "--prompt", type=str, default=None, help="Single-turn prompt text"
    )
    headless_parser.add_argument(
        "--session", type=str, default=None, help="Session name for persistence"
    )
    headless_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve all bash commands",
    )
    headless_parser.add_argument(
        "--model", type=str, default=None, help="Provider/model (e.g. anthropic/claude-sonnet-4-6)"
    )
    headless_parser.add_argument(
        "--max-turns", type=int, default=20, help="Max turns in multi-turn stdin mode"
    )
    headless_parser.add_argument(
        "--context-strategy",
        type=str,
        default="compact+window",
        help="Context management strategy",
    )
    headless_parser.add_argument(
        "--no-session",
        action="store_true",
        help="Don't persist session (ephemeral run)",
    )

    selfdev_parser = subparsers.add_parser(
        "selfdev", help="Run the self-improvement loop"
    )
    selfdev_parser.add_argument(
        "--model", type=str, default=None,
        help="Provider/model for headless agent (e.g. anthropic/claude-sonnet-4-6)",
    )
    selfdev_parser.add_argument(
        "--max-turns", type=int, default=40,
        help="Max turns per headless run (default: 40)",
    )
    selfdev_parser.add_argument(
        "--single", action="store_true",
        help="Run only one cycle then exit (don't loop)",
    )
    selfdev_parser.add_argument(
        "--legacy", action="store_true",
        help="Use legacy non-interactive selfdev loop output",
    )

    args = parser.parse_args()

    if args.command == "prompt" and args.prompt_command == "export":
        _export_prompt_command(args.out)
        return

    if args.command == "headless":
        _run_headless(args)
        return

    if args.command == "selfdev":
        _run_selfdev(args)
        return

    _run_agent()


if __name__ == "__main__":
    main()
