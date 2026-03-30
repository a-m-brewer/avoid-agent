"""TUI mode: interactive terminal UI for avoid-agent.

This module extracts the TUI logic from __main__.py into a clean entry point.
"""

from __future__ import annotations

import threading
from pathlib import Path

from dotenv import load_dotenv

from avoid_agent import providers
from avoid_agent.agent.context import ContextStrategy
from avoid_agent.agent.runtime import AgentRuntime, RuntimeEvent, TurnCancelledError, _parse_structured_action
from avoid_agent.agent.tools.finder import find_available_tools
from avoid_agent.cli.shared import gather_initial_context
from avoid_agent.infra.config import DEFAULT_MODEL, MAX_TOKENS, CONTEXT_STRATEGY, env_flag
from avoid_agent.permissions import load_allowed, save_allowed
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    list_available_models,
)
from avoid_agent.providers import get_saved_model, save_selected_model, load_user_config, save_user_config
from avoid_agent.tui import TUI
from avoid_agent.tui.components.conversation import (
    AssistantItem,
    ConversationItem,
    StatusItem,
    ToolCallItem,
    ToolResultItem,
    UserItem,
)
from avoid_agent.prompts import build_system_prompt
from avoid_agent.session import delete_session, list_sessions, load_session, save_session


def messages_to_items(messages: list[Message]) -> list[ConversationItem]:
    """Reconstruct TUI conversation items from a saved message list."""
    tool_name_map: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for tc in msg.tool_calls:
                tool_name_map[tc.id] = tc.name

    items: list[ConversationItem] = []
    for msg in messages:
        if isinstance(msg, Message):
            from avoid_agent.providers import ToolResultMessage
            if hasattr(msg, "text") and getattr(msg, "text", None):
                if isinstance(msg, AssistantMessage):
                    items.append(AssistantItem(text=_display_text_for_assistant_message(msg)))
                elif hasattr(msg, "tool_call_id"):
                    # This is a tool result
                    tool_name = tool_name_map.get(msg.tool_call_id, "tool")
                    items.append(ToolResultItem(
                        id=msg.tool_call_id,
                        name=tool_name,
                        content=msg.content if hasattr(msg, "content") else "",
                    ))
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                items.append(ToolCallItem(id=tc.id, name=tc.name, arguments=tc.arguments))
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


def run(_args=None) -> None:
    """Run the agent in interactive TUI mode.

    This is the main entry point for the `avoid-agent tui` command.
    The _args parameter is ignored for backward compatibility.
    """
    load_dotenv()
    default_model = get_saved_model() or DEFAULT_MODEL
    max_tokens = MAX_TOKENS
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
        tui.set_vision_enabled(provider.supports_vision)
        return provider

    provider = build_provider(active_model)

    valid_strategies: set[ContextStrategy] = {"window", "compact", "compact+window"}
    env_strategy = CONTEXT_STRATEGY
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

    def on_submit(text: str, images: list | None = None) -> None:
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

        if text.strip().startswith("/image "):
            # Load an image from a file path and queue it for the next message.
            path = text.strip()[len("/image "):].strip()
            if not path:
                tui.report_info("Usage: /image <path/to/image.png>")
                return
            if not provider.supports_vision:
                tui.report_info(
                    f"Warning: {active_model} does not support vision. "
                    "Image will be queued but the model may ignore it."
                )
            tui._try_load_image_from_path(path)
            return

        if text.strip().startswith("/learnings"):
            from avoid_agent.learnings import capture_session
            from avoid_agent.learnings_analyzer import analyze
            from pathlib import Path as PathLib

            stripped = text.strip()
            learnings_dir = PathLib(cwd) / ".learnings" / "sessions"

            if stripped == "/learnings clear":
                removed_count = _clear_learning_session_files(learnings_dir)
                tui.report_info(f"Cleared {removed_count} learning session file(s).")
                return

            if stripped == "/learnings suggest":
                suggestions = analyze(learnings_dir)
                tui.report_info(_format_learning_suggestions(suggestions))
                return

            if stripped == "/learnings":
                tui.report_info(_build_learnings_report(learnings_dir))
                return

            tui.report_error("Usage: /learnings [clear|suggest]")
            return

        if text.strip().startswith("/self-improve"):
            from avoid_agent.selfdev.loop import parse_backlog, run_one_cycle

            repo_root = Path(__file__).resolve().parent.parent
            items = parse_backlog(repo_root)

            if not items:
                tui.report_info("No pending backlog items")
                return

            tui.report_info(f"Next task: {items[0].text}")

            worker_done = threading.Event()
            worker_result: dict[str, str] = {"result": "error"}

            def worker() -> None:
                try:
                    result = run_one_cycle(repo_root, model=active_model)
                    worker_result["result"] = result
                except Exception as e:  # pylint: disable=broad-except
                    worker_result["result"] = "error"
                    worker_result["error"] = str(e)
                finally:
                    worker_done.set()

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            tui.report_info("Running selfdev cycle...")

            def poll_worker() -> None:
                while not worker_done.wait(timeout=0.1):
                    pass
                result = worker_result["result"]
                if result == "restart":
                    tui.report_info("Changes merged and ready. A restart is recommended.")
                elif result == "done":
                    tui.report_info("All backlog items completed!")
                elif result in ("failed", "error"):
                    error_msg = worker_result.get("error", "Cycle failed. Branch preserved for review.")
                    tui.report_info(f"Cycle failed. Branch preserved for review.")
                    if error_msg and error_msg != "error":
                        tui.report_error(error_msg)
                else:
                    tui.report_error(f"Unexpected result: {result}")

            threading.Thread(target=poll_worker, daemon=True).start()
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
            # Warn if images are attached but model doesn't support vision.
            if images and not provider.supports_vision:
                tui.report_info(
                    f"Warning: {active_model} does not support vision. Images will be ignored."
                )
            result = runtime.run_user_turn(messages, text, cancel_token=tui.cancel_token, images=images)
            messages = result.messages
            if messages and isinstance(messages[-1], AssistantMessage) and messages[-1].text:
                display_text = _display_text_for_assistant_message(messages[-1])
                if display_text != messages[-1].text:
                    tui.replace_last_assistant(display_text)
            tui.update_tokens(result.input_tokens)
            save_session(cwd, messages, active_session)

        except TurnCancelledError:
            # User cancelled mid-flight (e.g. typed /quit or another slash command
            # while the LLM was still running).  Restore the message history to its
            # pre-turn state so no partial assistant content is saved.
            messages = messages_checkpoint
            tui.report_info("Turn cancelled.")

        except Exception as e:  # pylint: disable=broad-except
            messages = messages_checkpoint
            tui.report_error(str(e))

    tui.on_submit = on_submit
    if restored:
        for item in messages_to_items(messages):
            tui.push_item(item)
    tui.run()


# Learnings helper functions (extracted from __main__.py)
def _clear_learning_session_files(learnings_dir: Path) -> int:
    """Clear all learning session files."""
    removed_count = 0
    if learnings_dir.exists() and learnings_dir.is_dir():
        for session_file in learnings_dir.glob("*.md"):
            if session_file.is_file():
                session_file.unlink()
                removed_count += 1
    return removed_count


def _format_learning_suggestions(suggestions: list[str]) -> str:
    """Format learnings suggestions for display."""
    if not suggestions:
        return "No suggestions yet."
    return "Suggestions:\n- " + "\n- ".join(suggestions)


def _build_learnings_report(learnings_dir: Path) -> str:
    """Build a report of learnings sessions."""
    session_files = _list_learning_session_files(learnings_dir)
    recent_errors = _extract_recent_learning_errors(session_files, limit=3)

    report = [f"Learnings sessions: {len(session_files)}"]
    if recent_errors:
        report.append("Recent errors:\n- " + "\n- ".join(recent_errors))
    else:
        report.append("Recent errors: none")
    return "\n".join(report)


def _list_learning_session_files(learnings_dir: Path) -> list[Path]:
    """List learning session files sorted by modification time."""
    if not learnings_dir.exists() or not learnings_dir.is_dir():
        return []
    return sorted(
        (path for path in learnings_dir.glob("*.md") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _extract_recent_learning_errors(session_files: list[Path], limit: int = 3) -> list[str]:
    """Extract recent errors from learning session files."""
    recent_errors: list[str] = []
    for session_file in session_files:
        in_errors_section = False
        for line in session_file.read_text(encoding="utf-8").splitlines():
            stripped_line = line.strip()
            if not in_errors_section:
                if stripped_line.lower() == "## errors":
                    in_errors_section = True
                continue
            if stripped_line.startswith("## "):
                break
            if stripped_line:
                recent_errors.append(stripped_line)
            if len(recent_errors) >= limit:
                return recent_errors[:limit]
    return recent_errors[:limit]
