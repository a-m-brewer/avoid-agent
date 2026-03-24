"""Main entry point for the Avoid Agent CLI."""

import argparse
import json
import os
import subprocess
import sys

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
from avoid_agent.providers import list_available_models, get_saved_model, save_selected_model
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

    active_model = default_model

    def build_provider(model: str):
        return providers.get_provider(
            model=model,
            system=system,
            max_tokens=max_tokens,
        )

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

    tui = TUI(model=active_model, on_submit=lambda _: None)

    def on_submit(text: str) -> None:
        nonlocal messages, active_session, context_strategy, provider, active_model

        if text.strip() == "/strategy":
            tui.report_info(
                f"Current strategy: {context_strategy}\n"
                f"Options: {', '.join(sorted(valid_strategies))}\n"
                f"Use /strategy <name> to switch."
            )
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

    args = parser.parse_args()

    if args.command == "prompt" and args.prompt_command == "export":
        _export_prompt_command(args.out)
        return

    if args.command == "headless":
        _run_headless(args)
        return

    _run_agent()


if __name__ == "__main__":
    main()
