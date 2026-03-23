"""Main entry point for the Avoid Agent CLI."""

import argparse
import os
import subprocess

from dotenv import load_dotenv

from avoid_agent.agent.runtime import AgentRuntime, RuntimeEvent, _looks_like_hallucinated_completion
from avoid_agent.agent.tools.finder import find_available_tools
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from avoid_agent import providers
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

CONTEXT_LIMIT = 200_000
COMPACTION_THRESHOLD = 0.75  # compact at 75% full


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


def format_initial_context(working_directory: str, git_status: str, top_level_structure: str) -> str:
    """Format initial context as a user message payload."""
    return (
        f"Working directory: {working_directory}\n\n"
        f"Git status:\n{git_status}\n\n"
        f"Top-level file structure:\n{top_level_structure}"
    )


def messages_to_items(messages: list[Message]) -> list[ConversationItem]:
    """Reconstruct TUI conversation items from a saved message list.

    Skips the first two messages (initial context exchange) and maps
    tool_call_id back to tool names via the preceding AssistantMessage.
    """
    # Build id -> name lookup from all assistant tool calls
    tool_name_map: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, AssistantMessage):
            for tc in msg.tool_calls:
                tool_name_map[tc.id] = tc.name

    items: list[ConversationItem] = []
    # Skip the first two messages (initial context + "Understood. Ready.")
    for msg in messages[2:]:
        if isinstance(msg, UserMessage):
            items.append(UserItem(text=msg.text))
        elif isinstance(msg, AssistantMessage):
            if msg.text:
                items.append(AssistantItem(text=msg.text))
            for tc in msg.tool_calls:
                items.append(ToolCallItem(id=tc.id, name=tc.name, arguments=tc.arguments))
        elif isinstance(msg, ToolResultMessage):
            name = tool_name_map.get(msg.tool_call_id, "tool")
            items.append(ToolResultItem(id=msg.tool_call_id, name=name, content=msg.content))
    return items


def gather_initial_context_messages() -> list[Message]:
    return [
        UserMessage(text="Ready for your first task."),
        AssistantMessage(text="Understood. Ready.", tool_calls=[]),
    ]


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
    default_model = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4-6")
    max_tokens = int(os.getenv("MAX_TOKENS", "8192"))
    tool_definitions = find_available_tools()

    cwd, git_status, top_level_structure = gather_initial_context()
    system = build_system_prompt(
        working_directory=cwd,
        git_status=git_status,
        top_level_file_structure=top_level_structure,
    )

    provider = providers.get_provider(
        model=default_model, system=system, max_tokens=max_tokens
    )

    allowed_prefixes = load_allowed()
    active_session = "default"
    saved = load_session(cwd, active_session)
    if saved is not None:
        messages = saved
        restored = True
    else:
        messages = gather_initial_context_messages()
        restored = False

    tui = TUI(model=default_model, on_submit=lambda _: None)

    def on_submit(text: str) -> None:
        nonlocal messages, active_session

        if text.strip() == "/clear":
            messages = gather_initial_context_messages()
            delete_session(cwd, active_session)
            tui.clear_conversation()
            return

        if text.strip() == "/compact":
            try:
                messages = provider.compact(messages, keep_last=6)
                save_session(cwd, messages, active_session)
                tui.clear_conversation()
                for item in messages_to_items(messages):
                    tui.push_item(item)
                tui.report_info("Conversation compacted.")
            except Exception as e:  # pylint: disable=broad-except
                tui.report_error(f"Compact failed: {e}")
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
            def handle_runtime_event(event: RuntimeEvent) -> None:
                if event.type == "provider_event" and event.provider_event:
                    provider_event = event.provider_event
                    if provider_event.type == "text_delta" and provider_event.text:
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
                elif event.type == "tool_execution_start" and event.tool_call:
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
                elif event.type == "validation_error" and event.message:
                    tui.report_error(event.message)

            runtime = AgentRuntime(
                provider=provider,
                tool_definitions=tool_definitions,
                allowed_prefixes=allowed_prefixes,
                request_permission=tui.ask_permission,
                save_allowed_prefixes=save_allowed,
                on_event=handle_runtime_event,
            )
            result = runtime.run_user_turn(messages, text)
            messages = result.messages
            tui.update_tokens(result.input_tokens)

            if result.input_tokens > CONTEXT_LIMIT * COMPACTION_THRESHOLD:
                messages = provider.compact(messages, keep_last=6)
            save_session(cwd, messages, active_session)

        except Exception as e:  # pylint: disable=broad-except
            messages = messages_checkpoint
            tui.report_error(str(e))

    tui.on_submit = on_submit
    if restored:
        for item in messages_to_items(messages):
            tui.push_item(item)
    tui.run()


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

    args = parser.parse_args()

    if args.command == "prompt" and args.prompt_command == "export":
        _export_prompt_command(args.out)
        return

    _run_agent()


if __name__ == "__main__":
    main()
