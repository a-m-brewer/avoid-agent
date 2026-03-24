"""Main entry point for the Avoid Agent CLI."""

import argparse
import os
import subprocess

from dotenv import load_dotenv

from avoid_agent.agent.context import ContextStrategy
from avoid_agent.agent.runtime import AgentRuntime, RuntimeEvent
from avoid_agent.agent.tools.finder import find_available_tools
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from avoid_agent import providers
from avoid_agent.providers import list_available_models
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
                items.append(AssistantItem(text=msg.text))
            for tc in msg.tool_calls:
                items.append(ToolCallItem(id=tc.id, name=tc.name, arguments=tc.arguments))
        elif isinstance(msg, ToolResultMessage):
            name = tool_name_map.get(msg.tool_call_id, "tool")
            items.append(ToolResultItem(id=msg.tool_call_id, name=name, content=msg.content))
    return items


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
            parts = text.strip().split()
            if len(parts) == 1:
                picked = tui.pick_from_list("Select model", list_available_models())
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
                        tui.reset_spinner_message()
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
                    tui.reset_spinner_message()
                elif event.type == "validation_error" and event.message:
                    tui.report_error(event.message)
                    tui.reset_spinner_message()
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
