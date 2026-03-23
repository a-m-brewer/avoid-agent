"""Main entry point for the Avoid Agent CLI."""

import os
import subprocess

from dotenv import load_dotenv

from avoid_agent.agent.tools.finder import find_available_tools
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    ToolResultMessage,
    UserMessage,
)
from avoid_agent import providers
from avoid_agent.agent.tools import run_tool
from avoid_agent.permissions import command_prefix, load_allowed, save_allowed
from avoid_agent.session import delete_session, load_session, save_session
from avoid_agent.prompts import build_system_prompt
from avoid_agent.tui import TUI
from avoid_agent.tui.components.conversation import (
    AssistantItem,
    ConversationItem,
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
                items.append(ToolCallItem(name=tc.name, arguments=tc.arguments))
        elif isinstance(msg, ToolResultMessage):
            name = tool_name_map.get(msg.tool_call_id, "tool")
            items.append(ToolResultItem(name=name, content=msg.content))
    return items


def gather_initial_context_messages() -> list[Message]:
    return [
        UserMessage(text="Ready for your first task."),
        AssistantMessage(text="Understood. Ready.", tool_calls=[]),
    ]


def main():
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
    saved = load_session(cwd)
    if saved is not None:
        messages = saved
        restored = True
    else:
        messages = gather_initial_context_messages()
        restored = False

    tui = TUI(model=default_model, on_submit=lambda _: None)

    def on_submit(text: str) -> None:
        nonlocal messages

        if text.strip() == "/clear":
            messages = gather_initial_context_messages()
            delete_session(cwd)
            tui.clear_conversation()
            return

        messages_checkpoint = messages[:]
        messages.append(UserMessage(text=text))

        try:
            while True:
                with provider.stream(messages=messages, tools=tool_definitions) as stream:
                    for chunk in stream.text_stream():
                        tui.append_chunk(chunk)
                    response = stream.get_final_message()
                    tui.update_tokens(response.input_tokens)

                messages.append(response.message)

                if response.stop:
                    if response.input_tokens > CONTEXT_LIMIT * COMPACTION_THRESHOLD:
                        messages = provider.compact(messages, keep_last=6)
                    save_session(cwd, messages)
                    break

                for tc in response.message.tool_calls:
                    tui.push_item(ToolCallItem(name=tc.name, arguments=tc.arguments))

                    if tc.name == "run_bash":
                        cmd = tc.arguments.get("command", "")
                        prefix = command_prefix(cmd)
                        if prefix not in allowed_prefixes:
                            decision = tui.ask_permission(cmd)
                            if decision == "deny":
                                result = "User denied this command."
                                messages.append(ToolResultMessage(tool_call_id=tc.id, content=result))
                                tui.push_item(ToolResultItem(name=tc.name, content=result))
                                continue
                            if decision == "save":
                                allowed_prefixes.add(prefix)
                                save_allowed(allowed_prefixes)

                    result = run_tool(tc.name, tc.arguments)
                    messages.append(ToolResultMessage(tool_call_id=tc.id, content=result))
                    tui.push_item(ToolResultItem(name=tc.name, content=result))

        except Exception as e:  # pylint: disable=broad-except
            messages = messages_checkpoint
            tui.report_error(str(e))

    tui.on_submit = on_submit
    if restored:
        for item in messages_to_items(messages):
            tui.push_item(item)
    tui.run()


if __name__ == "__main__":
    main()
