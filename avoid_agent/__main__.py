"""Main entry point for the Avoid Agent CLI."""

import argparse
import os
import re
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
from avoid_agent.prompts import build_system_prompt, export_system_prompt_markdown
from avoid_agent.prompts.system_prompt import SystemPromptOptions
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

# Patterns that indicate the model claims it performed file modifications.
# We require either a first-person subject ("I changed") or a sentence-start
# past tense ("Implemented the...") to avoid matching passive descriptions
# like "needs to be refactored".
_COMPLETION_CLAIM_RE = re.compile(
    r"("
    r"(?:^|\.\s+)(?:Implemented|Applied|Edited|Changed|Modified"
    r"|Updated|Patched|Wired|Refactored|Rewrote)"
    r"\s+(?:the|a|all|both|this|that|it|now|properly|correctly"
    r"|successfully|in|to|into|across|with)"
    r"|I\s+(?:changed|edited|modified|updated|applied|wired"
    r"|patched|rewrote|implemented|refactored)"
    r"|✅\s*(?:I\s|What\s|Changed)"
    r")",
    re.MULTILINE,
)

MAX_HALLUCINATION_RETRIES = 2

_HALLUCINATION_CORRECTION = (
    "[SYSTEM] Your previous response claimed to have made file changes, "
    "but no tool calls were executed. Text responses cannot modify files. "
    "You MUST call edit_file, write_file, or run_bash to make changes. "
    "Do not describe changes as complete until a tool result confirms them. "
    "Please use your tools now to perform the work."
)


def _looks_like_hallucinated_completion(message: AssistantMessage) -> bool:
    """Detect when the model claims it made edits but issued no tool calls."""
    if message.tool_calls:
        return False
    if not message.text:
        return False
    return bool(_COMPLETION_CLAIM_RE.search(message.text))


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

        hallucination_retries = 0

        try:
            while True:
                with provider.stream(messages=messages, tools=tool_definitions) as stream:
                    for event in stream.event_stream():
                        if event.type == "text_delta" and event.text:
                            tui.append_chunk(event.text)
                        elif event.type == "tool_call_detected" and event.tool_call:
                            tc_ev = event.tool_call
                            tui.push_item(ToolCallItem(
                                id=tc_ev.id,
                                name=tc_ev.name,
                                arguments=tc_ev.arguments,
                            ))
                    response = stream.get_final_message()
                    tui.update_tokens(response.input_tokens)

                messages.append(response.message)

                if response.stop:
                    if (
                        _looks_like_hallucinated_completion(response.message)
                        and hallucination_retries < MAX_HALLUCINATION_RETRIES
                    ):
                        hallucination_retries += 1
                        messages.append(
                            UserMessage(text=_HALLUCINATION_CORRECTION)
                        )
                        tui.report_error(
                            "Hallucination detected: claimed edits "
                            "without tool calls. Re-prompting..."
                        )
                        continue

                    if response.input_tokens > CONTEXT_LIMIT * COMPACTION_THRESHOLD:
                        messages = provider.compact(messages, keep_last=6)
                    save_session(cwd, messages)
                    break

                for tc in response.message.tool_calls:
                    tui.push_item(ToolCallItem(id=tc.id, name=tc.name, arguments=tc.arguments))

                    if tc.name == "run_bash":
                        cmd = tc.arguments.get("command", "")
                        prefix = command_prefix(cmd)
                        if prefix not in allowed_prefixes:
                            decision = tui.ask_permission(cmd)
                            if decision == "deny":
                                result = "User denied this command."
                                messages.append(ToolResultMessage(tool_call_id=tc.id, content=result))
                                tui.push_item(ToolResultItem(id=tc.id, name=tc.name, content=result))
                                continue
                            if decision == "save":
                                allowed_prefixes.add(prefix)
                                save_allowed(allowed_prefixes)

                    result = run_tool(tc.name, tc.arguments)
                    messages.append(ToolResultMessage(tool_call_id=tc.id, content=result))
                    tui.push_item(ToolResultItem(id=tc.id, name=tc.name, content=result))

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
