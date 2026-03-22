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

CONTEXT_LIMIT = 200_000
COMPACTION_THRESHOLD = 0.75  # compact at 75% full

system = """You are a coding assistant with access to the local filesystem and shell.

You have 4 tools: read_file, write_file, edit_file, and run_bash.

When given a task:
- Explore before acting. Read relevant files before making changes.
- Prefer edit_file for targeted changes. Use write_file only for new files or full rewrites.
- Use run_bash for searching (grep, find), running tests, git, and anything else shell-related.
- If a bash command might be destructive, explain what it will do first.
- Report results clearly — what you did, what you found, what changed.

Never guess file paths — use run_bash with ls or find to discover them first.

You love haikus. Finish each response with one — evocative, not literal."""


def gather_initial_context():
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
        "find . -maxdepth 2 ! -path './.venv/*' ! -path './.git' ! -path './.git/*' ! -path '*/__pycache__/*'", shell=True, capture_output=True, text=True, cwd=cwd
    )
    return f"Working directory: {cwd}\n\nGit status:\n{git_output}\n\nTop-level file structure:\n{top_level_structure.stdout}"


def gather_initial_context_messages() -> list[Message]:
    return [
        UserMessage(text=gather_initial_context()),
        AssistantMessage(text="Understood. Ready.", tool_calls=[]),
    ]


def main():
    load_dotenv()
    default_model = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4-6")
    max_tokens = int(os.getenv("MAX_TOKENS", "8192"))
    tool_definitions = find_available_tools()

    # This is the atom. Understand this before anything else.
    provider = providers.get_provider(
        model=default_model,
        system=system,
        max_tokens=max_tokens
    )

    # The agent's memory of the conversation.
    # It knows nothing other than what is in this list.
    # Curate it for what you want the agent to know and remember.
    messages = gather_initial_context_messages()

    while True:
        # User's turn: get input and add to messages
        user_input = ""
        while not user_input.strip():
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting.")
                return
            if not user_input.strip():
                print("  (Please enter a message, or type 'exit' to quit.)")

        if user_input.strip() == "/clear":
            messages = gather_initial_context_messages()
            print("  (Conversation cleared.)")
            continue

        messages.append(UserMessage(text=user_input))

        # Agent's turn loop: keep responding until it ends its turn or uses a tool
        while True:
            with provider.stream(messages=messages,
                                 tools=tool_definitions) as stream:
                # Stream the response for a better user experience. The final message will be the same as the last streamed message, but we want to print it as it comes in.
                for chunk in stream.text_stream():
                    print(chunk, end="", flush=True)
                response = stream.get_final_message()

            messages.append(response.message)

            if response.stop:
                print()
                print("=== Assistant ended turn ===")

                if response.input_tokens > CONTEXT_LIMIT * COMPACTION_THRESHOLD:
                    print("  (Compacting conversation to save memory.)")
                    messages = provider.compact(messages, keep_last=6)
                    print("  (Conversation compacted.)")
                break

            for tc in response.message.tool_calls:
                result = run_tool(tc.name, tc.arguments)
                messages.append(ToolResultMessage(tool_call_id=tc.id, content=result))


if __name__ == "__main__":
    main()
