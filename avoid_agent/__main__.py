import os
import anthropic
import subprocess

from dotenv import load_dotenv

# What does the agent have access to? Define tools here.
tools = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The path to the file to read."
                }
            },
            "required": ["path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file at the given path. Creates the file if it doesn't exist, overwrites if it does.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "content": {"type": "string", "description": "Content to write."}
            },
            "required": ["path", "content"]
        }
    },
    {
        "name": "edit_file",
        "description": "Replace an exact string in a file with new content. Use for surgical edits — read the file first to get the exact string. The old_string must appear exactly once in the file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "old_string": {"type": "string", "description": "The exact string to replace. Must be unique in the file."},
                "new_string": {"type": "string", "description": "The string to replace it with."}
            },
            "required": ["path", "old_string", "new_string"]
        }
    },
    {
        "name": "run_bash",
        "description": "Run a bash command and return stdout and stderr. Use for running tests, installing packages, checking git status, compiling code, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run."
                }
            },
            "required": ["command"]
        }
    }
]

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

# Implementation of the tools.
def run_tool(name, tool_input):
    print(f"Running tool: {name} with input: {tool_input}")

    if name == "read_file":
        path = tool_input["path"]
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"

    if name == "write_file":
        try:
            path = tool_input["path"]
            content = tool_input["content"]
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written to {path}"
        except Exception as e:
            return f"Error: {e}"

    if name == "edit_file":
        path = tool_input["path"]
        old_string = tool_input["old_string"]
        new_string = tool_input["new_string"]
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            count = content.count(old_string)
            if count == 0:
                return f"Error: old_string not found in {path}"
            if count > 1:
                return f"Error: old_string appears {count} times — make it more specific"
            with open(path, "w", encoding="utf-8") as f:
                f.write(content.replace(old_string, new_string, 1))
            return f"Edit applied to {path}"
        except Exception as e:
            return f"Error: {e}"

    if name == "run_bash":
        command = tool_input["command"]
        print(f"\n  $ {command}")
        confirm = input("  Run this? [y/N]: ").strip().lower()
        if confirm != "y":
            return "User denied this command."
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            check=False,
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        return output or "(no output)"

    return f"Unknown tool: {name}"

def gather_initial_context():
    cwd = os.getcwd()
    git_status = subprocess.run("git status --short", shell=True, capture_output=True, text=True, cwd=cwd)
    git_output = git_status.stdout.strip() if git_status.returncode == 0 else "Not a git repository"
    top_level_structure = subprocess.run("find . -maxdepth 2", shell=True, capture_output=True, text=True, cwd=cwd)
    return f'Working directory: {cwd}\n\nGit status:\n{git_output}\n\nTop-level file structure:\n{top_level_structure.stdout}'

def main():
    load_dotenv()
    default_model = os.getenv("DEFAULT_MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.getenv("MAX_TOKENS", "8192"))

    # This is the atom. Understand this before anything else.
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    # The agent's memory of the conversation.
    # It knows nothing other than what is in this list.
    # Curate it for what you want the agent to know and remember.
    messages = [
        {"role": "user", "content": gather_initial_context()},
        {"role": "assistant", "content": "Understood. Ready."}
    ]

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
            messages = [
                {"role": "user", "content": gather_initial_context()},
                {"role": "assistant", "content": "Understood. Ready."}
            ]
            print("  (Conversation cleared.)")
            continue

        messages.append({"role": "user", "content": user_input})

        # Agent's turn loop: keep responding until it ends its turn or uses a tool
        while True:
            with client.messages.stream(             
                model=default_model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages) as stream:
                # Stream the response for a better user experience. The final message will be the same as the last streamed message, but we want to print it as it comes in.
                for text in stream.text_stream:
                    print(text, end="", flush=True)
                response = stream.get_final_message()

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == 'end_turn':
                print()
                print('=== Assistant ended turn ===')
                break

            elif response.stop_reason == 'tool_use':
                print()
                tool_results = []

                for block in response.content:
                    if block.type == 'tool_use':
                        result = run_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })

                messages.append({
                    "role": "user",
                    "content": tool_results
                })

            else:
                print(f"Unexpected stop reason: {response.stop_reason}")
                break

if __name__ == "__main__":
    main()
