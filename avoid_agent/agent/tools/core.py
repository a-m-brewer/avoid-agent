"""Tool functions that the agent can use."""

import subprocess
import os

from typing_extensions import Annotated

from avoid_agent.agent.tools import tool


@tool
def read_file(path: Annotated[str, "The path to the file to read"]) -> str:
    """Read the contents of a file at the given path."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as e:
        return f"Error: {e}"

@tool
def write_file(
    path: Annotated[str, "Path to the file."],
    content: Annotated[str, "Content to write."],
) -> str:
    """Write content to a file at the given path. Creates the file if it doesn't exist, overwrites if it does."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written to {path}"
    except OSError as e:
        return f"Error: {e}"


@tool
def edit_file(
    path: Annotated[str, "Path to the file."],
    old_string: Annotated[str, "The exact string to replace. Must be unique in the file."],
    new_string: Annotated[str, "The string to replace it with."],
) -> str:
    """Replace an exact string in a file with new content. Use for surgical edits - read the file first to get the exact string. The old_string must appear exactly once in the file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {path}"
        if count > 1:
            return f"Error: old_string appears {count} times - make it more specific"

        with open(path, "w", encoding="utf-8") as f:
            f.write(content.replace(old_string, new_string, 1))

        return f"Edit applied to {path}"
    except OSError as e:
        return f"Error: {e}"


@tool
def run_bash(command: Annotated[str, "The bash command to run."]) -> str:
    """Run a bash command and return stdout and stderr. Use for running tests, installing packages, checking git status, compiling code, etc."""
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
