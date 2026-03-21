"""Tool functions that the agent can use."""

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
