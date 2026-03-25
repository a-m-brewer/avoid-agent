"""Sample extension tool for searching Python files."""

import subprocess

from typing_extensions import Annotated

from avoid_agent.agent.tools import ToolRunResult, tool


@tool
def file_search(
    pattern: Annotated[str, "Pattern to search for."],
    directory: Annotated[str, "Directory to search in."] = ".",
) -> ToolRunResult:
    """Search Python files for a text pattern."""
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", pattern, directory],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return ToolRunResult(content=f"Unable to run grep: {exc}")

    if result.returncode == 1:
        return ToolRunResult(content=f"No Python matches found for '{pattern}' in {directory}.")

    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip() or "Unknown grep error."
        return ToolRunResult(
            content=f"Search failed with exit code {result.returncode}: {error_text}"
        )

    lines = result.stdout.splitlines()
    limited_lines = lines[:50]
    output = "\n".join(limited_lines)
    if len(lines) > 50:
        output += f"\n... ({len(lines) - 50} more matches omitted)"

    return ToolRunResult(content=output)
