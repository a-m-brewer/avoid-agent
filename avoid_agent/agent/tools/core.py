"""Tool functions that the agent can use."""

import importlib.util
import logging
import os
import subprocess
from difflib import unified_diff
from hashlib import sha256
from pathlib import Path

from typing_extensions import Annotated

from avoid_agent.agent.tools import ToolRunResult, tool

logger = logging.getLogger(__name__)

_PREVIEW_LIMIT = 1200
_BASH_QUIET_THRESHOLD = 20
_DIFF_LIMIT = 4000


def _preview(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _slice_lines(content: str, start_line: int | None, limit: int | None) -> tuple[str, int, int, int, bool]:
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)

    if total_lines == 0:
        if start_line not in (None, 1):
            raise ValueError("start_line out of range for empty file")
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1")
        return "", 1, 0, 0, False

    if start_line is None:
        start_idx = 0
    else:
        if start_line < 1:
            raise ValueError("start_line must be >= 1")
        if start_line > total_lines:
            raise ValueError(f"start_line {start_line} exceeds total lines {total_lines}")
        start_idx = start_line - 1

    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")

    if limit is None:
        end_idx = total_lines
    else:
        end_idx = min(start_idx + limit, total_lines)

    selected = "".join(lines[start_idx:end_idx])
    actual_start = start_idx + 1
    actual_end = end_idx
    truncated = actual_start != 1 or actual_end != total_lines
    return selected, actual_start, actual_end, total_lines, truncated


def _replace_line_range(
    content: str,
    start_line: int,
    end_line: int,
    replacement: str,
) -> tuple[str, int, int, int]:
    lines = content.splitlines(keepends=True)
    total_lines = len(lines)

    if start_line < 1 or end_line < 1:
        raise ValueError("start_line and end_line must be >= 1")
    if start_line > end_line:
        raise ValueError("start_line must be <= end_line")
    if total_lines == 0:
        raise ValueError("Cannot apply a line-range edit to an empty file")
    if end_line > total_lines:
        raise ValueError(f"end_line {end_line} exceeds total lines {total_lines}")

    start_idx = start_line - 1
    end_idx = end_line
    updated_lines = [*lines[:start_idx], replacement, *lines[end_idx:]]
    return "".join(updated_lines), start_line, end_line, total_lines


def _diff_preview(path: str, before: str, after: str) -> str:
    diff = "".join(
        unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
        )
    )
    return _preview(diff, _DIFF_LIMIT)


def _load_tool_module(module_path: Path) -> None:
    module_name = f"avoid_agent_extensions_{module_path.stem}_{abs(hash(module_path.resolve()))}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        logger.warning("Skipping extension module '%s': unable to load module spec", module_path)
        return

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Failed to load extension module '%s'", module_path)


def _extension_directories() -> list[Path]:
    configured = os.environ.get("AVOID_AGENT_EXTENSIONS_DIRS")
    if configured:
        return [Path(path).expanduser() for path in configured.split(os.pathsep) if path]
    return [Path(__file__).resolve().parents[3] / "extensions"]


def _discover_extension_tools() -> None:
    for extensions_dir in _extension_directories():
        if not extensions_dir.is_dir():
            continue

        for child in sorted(extensions_dir.iterdir()):
            if child.is_file() and child.suffix == ".py" and child.name != "__init__.py":
                _load_tool_module(child)
                continue

            if child.is_dir() and (child / "__init__.py").is_file():
                _load_tool_module(child / "__init__.py")


_discover_extension_tools()


@tool
def read_file(
    path: Annotated[str, "The path to the file to read"],
    start_line: Annotated[int, "Optional 1-based starting line for a partial read."] | None = None,
    limit: Annotated[int, "Optional maximum number of lines to return."] | None = None,
) -> str:
    """Read a file. For large files, prefer partial reads with start_line and limit."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        selected, actual_start, actual_end, total_lines, truncated = _slice_lines(
            content, start_line, limit
        )
        return ToolRunResult(
            content=selected,
            details={
                "proof": {
                    "kind": "file_read",
                    "path": path,
                    "sha256": _sha256(selected),
                    "chars": len(selected),
                    "total_lines": total_lines,
                    "start_line": actual_start,
                    "end_line": actual_end,
                    "truncated": truncated,
                    "preview": _preview(selected),
                }
            },
        )
    except ValueError as e:
        return ToolRunResult(content=f"Error: {e}")
    except OSError as e:
        return ToolRunResult(content=f"Error: {e}")


@tool
def write_file(
    path: Annotated[str, "Path to the file."],
    content: Annotated[str, "Content to write."],
) -> str:
    """Write content to a file at the given path. Creates the file if it doesn't exist, overwrites if it does."""
    try:
        target = Path(path)
        before_exists = target.exists()
        before_content = target.read_text(encoding="utf-8") if before_exists else ""

        with open(target, "w", encoding="utf-8") as f:
            f.write(content)
        after_content = target.read_text(encoding="utf-8")
        changed = (not before_exists) or before_content != after_content
        return ToolRunResult(
            content=f"Written to {path}",
            details={
                "proof": {
                    "kind": "file_change",
                    "path": path,
                    "before_exists": before_exists,
                    "after_exists": target.exists(),
                    "changed": changed,
                    "before_sha256": _sha256(before_content) if before_exists else None,
                    "after_sha256": _sha256(after_content),
                    "expected_sha256": _sha256(content),
                    "after_preview": _preview(after_content),
                    "diff_preview": _diff_preview(path, before_content, after_content),
                }
            },
        )
    except OSError as e:
        return ToolRunResult(content=f"Error: {e}")


@tool
def edit_file(
    path: Annotated[str, "Path to the file."],
    old_string: Annotated[str, "The exact string to replace. Must be unique in the file."] | None = None,
    new_string: Annotated[str, "The string to replace it with."] | None = None,
    start_line: Annotated[int, "Optional 1-based start line for range replacement."] | None = None,
    end_line: Annotated[int, "Optional 1-based end line for range replacement."] | None = None,
    replacement: Annotated[str, "Replacement text for range replacement mode."] | None = None,
) -> str:
    """Edit an existing file. Use either exact-string replacement or line-range replacement for surgical edits."""
    try:
        target = Path(path)
        with open(target, "r", encoding="utf-8") as f:
            content = f.read()

        string_mode = old_string is not None or new_string is not None
        range_mode = start_line is not None or end_line is not None or replacement is not None

        if string_mode and range_mode:
            return ToolRunResult(
                content="Error: edit_file accepts either string replacement or line-range replacement, not both"
            )
        if not string_mode and not range_mode:
            return ToolRunResult(
                content="Error: edit_file requires either old_string/new_string or start_line/end_line/replacement"
            )

        edit_mode: str
        affected_start_line: int | None = None
        affected_end_line: int | None = None

        if range_mode:
            if start_line is None or end_line is None or replacement is None:
                return ToolRunResult(
                    content="Error: line-range replacement requires start_line, end_line, and replacement"
                )
            updated, affected_start_line, affected_end_line, total_lines = _replace_line_range(
                content, start_line, end_line, replacement
            )
            edit_mode = "line_range"
        else:
            if old_string is None or new_string is None:
                return ToolRunResult(
                    content="Error: string replacement requires old_string and new_string"
                )

            count = content.count(old_string)
            if count == 0:
                return ToolRunResult(content=f"Error: old_string not found in {path}")
            if count > 1:
                return ToolRunResult(
                    content=f"Error: old_string appears {count} times - make it more specific"
                )

            updated = content.replace(old_string, new_string, 1)
            total_lines = content.count("\n") + (0 if not content else 1)
            edit_mode = "string"
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)

        after_content = target.read_text(encoding="utf-8")
        changed = content != after_content
        return ToolRunResult(
            content=f"Edit applied to {path}",
            details={
                "proof": {
                    "kind": "file_change",
                    "path": path,
                    "before_exists": True,
                    "after_exists": target.exists(),
                    "changed": changed,
                    "before_sha256": _sha256(content),
                    "after_sha256": _sha256(after_content),
                    "edit_mode": edit_mode,
                    "total_lines": total_lines,
                    "start_line": affected_start_line,
                    "end_line": affected_end_line,
                    "after_preview": _preview(after_content),
                    "diff_preview": _diff_preview(path, content, after_content),
                }
            },
        )
    except ValueError as e:
        return ToolRunResult(content=f"Error: {e}")
    except OSError as e:
        return ToolRunResult(content=f"Error: {e}")


@tool
def run_bash(command: Annotated[str, "The bash command to run."]) -> str:
    """Run a bash command and return stdout and stderr. Use for running tests, installing packages, checking git status, compiling code, etc."""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
        check=False,
    )

    if result.returncode == 0:
        stdout_lines = result.stdout.count("\n")
        if stdout_lines > _BASH_QUIET_THRESHOLD:
            display_stdout = (
                f"(success, {stdout_lines} lines of output suppressed — re-run with verbose flag if needed)"
            )
        else:
            display_stdout = result.stdout
    else:
        display_stdout = result.stdout

    parts = [f"Exit code: {result.returncode}"]
    if display_stdout:
        parts.append(f"STDOUT:\n{display_stdout}")
    if result.stderr:
        parts.append(f"STDERR:\n{result.stderr}")
    if not result.stdout and not result.stderr:
        parts.append("(no output)")
    return ToolRunResult(
        content="\n\n".join(part.rstrip() for part in parts if part),
        details={
            "proof": {
                "kind": "command",
                "command": command,
                "cwd": os.getcwd(),
                "exit_code": result.returncode,
                "stdout_preview": _preview(result.stdout),
                "stderr_preview": _preview(result.stderr),
            }
        },
    )
