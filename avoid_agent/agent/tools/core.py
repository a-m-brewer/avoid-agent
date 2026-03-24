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
_DIFF_LIMIT = 4000


def _preview(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


def _sha256(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


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
def read_file(path: Annotated[str, "The path to the file to read"]) -> str:
    """Read the contents of a file at the given path."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return ToolRunResult(
            content=content,
            details={
                "proof": {
                    "kind": "file_read",
                    "path": path,
                    "sha256": _sha256(content),
                    "chars": len(content),
                    "preview": _preview(content),
                }
            },
        )
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
    old_string: Annotated[str, "The exact string to replace. Must be unique in the file."],
    new_string: Annotated[str, "The string to replace it with."],
) -> str:
    """Replace an exact string in a file with new content. Use for surgical edits - read the file first to get the exact string. The old_string must appear exactly once in the file."""
    try:
        target = Path(path)
        with open(target, "r", encoding="utf-8") as f:
            content = f.read()

        count = content.count(old_string)
        if count == 0:
            return ToolRunResult(content=f"Error: old_string not found in {path}")
        if count > 1:
            return ToolRunResult(
                content=f"Error: old_string appears {count} times - make it more specific"
            )

        updated = content.replace(old_string, new_string, 1)
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
                    "after_preview": _preview(after_content),
                    "diff_preview": _diff_preview(path, content, after_content),
                }
            },
        )
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
    parts = [f"Exit code: {result.returncode}"]
    if result.stdout:
        parts.append(f"STDOUT:\n{result.stdout}")
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
