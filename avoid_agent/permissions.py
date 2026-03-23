"""Persistent permission store for run_bash commands."""

import json
from pathlib import Path

_PERMS_PATH = Path.home() / ".avoid-agent" / "permissions.json"


def load_allowed() -> set[str]:
    if not _PERMS_PATH.exists():
        return set()
    try:
        return set(json.loads(_PERMS_PATH.read_text()).get("allowed_prefixes", []))
    except Exception:  # pylint: disable=broad-except
        return set()


def save_allowed(prefixes: set[str]) -> None:
    _PERMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PERMS_PATH.write_text(json.dumps({"allowed_prefixes": sorted(prefixes)}, indent=2))


def command_prefix(command: str) -> str:
    """Return the first word of a command (the program being run)."""
    return command.strip().split()[0] if command.strip() else ""
