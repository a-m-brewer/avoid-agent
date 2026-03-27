"""Backlog parsing and mutation for the selfdev workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacklogItem:
    """A single item from the backlog."""
    line_number: int
    text: str
    raw_line: str


def parse_backlog(repo_root: Path) -> list[BacklogItem]:
    """Parse backlog.md and return unchecked items in order.

    Args:
        repo_root: Path to the repository root containing backlog.md

    Returns:
        List of unchecked BacklogItems in the order they appear in the file.
        Returns an empty list if backlog.md doesn't exist or has no unchecked items.
    """
    backlog_path = repo_root / "backlog.md"
    if not backlog_path.exists():
        return []

    items: list[BacklogItem] = []
    with open(backlog_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, start=1):
        match = re.match(r"^- \[ \] (.+)$", line.strip())
        if match:
            items.append(BacklogItem(
                line_number=i,
                text=match.group(1).strip(),
                raw_line=line,
            ))

    return items


def mark_backlog_item(repo_root: Path, item: BacklogItem, status: str, note: str = "") -> None:
    """Mark a backlog item as done [x] or failed [!].

    Args:
        repo_root: Path to the repository root containing backlog.md
        item: The BacklogItem to mark
        status: Either "done" or "failed"
        note: Optional note to append for failed items
    """
    backlog_path = repo_root / "backlog.md"
    with open(backlog_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if status == "done":
        marker = "[x]"
    elif status == "failed":
        marker = "[!]"
    else:
        return

    idx = item.line_number - 1
    if 0 <= idx < len(lines):
        lines[idx] = lines[idx].replace("[ ]", marker, 1)
        if note and status == "failed":
            lines[idx] = lines[idx].rstrip() + f" <!-- {note} -->\n"

    with open(backlog_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
