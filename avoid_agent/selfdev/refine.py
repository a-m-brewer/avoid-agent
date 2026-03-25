"""Refinement system for breaking broad backlog items into smaller sub-tasks.

Manages refined task files in the `refined/` directory. Each file corresponds
to a backlog item and contains a checklist of sub-tasks the worker can execute.

This module is FROZEN - the agent must never modify it during self-improvement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SubTask:
    """A single sub-task within a refined task file."""
    line_number: int
    text: str
    raw_line: str
    status: str  # "pending", "done", "failed"
    parent_path: Path = field(repr=False)


@dataclass
class RefinedTask:
    """A refined task file with its metadata and sub-tasks."""
    path: Path
    source: str
    source_line: int
    status: str  # "pending", "in-progress", "done", "failed"
    subtasks: list[SubTask] = field(default_factory=list)


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse simple YAML-style frontmatter from a refined task file."""
    result: dict[str, str] = {}
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return result
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def parse_refined_file(path: Path) -> RefinedTask | None:
    """Parse a single refined task file and return a RefinedTask."""
    if not path.exists():
        return None

    text = path.read_text(encoding="utf-8")
    fm = _parse_frontmatter(text)

    source = fm.get("source", "")
    try:
        source_line = int(fm.get("source_line", "0"))
    except ValueError:
        source_line = 0
    status = fm.get("status", "pending")

    subtasks: list[SubTask] = []
    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        unchecked = re.match(r"^- \[ \] (.+)$", line.strip())
        if unchecked:
            subtasks.append(SubTask(
                line_number=i,
                text=unchecked.group(1).strip(),
                raw_line=line,
                status="pending",
                parent_path=path,
            ))
            continue
        done = re.match(r"^- \[x\] (.+)$", line.strip())
        if done:
            subtasks.append(SubTask(
                line_number=i,
                text=done.group(1).strip(),
                raw_line=line,
                status="done",
                parent_path=path,
            ))
            continue
        failed = re.match(r"^- \[!\] (.+)$", line.strip())
        if failed:
            subtasks.append(SubTask(
                line_number=i,
                text=failed.group(1).strip(),
                raw_line=line,
                status="failed",
                parent_path=path,
            ))

    return RefinedTask(
        path=path,
        source=source,
        source_line=source_line,
        status=status,
        subtasks=subtasks,
    )


def parse_refined_tasks(repo_root: Path) -> list[RefinedTask]:
    """Scan refined/ directory for all refined task files, ordered by name."""
    refined_dir = repo_root / "refined"
    if not refined_dir.is_dir():
        return []

    tasks = []
    for path in sorted(refined_dir.glob("*.md")):
        task = parse_refined_file(path)
        if task:
            tasks.append(task)
    return tasks


def find_next_subtask(repo_root: Path) -> SubTask | None:
    """Return the first pending sub-task from any in-progress refined file.

    Priority: files with status 'in-progress' first, then 'pending'.
    """
    tasks = parse_refined_tasks(repo_root)

    # First pass: in-progress files
    for task in tasks:
        if task.status == "in-progress":
            for st in task.subtasks:
                if st.status == "pending":
                    return st

    # Second pass: pending files (auto-promote to in-progress)
    for task in tasks:
        if task.status == "pending":
            for st in task.subtasks:
                if st.status == "pending":
                    _update_frontmatter_status(task.path, "in-progress")
                    return st

    return None


def mark_subtask(path: Path, line_number: int, status: str, note: str = "") -> None:
    """Mark a sub-task as done [x] or failed [!] in its refined task file."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

    if status == "done":
        marker = "[x]"
    elif status == "failed":
        marker = "[!]"
    else:
        return

    idx = line_number - 1
    if 0 <= idx < len(lines):
        lines[idx] = lines[idx].replace("[ ]", marker, 1)
        if note and status == "failed":
            lines[idx] = lines[idx].rstrip() + f" <!-- {note} -->\n"

    path.write_text("".join(lines), encoding="utf-8")


def update_refined_status(path: Path) -> str:
    """Update the refined file's frontmatter status based on sub-task completion.

    Returns the new status.
    """
    task = parse_refined_file(path)
    if not task:
        return "unknown"

    pending = sum(1 for st in task.subtasks if st.status == "pending")
    done = sum(1 for st in task.subtasks if st.status == "done")
    failed = sum(1 for st in task.subtasks if st.status == "failed")
    total = len(task.subtasks)

    if total == 0:
        new_status = "pending"
    elif done == total:
        new_status = "done"
    elif failed > 0 and pending == 0:
        new_status = "failed"
    elif done > 0 or failed > 0:
        new_status = "in-progress"
    else:
        new_status = "pending"

    if new_status != task.status:
        _update_frontmatter_status(path, new_status)

    return new_status


def _update_frontmatter_status(path: Path, new_status: str) -> None:
    """Update the status field in the frontmatter of a refined task file."""
    text = path.read_text(encoding="utf-8")
    updated = re.sub(
        r"^(status:\s*).+$",
        rf"\g<1>{new_status}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if updated != text:
        path.write_text(updated, encoding="utf-8")
