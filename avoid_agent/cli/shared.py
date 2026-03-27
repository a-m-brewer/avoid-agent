"""Shared CLI helpers used across multiple command modes.

These are pure functions with no side effects at the CLI layer.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from avoid_agent.providers import Message


def _is_worktree_context(cwd: str) -> bool:
    """Detect if running in a git worktree (for skipping redundant git operations)."""
    return ".worktrees" in cwd or "/.worktrees/" in cwd


def gather_initial_context() -> tuple[str, str, str]:
    """Collect runtime context used by the system prompt and initial conversation.

    Skips git status in worktree contexts to save tokens and avoid noise.
    """
    cwd = os.getcwd()

    # Skip git status in worktrees - the status is always "clean" or shows
    # uncommitted changes from the agent, which is not useful context
    if _is_worktree_context(cwd):
        git_output = "(worktree - git status skipped for token savings)"
    else:
        git_status = subprocess.run(
            "git status --short", shell=True, capture_output=True, text=True, cwd=cwd
        )
        git_output = (
            git_status.stdout.strip()
            if git_status.returncode == 0
            else "Not a git repository"
        )

    top_level_structure = subprocess.run(
        "find . -maxdepth 2 ! -path './.venv/*' ! -path './.git' ! -path './.git/*' ! -path '*/__pycache__/*'",
        shell=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return cwd, git_output, top_level_structure.stdout


def _count_backlog_totals(repo_root: Path) -> tuple[int, int]:
    """Return (completed, total) for markdown checklist items in backlog.md."""
    backlog_path = repo_root / "backlog.md"
    if not backlog_path.exists():
        return (0, 0)

    completed = 0
    total = 0
    with open(backlog_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if re.match(r"^- \[( |x|!)\] .+", line):
                total += 1
            if re.match(r"^- \[x\] .+", line):
                completed += 1
    return (completed, total)


def _count_backlog_status(repo_root: Path) -> tuple[int, int, int]:
    """Return (completed, pending, failed) counts from backlog.md."""
    backlog_path = repo_root / "backlog.md"
    if not backlog_path.exists():
        return (0, 0, 0)
    completed = pending = failed = 0
    with open(backlog_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if re.match(r"^- \[x\]", stripped):
                completed += 1
            elif re.match(r"^- \[ \]", stripped):
                pending += 1
            elif re.match(r"^- \[!\]", stripped):
                failed += 1
    return (completed, pending, failed)
