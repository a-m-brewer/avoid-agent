"""Prompt building for the selfdev workflow.

This module contains the logic for building prompts that are sent to the
headless agent for implementing self-improvement tasks.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import logging

log = logging.getLogger(__name__)


def _load_yaml_policy(policy_path: Path) -> dict:
    """Load and parse a YAML policy file.

    Args:
        policy_path: Path to the YAML file

    Returns:
        Parsed YAML content as a dict, or empty dict if file doesn't exist or parsing fails
    """
    try:
        import yaml
        with open(policy_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:  # pylint: disable=broad-except
        return {}


def _load_frozen_patterns(repo_root: Path) -> list[str]:
    """Load frozen file patterns from selfdev-policy.yaml.

    Frozen files are those that the agent must never modify.

    Args:
        repo_root: Path to the repository root

    Returns:
        List of frozen file patterns, or empty list if no policy exists
    """
    policy_path = repo_root / "selfdev-policy.yaml"
    if not policy_path.exists():
        return []
    policy = _load_yaml_policy(policy_path)
    return policy.get("frozen", [])


def _load_allowed_patterns(repo_root: Path) -> list[str]:
    """Load allowed file patterns from selfdev-policy.yaml.

    Allowed files are those that the agent may modify.

    Args:
        repo_root: Path to the repository root

    Returns:
        List of allowed file patterns (including unrestricted), or empty list if no policy exists
    """
    policy_path = repo_root / "selfdev-policy.yaml"
    if not policy_path.exists():
        return []
    policy = _load_yaml_policy(policy_path)
    return policy.get("allowed", []) + policy.get("unrestricted", [])


def _gather_completed_tasks(repo_root: Path) -> list[str]:
    """Return descriptions of already-completed backlog items.

    This helps the agent understand what's already been done.

    Args:
        repo_root: Path to the repository root

    Returns:
        List of descriptions of completed tasks
    """
    import re
    backlog_path = repo_root / "backlog.md"
    if not backlog_path.exists():
        return []
    completed = []
    with open(backlog_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^- \[x\] (.+)$", line.strip())
            if match:
                completed.append(match.group(1).strip())
    return completed


def _gather_recent_selfdev_commits(repo_root: Path, limit: int = 10) -> str:
    """Return recent selfdev merge commits for context.

    Args:
        repo_root: Path to the repository root
        limit: Maximum number of commits to return

    Returns:
        Formatted string of recent selfdev commit messages, or empty string if none found
    """
    result = subprocess.run(
        ["git", "log", "--oneline", f"--grep=selfdev:", f"-{limit}"],
        capture_output=True, text=True, cwd=repo_root, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _gather_learnings(repo_root: Path, limit: int = 5) -> str:
    """Return the most recent learnings entries, if any.

    Controlled by SELFDEV_INCLUDE_LEARNINGS env var (default: false for token savings).

    Args:
        repo_root: Path to the repository root
        limit: Maximum number of learning files to include

    Returns:
        Formatted learnings content, or empty string if none available or disabled
    """
    # Default to NOT including learnings to save tokens (~1500-2500 per worker run)
    if not os.getenv("SELFDEV_INCLUDE_LEARNINGS", "").lower() in ("1", "true", "yes"):
        return ""

    learnings_dir = repo_root / ".learnings" / "sessions"
    if not learnings_dir.exists():
        return ""
    files = sorted(learnings_dir.glob("*.md"), reverse=True)[:limit]
    parts = []
    for f in files:
        content = f.read_text(encoding="utf-8").strip()
        if content:
            parts.append(f"### {f.stem}\n{content[:500]}")
    return "\n\n".join(parts)


def _gather_file_tree(worktree_path: Path) -> str:
    """Return a compact file tree of the worktree (excluding noise).

    Args:
        worktree_path: Path to the worktree

    Returns:
        Formatted file tree string, or empty string if command fails
    """
    result = subprocess.run(
        "find . -maxdepth 2 "
        "! -path './.venv/*' ! -path './.git' ! -path './.git/*' "
        "! -path '*/__pycache__/*' ! -path './.worktrees/*' "
        "! -path './avoid_agent.egg-info/*' "
        "| head -80",
        shell=True, capture_output=True, text=True,
        cwd=worktree_path, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _format_frozen_allowed(frozen: list[str], allowed: list[str]) -> str:
    """Format frozen and allowed file patterns for the prompt.

    Args:
        frozen: List of frozen file patterns
        allowed: List of allowed file patterns

    Returns:
        Formatted string for inclusion in the prompt
    """
    frozen_text = "\n".join(f"  - `{p}`" for p in frozen) if frozen else "  (none)"
    allowed_text = "\n".join(f"  - `{p}`" for p in allowed) if allowed else "  (none)"
    return frozen_text, allowed_text


def build_prompt_for_task(task_text: str, repo_root: Path, worktree_path: Path) -> str:
    """Build the headless prompt with project context for a backlog task.

    This function assembles a comprehensive prompt that includes:
    - Critical file modification rules (frozen/allowed files)
    - The task description
    - Project structure snapshot
    - History of completed tasks
    - Recent selfdev commits
    - Learnings from prior sessions
    - General instructions

    Args:
        task_text: The task description from the backlog
        repo_root: Path to the repository root
        worktree_path: Path to the worktree where the task will be implemented

    Returns:
        A complete prompt string for the headless agent
    """
    sections: list[str] = []

    # CRITICAL constraints first — frozen files
    frozen = _load_frozen_patterns(repo_root)
    allowed = _load_allowed_patterns(repo_root)
    frozen_text, allowed_text = _format_frozen_allowed(frozen, allowed)

    sections.append(
        "## CRITICAL: File modification rules\n"
        "Your changes will be AUTOMATICALLY REJECTED if you modify any frozen file.\n"
        "This has happened before and wasted an entire run. DO NOT touch these files.\n\n"
        "**FROZEN (do NOT read, modify, create, or overwrite these):**\n"
        f"{frozen_text}\n\n"
        "**ALLOWED (you may only modify files matching these patterns):**\n"
        f"{allowed_text}\n\n"
        "If you need functionality from a frozen file, use it as-is. Do not copy, "
        "recreate, or modify it."
    )

    # Task
    sections.append(
        f"You are working on a self-improvement task for the avoid-agent project.\n\n"
        f"## Task\n{task_text}"
    )

    # Project snapshot
    file_tree = _gather_file_tree(worktree_path)
    if file_tree:
        sections.append(f"## Project structure\n```\n{file_tree}\n```")

    # Completed tasks (so the agent knows what's already been done)
    completed = _gather_completed_tasks(repo_root)
    if completed:
        completed_text = "\n".join(f"- {t}" for t in completed)
        sections.append(
            f"## Already completed selfdev tasks\n"
            f"These have already been merged to main. Build on them, don't redo them.\n"
            f"{completed_text}"
        )

    # Recent selfdev commits
    commits = _gather_recent_selfdev_commits(repo_root)
    if commits:
        sections.append(f"## Recent selfdev commits\n```\n{commits}\n```")

    # Learnings from prior sessions
    learnings = _gather_learnings(repo_root)
    if learnings:
        sections.append(
            f"## Learnings from prior sessions\n"
            f"These are patterns and issues observed in earlier runs. Avoid repeating them.\n"
            f"{learnings}"
        )

    # Instructions
    sections.append(
        "## Instructions\n"
        "- You are running autonomously. Do NOT ask the user for feedback, clarification, "
        "or approval. Make your best judgment and proceed.\n"
        "- Do NOT present options or ask 'how would you like me to...' — just do it. "
        "If unsure about a design decision, pick the simplest approach that works.\n"
        "- Read the relevant code before making changes\n"
        "- Make minimal, focused changes\n"
        "- REMINDER: Any modification to frozen files (listed above) will cause "
        "AUTOMATIC REJECTION of all your work. This includes avoid_agent/selfdev/*, "
        "supervisor.sh, selfdev-policy.yaml, .env, and .env.example\n"
        "- Only modify files that match the ALLOWED patterns listed above\n"
        "- After making changes, verify they work by reading the modified files\n"
        "- If you cannot complete the task, explain why"
    )

    return "\n\n".join(sections)
