"""Self-development loop for avoid-agent.

Reads the backlog, picks the next task, creates a git worktree,
runs the agent in headless mode to implement the task, validates
the result, and merges back to main.

This module is FROZEN - the agent must never modify it.
"""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path

from avoid_agent.selfdev import RESTART_EXIT_CODE
from avoid_agent.selfdev.validate import validate_worktree
from avoid_agent.selfdev.workflow import (
    BacklogItem,
    cleanup_worktree,
    commit_if_dirty,
    create_worktree,
    detach_worktree,
    merge_worktree,
    parse_backlog,
    run_agent_headless,
)
from avoid_agent.selfdev.workflow.backlog import mark_backlog_item
from avoid_agent.selfdev.workflow.prompt_builder import build_prompt_for_task

log = logging.getLogger(__name__)


def run_one_cycle(
    repo_root: Path,
    model: str | None = None,
    max_turns: int = 40,
) -> str:
    """Run one self-improvement cycle.

    This function:
    1. Parses the backlog and picks the first unchecked item
    2. Creates a worktree for the task
    3. Runs the agent headless to implement the task
    4. Validates the result
    5. Merges back to main on success, or preserves the branch on failure

    Args:
        repo_root: Path to the repository root
        model: Optional model override for the headless agent
        max_turns: Maximum number of turns per agent run

    Returns:
        "restart" - changes merged, supervisor should restart
        "done" - no more backlog items
        "failed" - task failed but branch preserved for review
        "error" - unexpected error
    """
    items = parse_backlog(repo_root)
    if not items:
        log.info("No unchecked backlog items. Nothing to do.")
        return "done"

    item = items[0]
    log.info(f"Picked task: {item.text}")

    branch_name = f"selfdev/{re.sub(r'[^a-z0-9]+', '-', item.text.lower())[:50].strip('-')}"
    worktree_path: Path | None = None

    try:
        # Create worktree
        log.info(f"Creating worktree on branch: {branch_name}")
        worktree_path = create_worktree(repo_root, branch_name)

        # Run the agent
        prompt = build_prompt_for_task(item.text, repo_root, worktree_path)
        result = run_agent_headless(
            worktree_path, prompt, model=model, max_turns=max_turns,
        )

        if not result.get("success"):
            error = result.get("error", "unknown error")
            log.error(f"Agent failed: {error}")
            # Commit any partial work before preserving the branch
            if worktree_path:
                commit_if_dirty(worktree_path, item.text)
            mark_backlog_item(repo_root, item, "failed",
                              note=f"{error[:80]} | branch: {branch_name}")
            detach_worktree(repo_root, branch_name)
            return "failed"

        log.info(f"Agent completed. Tool calls: {len(result.get('tool_calls', []))}")

        # Check if the agent actually made changes
        diff_result = subprocess.run(
            ["git", "diff", "--stat", "main", "HEAD"],
            capture_output=True, text=True, cwd=worktree_path, check=False,
        )
        if not diff_result.stdout.strip():
            # Agent may not have committed — check working tree
            if not commit_if_dirty(worktree_path, item.text):
                log.error("No changes were made. Marking as failed.")
                mark_backlog_item(repo_root, item, "failed", note="no changes produced")
                cleanup_worktree(repo_root, branch_name)
                return "failed"

        # Validate
        log.info("Running validation...")
        validation = validate_worktree(repo_root, worktree_path)
        log.info(f"Validation {'PASSED' if validation.passed else 'FAILED'}:")
        print(validation.summary, flush=True)

        if not validation.passed:
            log.error("Validation failed. Branch preserved for review.")
            mark_backlog_item(repo_root, item, "failed",
                              note=f"validation failed | branch: {branch_name}")
            detach_worktree(repo_root, branch_name)
            return "failed"

        # Merge back to main
        log.info("Merging to main...")
        if not merge_worktree(repo_root, branch_name):
            mark_backlog_item(repo_root, item, "failed",
                              note=f"merge conflict | branch: {branch_name}")
            detach_worktree(repo_root, branch_name)
            return "failed"

        mark_backlog_item(repo_root, item, "done")
        cleanup_worktree(repo_root, branch_name)
        log.info(f"Task completed and merged: {item.text}")
        return "restart"

    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        if worktree_path:
            commit_if_dirty(worktree_path, item.text)
            mark_backlog_item(repo_root, item, "failed",
                              note=f"{str(e)[:80]} | branch: {branch_name}")
            detach_worktree(repo_root, branch_name)
        else:
            mark_backlog_item(repo_root, item, "failed", note=str(e)[:100])
        return "error"


def run_loop(
    repo_root: Path,
    model: str | None = None,
    max_turns: int = 40,
    single: bool = False,
) -> int:
    """Run the self-improvement loop.

    Args:
        repo_root: Path to the repository root.
        model: Model to use for headless agent.
        max_turns: Max turns per headless run.
        single: If True, run only one cycle then exit.

    Returns:
        Exit code (42 = restart requested, 0 = done, 1 = error).
    """
    log.info("Self-improvement loop starting")
    log.info(f"Repo: {repo_root}")
    log.info(f"Model: {model or 'default'}")
    log.info(f"Single mode: {single}")
    print("", flush=True)

    result = run_one_cycle(repo_root, model=model, max_turns=max_turns)

    if result == "restart":
        log.info("Requesting supervisor restart...")
        return RESTART_EXIT_CODE
    elif result == "done":
        log.info("All backlog items complete.")
        return 0
    elif result == "failed":
        if single:
            log.info("Task failed in single mode. Exiting.")
            return 1
        # In loop mode, try next item
        log.info("Task failed. Trying next item...")
        next_result = run_one_cycle(repo_root, model=model, max_turns=max_turns)
        if next_result == "restart":
            return RESTART_EXIT_CODE
        elif next_result == "done":
            return 0
        else:
            log.info("Second task also failed. Exiting to avoid infinite failure loop.")
            return 1
    else:
        log.error("Unexpected result. Exiting.")
        return 1


# Re-export for backward compatibility
_commit_if_dirty = commit_if_dirty


def __getattr__(name: str):
    if name == "BacklogItem":
        from avoid_agent.selfdev.workflow.backlog import BacklogItem
        return BacklogItem
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
