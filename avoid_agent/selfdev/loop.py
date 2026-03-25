"""Self-development loop for avoid-agent.

Reads the backlog, picks the next task, creates a git worktree,
runs the agent in headless mode to implement the task, validates
the result, and merges back to main.

This module is FROZEN - the agent must never modify it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from avoid_agent.selfdev import RESTART_EXIT_CODE
from avoid_agent.selfdev.validate import validate_worktree


@dataclass
class BacklogItem:
    line_number: int
    text: str
    raw_line: str


def parse_backlog(repo_root: Path) -> list[BacklogItem]:
    """Parse backlog.md and return unchecked items in order."""
    backlog_path = repo_root / "backlog.md"
    if not backlog_path.exists():
        return []

    items = []
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
    """Mark a backlog item as done [x] or failed [!]."""
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


def _branch_exists(repo_root: Path, branch_name: str) -> bool:
    """Return True when a local branch already exists."""
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def create_worktree(repo_root: Path, branch_name: str) -> Path:
    """Create a git worktree for the given branch.

    If the branch already exists (for example from a preserved failed run),
    re-attach that branch in a fresh worktree instead of trying to recreate it.
    """
    worktree_path = repo_root / ".worktrees" / branch_name
    if worktree_path.exists():
        shutil.rmtree(worktree_path)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean up stale metadata from paths that were manually removed.
    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    if _branch_exists(repo_root, branch_name):
        cmd = ["git", "worktree", "add", str(worktree_path), branch_name]
    else:
        cmd = ["git", "worktree", "add", "-b", branch_name, str(worktree_path), "main"]

    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return worktree_path


def cleanup_worktree(repo_root: Path, branch_name: str) -> None:
    """Remove a worktree and its branch entirely."""
    worktree_path = repo_root / ".worktrees" / branch_name
    if worktree_path.exists():
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_path), "--force"],
            cwd=repo_root, check=False, capture_output=True, text=True,
        )
    subprocess.run(
        ["git", "branch", "-D", branch_name],
        cwd=repo_root, check=False, capture_output=True, text=True,
    )


def detach_worktree(repo_root: Path, branch_name: str) -> None:
    """Remove the worktree directory but keep the branch for review."""
    worktree_path = repo_root / ".worktrees" / branch_name
    if worktree_path.exists():
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_path), "--force"],
            cwd=repo_root, check=False, capture_output=True, text=True,
        )
    log(f"Branch preserved for review: {branch_name}")
    log(f"  Inspect with: git log main..{branch_name}")
    log(f"  Diff with:    git diff main...{branch_name}")
    log(f"  Delete with:  git branch -D {branch_name}")


def merge_worktree(repo_root: Path, branch_name: str) -> bool:
    """Merge the worktree branch back to main. Returns True on success."""
    result = subprocess.run(
        ["git", "merge", "--no-ff", branch_name, "-m", f"selfdev: merge {branch_name}"],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        log(f"Merge failed: {result.stderr}")
        subprocess.run(
            ["git", "merge", "--abort"],
            cwd=repo_root, check=False, capture_output=True, text=True,
        )
        return False
    return True


def _stream_stderr(pipe, tool_count_ref: list[int]) -> None:
    """Read stderr lines from the agent process and log progress live."""
    for raw_line in pipe:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            etype = event.get("type", "")
            if etype == "tool_execution_start":
                tool_count_ref[0] += 1
                tool_name = event.get("name", "?")
                log(f"  [{tool_count_ref[0]}] tool: {tool_name}")
            elif etype == "turn_start":
                turn = event.get("turn", "?")
                log(f"  --- turn {turn} ---")
            elif etype == "turn_complete":
                turn = event.get("turn", "?")
                tokens = event.get("input_tokens", "?")
                log(f"  turn {turn} done ({tokens} input tokens)")
            elif etype in ("error", "fatal", "validation_error"):
                log(f"  [{etype}] {event.get('message', '')[:200]}")
        except json.JSONDecodeError:
            pass


def run_agent_headless(
    worktree_path: Path,
    prompt: str,
    model: str | None = None,
    max_turns: int = 40,
) -> dict:
    """Run avoid-agent in headless mode within the worktree.

    Streams stderr events live to the terminal for progress visibility.
    No hard timeout — max_turns controls when the agent stops.
    """
    cmd = [
        sys.executable, "-m", "avoid_agent", "headless",
        "--prompt", prompt,
        "--auto-approve",
        "--no-session",
        "--max-turns", str(max_turns),
    ]
    if model:
        cmd.extend(["--model", model])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(worktree_path)

    log(f"Running headless agent in {worktree_path}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=worktree_path, env=env,
    )

    # Stream stderr in a background thread for live progress
    tool_count = [0]
    stderr_thread = threading.Thread(
        target=_stream_stderr, args=(proc.stderr, tool_count), daemon=True,
    )
    stderr_thread.start()

    # Read stdout directly (communicate() would compete with stderr thread)
    stdout = proc.stdout.read()
    proc.wait()
    stderr_thread.join(timeout=5)

    log(f"Agent finished. Exit code: {proc.returncode}, tool calls: {tool_count[0]}")

    # Parse result from stdout
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": f"Failed to parse agent output. Exit code: {proc.returncode}",
            "stdout_preview": stdout[:500] if stdout else "",
        }


def _gather_completed_tasks(repo_root: Path) -> list[str]:
    """Return descriptions of already-completed backlog items."""
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
    """Return recent selfdev merge commits for context."""
    result = subprocess.run(
        ["git", "log", "--oneline", f"--grep=selfdev:", f"-{limit}"],
        capture_output=True, text=True, cwd=repo_root, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _gather_learnings(repo_root: Path, limit: int = 5) -> str:
    """Return the most recent learnings entries, if any."""
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
    """Return a compact file tree of the worktree (excluding noise)."""
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


def _load_frozen_patterns(repo_root: Path) -> list[str]:
    """Load frozen file patterns from selfdev-policy.yaml."""
    policy_path = repo_root / "selfdev-policy.yaml"
    if not policy_path.exists():
        return []
    try:
        import yaml
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = yaml.safe_load(f) or {}
        return policy.get("frozen", [])
    except Exception:
        return []


def _load_allowed_patterns(repo_root: Path) -> list[str]:
    """Load allowed file patterns from selfdev-policy.yaml."""
    policy_path = repo_root / "selfdev-policy.yaml"
    if not policy_path.exists():
        return []
    try:
        import yaml
        with open(policy_path, "r", encoding="utf-8") as f:
            policy = yaml.safe_load(f) or {}
        return policy.get("allowed", []) + policy.get("unrestricted", [])
    except Exception:
        return []


def build_prompt_for_task(task_text: str, repo_root: Path, worktree_path: Path) -> str:
    """Build the headless prompt with project context for a backlog task."""
    sections = []

    # CRITICAL constraints first — frozen files
    frozen = _load_frozen_patterns(repo_root)
    allowed = _load_allowed_patterns(repo_root)
    frozen_text = "\n".join(f"  - `{p}`" for p in frozen) if frozen else "  (none)"
    allowed_text = "\n".join(f"  - `{p}`" for p in allowed) if allowed else "  (none)"

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


def log(message: str) -> None:
    """Print a log message with timestamp."""
    ts = time.strftime("%H:%M:%S")
    print(f"[selfdev {ts}] {message}", flush=True)


def run_one_cycle(
    repo_root: Path,
    model: str | None = None,
    max_turns: int = 40,
) -> str:
    """Run one self-improvement cycle.

    Returns:
        "restart" - changes merged, supervisor should restart
        "done" - no more backlog items
        "failed" - task failed but branch preserved for review
        "error" - unexpected error
    """
    items = parse_backlog(repo_root)
    if not items:
        log("No unchecked backlog items. Nothing to do.")
        return "done"

    item = items[0]
    log(f"Picked task: {item.text}")

    branch_name = f"selfdev/{re.sub(r'[^a-z0-9]+', '-', item.text.lower())[:50].strip('-')}"
    worktree_path = None

    try:
        # Create worktree
        log(f"Creating worktree on branch: {branch_name}")
        worktree_path = create_worktree(repo_root, branch_name)

        # Run the agent
        prompt = build_prompt_for_task(item.text, repo_root, worktree_path)
        result = run_agent_headless(
            worktree_path, prompt, model=model, max_turns=max_turns,
        )

        if not result.get("success"):
            error = result.get("error", "unknown error")
            log(f"Agent failed: {error}")
            # Commit any partial work before preserving the branch
            _commit_if_dirty(worktree_path, item.text)
            mark_backlog_item(repo_root, item, "failed",
                              note=f"{error[:80]} | branch: {branch_name}")
            detach_worktree(repo_root, branch_name)
            return "failed"

        log(f"Agent completed. Tool calls: {len(result.get('tool_calls', []))}")

        # Check if the agent actually made changes
        diff_result = subprocess.run(
            ["git", "diff", "--stat", "main", "HEAD"],
            capture_output=True, text=True, cwd=worktree_path, check=False,
        )
        if not diff_result.stdout.strip():
            # Agent may not have committed — check working tree
            if not _commit_if_dirty(worktree_path, item.text):
                log("No changes were made. Marking as failed.")
                mark_backlog_item(repo_root, item, "failed", note="no changes produced")
                cleanup_worktree(repo_root, branch_name)
                return "failed"

        # Validate
        log("Running validation...")
        validation = validate_worktree(repo_root, worktree_path)
        log(f"Validation {'PASSED' if validation.passed else 'FAILED'}:")
        print(validation.summary, flush=True)

        if not validation.passed:
            log("Validation failed. Branch preserved for review.")
            mark_backlog_item(repo_root, item, "failed",
                              note=f"validation failed | branch: {branch_name}")
            detach_worktree(repo_root, branch_name)
            return "failed"

        # Merge back to main
        log("Merging to main...")
        if not merge_worktree(repo_root, branch_name):
            mark_backlog_item(repo_root, item, "failed",
                              note=f"merge conflict | branch: {branch_name}")
            detach_worktree(repo_root, branch_name)
            return "failed"

        mark_backlog_item(repo_root, item, "done")
        cleanup_worktree(repo_root, branch_name)
        log(f"Task completed and merged: {item.text}")
        return "restart"

    except Exception as e:
        log(f"Unexpected error: {e}")
        if worktree_path:
            _commit_if_dirty(worktree_path, item.text)
            mark_backlog_item(repo_root, item, "failed",
                              note=f"{str(e)[:80]} | branch: {branch_name}")
            detach_worktree(repo_root, branch_name)
        else:
            mark_backlog_item(repo_root, item, "failed", note=str(e)[:100])
        return "error"


def _commit_if_dirty(worktree_path: Path, task_text: str) -> bool:
    """Commit any uncommitted changes in the worktree. Returns True if committed."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=worktree_path, check=False,
    )
    if not status.stdout.strip():
        return False
    subprocess.run(
        ["git", "add", "-A"], cwd=worktree_path, check=True,
        capture_output=True, text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", f"selfdev (partial): {task_text[:72]}"],
        cwd=worktree_path, check=True, capture_output=True, text=True,
    )
    log("Committed working tree changes.")
    return True


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
    log("Self-improvement loop starting")
    log(f"Repo: {repo_root}")
    log(f"Model: {model or 'default'}")
    log(f"Single mode: {single}")
    print("", flush=True)

    result = run_one_cycle(repo_root, model=model, max_turns=max_turns)

    if result == "restart":
        log("Requesting supervisor restart...")
        return RESTART_EXIT_CODE
    elif result == "done":
        log("All backlog items complete.")
        return 0
    elif result == "failed":
        if single:
            log("Task failed in single mode. Exiting.")
            return 1
        # In loop mode, try next item
        log("Task failed. Trying next item...")
        next_result = run_one_cycle(repo_root, model=model, max_turns=max_turns)
        if next_result == "restart":
            return RESTART_EXIT_CODE
        elif next_result == "done":
            return 0
        else:
            log("Second task also failed. Exiting to avoid infinite failure loop.")
            return 1
    else:
        log("Unexpected error. Exiting.")
        return 1
