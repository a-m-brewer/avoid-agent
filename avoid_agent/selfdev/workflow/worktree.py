"""Git worktree management for the selfdev workflow."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import logging

log = logging.getLogger(__name__)


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

    Args:
        repo_root: Path to the repository root
        branch_name: Name for the new branch and worktree

    Returns:
        Path to the newly created worktree directory

    Raises:
        subprocess.CalledProcessError: If git command fails
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
    log.info(f"Created worktree at {worktree_path}")
    return worktree_path


def cleanup_worktree(repo_root: Path, branch_name: str) -> None:
    """Remove a worktree and its branch entirely.

    Args:
        repo_root: Path to the repository root
        branch_name: Name of the branch to remove
    """
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
    log.info(f"Cleaned up worktree and branch: {branch_name}")


def detach_worktree(repo_root: Path, branch_name: str) -> None:
    """Remove the worktree directory but keep the branch for review.

    Args:
        repo_root: Path to the repository root
        branch_name: Name of the branch to detach
    """
    worktree_path = repo_root / ".worktrees" / branch_name
    if worktree_path.exists():
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_path), "--force"],
            cwd=repo_root, check=False, capture_output=True, text=True,
        )
    log.info(f"Branch preserved for review: {branch_name}")
    log.info(f"  Inspect with: git log main..{branch_name}")
    log.info(f"  Diff with:    git diff main...{branch_name}")
    log.info(f"  Delete with:  git branch -D {branch_name}")


def merge_worktree(repo_root: Path, branch_name: str) -> bool:
    """Merge the worktree branch back to main. Returns True on success.

    Args:
        repo_root: Path to the repository root
        branch_name: Name of the branch to merge

    Returns:
        True if merge succeeded, False otherwise
    """
    result = subprocess.run(
        ["git", "merge", "--no-ff", branch_name, "-m", f"selfdev: merge {branch_name}"],
        cwd=repo_root, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        log.error(f"Merge failed: {result.stderr}")
        subprocess.run(
            ["git", "merge", "--abort"],
            cwd=repo_root, check=False, capture_output=True, text=True,
        )
        return False
    return True


def commit_if_dirty(worktree_path: Path, task_text: str) -> bool:
    """Commit any uncommitted changes in the worktree.

    Args:
        worktree_path: Path to the worktree
        task_text: Commit message for the partial work

    Returns:
        True if changes were committed, False if working tree was clean
    """
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
    log.info("Committed working tree changes.")
    return True


class WorktreeManager:
    """A class to manage git worktree operations for a single selfdev cycle.

    This encapsulates the state and operations for a worktree, making it
    easier to track and clean up resources.
    """

    def __init__(self, repo_root: Path, branch_name: str):
        self.repo_root = repo_root
        self.branch_name = branch_name
        self._path: Path | None = None
        self._attached = False

    @property
    def path(self) -> Path | None:
        """Get the worktree path if created."""
        return self._path

    def create(self) -> Path:
        """Create the worktree and branch.

        Returns:
            Path to the created worktree
        """
        self._path = create_worktree(self.repo_root, self.branch_name)
        self._attached = True
        return self._path

    def cleanup(self) -> None:
        """Remove the worktree and branch entirely."""
        if self._attached:
            cleanup_worktree(self.repo_root, self.branch_name)
            self._attached = False
            self._path = None

    def detach(self) -> None:
        """Remove the worktree directory but keep the branch."""
        if self._attached:
            detach_worktree(self.repo_root, self.branch_name)
            self._attached = False
            self._path = None

    def merge(self) -> bool:
        """Merge the branch back to main.

        Returns:
            True if merge succeeded, False otherwise
        """
        if not self._attached:
            return False
        return merge_worktree(self.repo_root, self.branch_name)

    def commit_partial(self, task_text: str) -> bool:
        """Commit any dirty working tree changes.

        Returns:
            True if changes were committed, False if clean
        """
        if not self._path:
            return False
        return commit_if_dirty(self._path, task_text)

    def __enter__(self) -> "WorktreeManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._attached:
            self.cleanup()
