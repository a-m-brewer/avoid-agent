"""Workflow module for selfdev — contains single-responsibility step modules."""

from avoid_agent.selfdev.workflow.backlog import BacklogItem, mark_backlog_item, parse_backlog
from avoid_agent.selfdev.workflow.worktree import (
    WorktreeManager,
    cleanup_worktree,
    commit_if_dirty,
    create_worktree,
    detach_worktree,
    merge_worktree,
)
from avoid_agent.selfdev.workflow.runner import run_agent_headless
from avoid_agent.selfdev.workflow.prompt_builder import build_prompt_for_task

__all__ = [
    "BacklogItem",
    "WorktreeManager",
    "build_prompt_for_task",
    "cleanup_worktree",
    "commit_if_dirty",
    "create_worktree",
    "detach_worktree",
    "mark_backlog_item",
    "merge_worktree",
    "parse_backlog",
    "run_agent_headless",
]
