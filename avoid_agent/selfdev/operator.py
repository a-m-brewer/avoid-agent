"""Operator agent for the selfdev workflow.

The operator is a persistent LLM-based supervisor that:
- Reads the backlog and refined/ directory
- Picks items based on readiness and complexity
- Refines broad items into sub-tasks (creates refined/<slug>.md files)
- Launches workers (headless agents) for each sub-task
- Reviews worker results and decides next steps (merge, retry, skip)
- Stays alive across worker runs

The operator runs as an avoid-agent instance with a special system prompt.
It uses the standard 4 tools (read_file, write_file, edit_file, run_bash)
to manage everything.

This module is FROZEN - the agent must never modify it during self-improvement.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from avoid_agent.selfdev.operator_prompt import build_operator_prompt


def run_operator(
    repo_root: Path,
    model: str | None = None,
    max_cycles: int = 10,
    max_turns_per_worker: int = 40,
) -> int:
    """Run the operator agent that manages the selfdev workflow.

    The operator is launched as a headless avoid-agent with a special system prompt.
    It uses run_bash to create worktrees, launch workers, validate, and merge.

    Args:
        repo_root: Path to the repository root.
        model: Model override for the operator agent.
        max_cycles: Maximum number of backlog items to process before exiting.
        max_turns_per_worker: Max turns for each worker headless run.

    Returns:
        Exit code (42 = restart, 0 = done, 1 = error).
    """
    from avoid_agent.selfdev import RESTART_EXIT_CODE

    prompt = build_operator_prompt(
        repo_root=repo_root,
        max_cycles=max_cycles,
        max_turns_per_worker=max_turns_per_worker,
    )

    cmd = [
        sys.executable, "-m", "avoid_agent", "headless",
        "--prompt", prompt,
        "--auto-approve",
        "--no-session",
        "--max-turns", str(max_cycles * 20),  # generous turn budget for operator
    ]
    if model:
        cmd.extend(["--model", model])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)

    return cmd, env, repo_root


def run_operator_headless(
    repo_root: Path,
    model: str | None = None,
    max_cycles: int = 10,
    max_turns_per_worker: int = 40,
) -> dict:
    """Run the operator and return the result dict (for non-TUI usage)."""
    cmd, env, _ = run_operator(repo_root, model, max_cycles, max_turns_per_worker)

    proc = subprocess.Popen(
        cmd, cwd=repo_root, env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    stdout = proc.stdout.read()
    proc.wait()

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": f"Failed to parse operator output. Exit code: {proc.returncode}",
        }


def stream_operator_to_tui(
    repo_root: Path,
    tui,
    stderr_streamer,
    model: str | None = None,
    max_cycles: int = 10,
    max_turns_per_worker: int = 40,
) -> dict:
    """Run the operator with stderr streamed to TUI. Returns result dict."""
    cmd, env, _ = run_operator(repo_root, model, max_cycles, max_turns_per_worker)

    proc = subprocess.Popen(
        cmd, cwd=repo_root, env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    tool_count = [0]
    stderr_thread = threading.Thread(
        target=stderr_streamer,
        args=(proc.stderr, tui, tool_count),
        daemon=True,
    )
    stderr_thread.start()

    stdout = proc.stdout.read()
    proc.wait()
    stderr_thread.join(timeout=10)

    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": f"Failed to parse operator output. Exit code: {proc.returncode}",
        }
