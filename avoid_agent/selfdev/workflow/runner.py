"""Agent runner for the selfdev workflow.

This module handles running the headless agent in a worktree and
streaming stderr events for progress visibility.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import logging

log = logging.getLogger(__name__)


def _stream_stderr(pipe, tool_count_ref: list[int]) -> None:
    """Read stderr lines from the agent process and log progress live.

    This function runs in a background thread and parses JSON events
    from stderr, logging tool executions and turn completions.

    Args:
        pipe: The stderr pipe from the subprocess
        tool_count_ref: A mutable list containing a single int for tool count
    """
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
                log.info(f"  [{tool_count_ref[0]}] tool: {tool_name}")
            elif etype == "turn_start":
                turn = event.get("turn", "?")
                log.info(f"  --- turn {turn} ---")
            elif etype == "turn_complete":
                turn = event.get("turn", "?")
                tokens = event.get("input_tokens", "?")
                log.info(f"  turn {turn} done ({tokens} input tokens)")
            elif etype in ("error", "fatal", "validation_error"):
                log.warning(f"  [{etype}] {event.get('message', '')[:200]}")
        except json.JSONDecodeError:
            pass


def run_agent_headless(
    worktree_path: Path,
    prompt: str,
    model: str | None = None,
    max_turns: int = 40,
    stderr_streamer: Callable[[any, list[int]], None] | None = None,
) -> dict:
    """Run avoid-agent in headless mode within the worktree.

    Streams stderr events live to the terminal for progress visibility.
    No hard timeout — max_turns controls when the agent stops.

    Args:
        worktree_path: Path to the worktree where the agent should run
        prompt: The prompt to send to the agent
        model: Optional model override (e.g. "anthropic/claude-sonnet-4-6")
        max_turns: Maximum number of turns for the agent
        stderr_streamer: Optional custom stderr streamer function

    Returns:
        A dict with keys:
        - success: bool indicating if the agent completed successfully
        - error: str with error message if success is False
        - stdout_preview: str preview of stdout if parsing failed
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

    log.info(f"Running headless agent in {worktree_path}")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=worktree_path, env=env,
    )

    # Stream stderr in a background thread for live progress
    tool_count = [0]
    streamer = stderr_streamer or _stream_stderr
    stderr_thread = threading.Thread(
        target=streamer, args=(proc.stderr, tool_count), daemon=True,
    )
    stderr_thread.start()

    # Read stdout directly (communicate() would compete with stderr thread)
    stdout = proc.stdout.read()
    proc.wait()
    stderr_thread.join(timeout=5)

    log.info(f"Agent finished. Exit code: {proc.returncode}, tool calls: {tool_count[0]}")

    # Parse result from stdout
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "success": False,
            "error": f"Failed to parse agent output. Exit code: {proc.returncode}",
            "stdout_preview": stdout[:500] if stdout else "",
        }


class AgentRunner:
    """A class to manage running the headless agent for a selfdev cycle.

    This encapsulates the state and operations for running an agent,
    making it easier to track progress and handle results.
    """

    def __init__(
        self,
        worktree_path: Path,
        prompt: str,
        model: str | None = None,
        max_turns: int = 40,
        stderr_streamer: Callable[[any, list[int]], None] | None = None,
    ):
        self.worktree_path = worktree_path
        self.prompt = prompt
        self.model = model
        self.max_turns = max_turns
        self.stderr_streamer = stderr_streamer
        self._result: dict | None = None

    @property
    def result(self) -> dict | None:
        """Get the result from the last run."""
        return self._result

    @property
    def success(self) -> bool:
        """Check if the last run was successful."""
        return self._result is not None and self._result.get("success", False)

    @property
    def tool_call_count(self) -> int:
        """Get the number of tool calls from the last run."""
        if self._result is None:
            return 0
        return len(self._result.get("tool_calls", []))

    def run(self) -> dict:
        """Run the agent and store the result.

        Returns:
            The result dict from the agent
        """
        self._result = run_agent_headless(
            self.worktree_path,
            self.prompt,
            model=self.model,
            max_turns=self.max_turns,
            stderr_streamer=self.stderr_streamer,
        )
        return self._result
