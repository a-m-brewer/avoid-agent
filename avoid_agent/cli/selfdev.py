"""Selfdev mode: self-improvement loop for avoid-agent.

This module provides the self-improvement workflow that:
- Reads backlog.md for pending tasks
- Creates git worktrees for each task
- Runs the agent headless to implement tasks
- Validates and merges results
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from avoid_agent.cli.shared import _count_backlog_status, _count_backlog_totals, gather_initial_context
from avoid_agent.infra.config import DEFAULT_MODEL
from avoid_agent.selfdev import RESTART_EXIT_CODE
from avoid_agent.selfdev.loop import (
    run_loop,
)
from avoid_agent.selfdev.workflow import (
    commit_if_dirty,
    build_prompt_for_task,
    cleanup_worktree,
    create_worktree,
    detach_worktree,
    mark_backlog_item,
    merge_worktree,
    parse_backlog,
)
from avoid_agent.selfdev.refine import (
    find_next_subtask,
    mark_subtask,
    update_refined_status,
)
from avoid_agent.selfdev.validate import validate_worktree
from avoid_agent.tui import TUI
from avoid_agent.tui.components.conversation import (
    AssistantItem,
    StatusItem,
    ToolCallItem,
    ToolResultItem,
)
from avoid_agent.learnings_analyzer import analyze

if TYPE_CHECKING:
    argparse.Namespace


def _notify(url: str, payload: dict) -> None:
    """POST a JSON payload to url as a webhook notification.

    Configure via the SELFDEV_WEBHOOK_URL environment variable.
    Failures are logged to stderr but never raise.
    """
    import urllib.request
    import urllib.error
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            pass  # fire and forget
    except Exception as e:  # pylint: disable=broad-except
        print(f"[selfdev] webhook error: {e}", file=sys.stderr)


def _stream_selfdev_headless_stderr(pipe, tui: TUI, tool_count_ref: list[int]) -> None:
    """Map headless stderr JSON events into rich TUI updates."""
    text_buffer = ""
    suppress_streaming = False

    for raw_line in pipe:
        line = raw_line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "turn_start":
            turn = event.get("turn", "?")
            tui.set_phase(f"turn {turn}")
            tui.push_item(StatusItem(text=f"turn {turn} started"))
            tui.set_spinner_message("thinking...")

        elif etype == "text_delta":
            delta = event.get("text", "")
            if delta:
                text_buffer += delta
                if not suppress_streaming:
                    if text_buffer.lstrip().startswith("{"):
                        suppress_streaming = True
                    else:
                        tui.append_chunk(delta)

        elif etype == "tool_call_detected":
            call_id = event.get("id")
            name = event.get("name", "tool")
            args = event.get("arguments", {})
            tui.push_item(ToolCallItem(id=call_id, name=name, arguments=args, status="pending"))
            tui.set_spinner_message(f"tool detected: {name}")

        elif etype == "tool_execution_start":
            text_buffer = ""
            suppress_streaming = False
            tool_count_ref[0] += 1
            call_id = event.get("id")
            name = event.get("name", "tool")
            args = event.get("arguments", {})
            tui.push_item(ToolCallItem(id=call_id, name=name, arguments=args, status="running"))
            if call_id:
                tui.update_tool_status(call_id, "running")
            tui.set_spinner_message(f"running tool: {name}")

        elif etype == "tool_result":
            call_id = event.get("id")
            name = event.get("name") or "tool"
            content = str(event.get("content") or "")
            is_error = bool(event.get("is_error", False))
            status = "failed" if is_error else "done"
            if call_id:
                tui.update_tool_status(call_id, status)
            tui.push_item(
                ToolResultItem(
                    id=call_id,
                    name=name,
                    content=content,
                    status=status,
                )
            )
            tui.reset_spinner_message()

        elif etype == "reasoning":
            item = event.get("item")
            summary = "reasoning"
            if isinstance(item, dict):
                value = item.get("summary")
                if isinstance(value, list):
                    summary = " ".join(str(x) for x in value if x)
                elif value:
                    summary = str(value)
            tui.push_item(StatusItem(text=f"reasoning: {summary}"))
            tui.set_spinner_message("reasoning...")

        elif etype == "status":
            message = str(event.get("message") or "")
            if message:
                tui.push_item(StatusItem(text=message))
                tui.set_spinner_message(message)

        elif etype == "structured_action":
            text_buffer = ""
            suppress_streaming = False
            message = event.get("message")
            if isinstance(message, str) and message.strip():
                tui.replace_last_assistant(message)

        elif etype == "context_trimmed":
            message = event.get("message")
            if isinstance(message, str) and message.strip():
                tui.push_item(StatusItem(text=message))

        elif etype in ("error", "fatal", "validation_error"):
            message = str(event.get("message") or etype)
            tui.report_error(message)
            tui.reset_spinner_message()

        elif etype == "turn_complete":
            tokens = event.get("input_tokens")
            if isinstance(tokens, int):
                tui.update_tokens(tokens)
            turn = event.get("turn", "?")
            tui.push_item(StatusItem(text=f"turn {turn} complete"))
            tui.set_phase("awaiting feedback")


def run(args) -> int:
    """Run the self-improvement loop.

    This is the main entry point for the `avoid-agent selfdev` command.

    Args:
        args: argparse.Namespace with the following attributes:
            - model: Model override for headless agent
            - max_turns: Max turns per headless run
            - single: Run only one cycle then exit
            - operator: Use operator agent mode
            - interactive: Use interactive mode with user feedback
            - legacy: Use legacy non-interactive output

    Returns:
        Exit code (42 = restart requested, 0 = done, 1 = error)
    """
    load_dotenv()
    repo_root = Path(__file__).resolve().parent.parent
    model = args.model or DEFAULT_MODEL

    # Print learnings suggestions before starting the loop
    learnings_dir = repo_root / ".learnings" / "sessions"
    suggestions = analyze(learnings_dir)
    if suggestions:
        ts = time.strftime("%H:%M:%S")
        for suggestion in suggestions:
            print(f"[selfdev {ts}] [suggestion] {suggestion}", flush=True)

    if getattr(args, "operator", False):
        exit_code = _run_selfdev_operator(
            repo_root=repo_root,
            model=model,
            max_turns=args.max_turns,
        )
    elif getattr(args, "legacy", False):
        from avoid_agent.selfdev.loop import run_loop
        exit_code = run_loop(
            repo_root=repo_root,
            model=model,
            max_turns=args.max_turns,
            single=args.single,
        )
    elif getattr(args, "interactive", False):
        exit_code = _run_selfdev_interactive(
            repo_root=repo_root,
            model=model,
            max_turns=args.max_turns,
        )
    else:
        # Default: observation mode (read-only TUI, autonomous agent)
        exit_code = _run_selfdev_observe(
            repo_root=repo_root,
            model=model,
            max_turns=args.max_turns,
        )

    # Write status JSON for external monitoring
    completed_count, pending_count, failed_count = _count_backlog_status(repo_root)
    status_path = repo_root / "selfdev-status.json"
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump({
            "last_run": datetime.now().isoformat(),
            "exit_code": exit_code,
            "completed_count": completed_count,
            "pending_count": pending_count,
            "failed_count": failed_count,
        }, f, indent=2)
        f.write("\n")

    # Send webhook notification if configured
    webhook_url = os.environ.get("SELFDEV_WEBHOOK_URL")
    if webhook_url:
        _notify(webhook_url, {
            "event": "cycle_complete",
            "exit_code": exit_code,
            "timestamp": datetime.now().isoformat(),
            "completed": completed_count,
            "pending": pending_count,
            "failed": failed_count,
        })

    return exit_code


def _run_selfdev_observe(repo_root: Path, model: str | None, max_turns: int) -> int:
    """Run one selfdev cycle with read-only TUI streaming — no user input, fully autonomous."""
    # Check refined/ for pending sub-tasks first, then fall back to backlog
    subtask = find_next_subtask(repo_root)
    backlog_item = None

    if subtask:
        task_text = subtask.text
        is_refined = True
    else:
        items = parse_backlog(repo_root)
        if not items:
            print("No unchecked backlog items or refined sub-tasks. Nothing to do.")
            return 0
        backlog_item = items[0]
        task_text = backlog_item.text
        is_refined = False

    done_count, total_count = _count_backlog_totals(repo_root)
    progress_current = done_count + 1

    branch_name = f"selfdev/{re.sub(r'[^a-z0-9]+', '-', task_text.lower())[:50].strip('-')}"
    worktree_path: Path | None = None

    tui = TUI(on_submit=lambda _text: None, model=model, auto_spinner_on_submit=False, read_only=True)
    tui.set_phase("preparing")
    tui.set_progress(progress_current, total_count)
    source_label = f"refined sub-task: {task_text}" if is_refined else f"task: {task_text}"
    tui.push_item(StatusItem(text=source_label))

    worker_done = threading.Event()
    worker_result: dict[str, object] = {"result": "error", "error": "unexpected"}

    def _mark_done() -> None:
        if is_refined and subtask:
            mark_subtask(subtask.parent_path, subtask.line_number, "done")
            update_refined_status(subtask.parent_path)
        elif backlog_item:
            mark_backlog_item(repo_root, backlog_item, "done")

    def _mark_failed(note: str) -> None:
        if is_refined and subtask:
            mark_subtask(subtask.parent_path, subtask.line_number, "failed", note=note)
            update_refined_status(subtask.parent_path)
        elif backlog_item:
            mark_backlog_item(repo_root, backlog_item, "failed", note=note)

    def worker() -> None:
        nonlocal worktree_path
        try:
            tui.set_phase("creating worktree")
            worktree_path = create_worktree(repo_root, branch_name)

            prompt = build_prompt_for_task(task_text, repo_root, worktree_path)

            cmd = [
                sys.executable, "-m", "avoid_agent", "headless",
                "--prompt", prompt,
                "--auto-approve", "--no-session",
                "--max-turns", str(max_turns),
            ]
            if model:
                cmd.extend(["--model", model])

            env = os.environ.copy()
            env["PYTHONPATH"] = str(worktree_path)

            tui.set_phase("running agent")
            tui._start_spinner()  # pylint: disable=protected-access

            proc = subprocess.Popen(
                cmd, cwd=worktree_path, env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )

            tool_count = [0]
            stderr_thread = threading.Thread(
                target=_stream_selfdev_headless_stderr,
                args=(proc.stderr, tui, tool_count),
                daemon=True,
            )
            stderr_thread.start()

            stdout = proc.stdout.read()
            proc.wait()
            stderr_thread.join(timeout=5)

            try:
                result = json.loads(stdout)
            except json.JSONDecodeError:
                result = {"success": False, "error": "Failed to parse agent output"}

            if not result.get("success"):
                error = result.get("error", "unknown error")
                tui.report_error(str(error))
                if worktree_path:
                    commit_if_dirty(worktree_path, task_text)
                _mark_failed(f"{str(error)[:80]} | branch: {branch_name}")
                detach_worktree(repo_root, branch_name)
                worker_result["result"] = "failed"
                worker_result["error"] = error
                return

            tui.push_item(StatusItem(text=f"agent completed ({len(result.get('tool_calls', []))} tool calls)"))

            # Check for changes
            diff_result = subprocess.run(
                ["git", "diff", "--stat", "main", "HEAD"],
                capture_output=True, text=True, cwd=worktree_path, check=False,
            )
            if not diff_result.stdout.strip() and not commit_if_dirty(worktree_path, task_text):
                _mark_failed("no changes produced")
                cleanup_worktree(repo_root, branch_name)
                worker_result["result"] = "failed"
                worker_result["error"] = "no changes produced"
                tui.report_error("No changes were produced.")
                return

            # Validate
            tui.set_phase("validating")
            validation = validate_worktree(repo_root, worktree_path)
            if not validation.passed:
                _mark_failed(f"validation failed | branch: {branch_name}")
                detach_worktree(repo_root, branch_name)
                worker_result["result"] = "failed"
                worker_result["error"] = "validation failed"
                tui.report_error("Validation failed. Branch preserved for review.")
                tui.push_item(AssistantItem(text=validation.summary))
                return

            # Merge
            tui.set_phase("merging")
            if not merge_worktree(repo_root, branch_name):
                _mark_failed(f"merge conflict | branch: {branch_name}")
                detach_worktree(repo_root, branch_name)
                worker_result["result"] = "failed"
                worker_result["error"] = "merge conflict"
                tui.report_error("Merge failed. Branch preserved for review.")
                return

            _mark_done()
            cleanup_worktree(repo_root, branch_name)
            worker_result["result"] = "restart"
            tui.set_phase("done")
            tui.push_item(StatusItem(text=f"Task completed and merged: {task_text}"))

        except Exception as e:  # pylint: disable=broad-except
            worker_result["result"] = "error"
            worker_result["error"] = str(e)
            tui.report_error(str(e))
            if worktree_path:
                try:
                    commit_if_dirty(worktree_path, task_text)
                    _mark_failed(f"{str(e)[:80]} | branch: {branch_name}")
                    detach_worktree(repo_root, branch_name)
                except Exception:  # pylint: disable=broad-except
                    pass
        finally:
            tui._stop_spinner()  # pylint: disable=protected-access
            worker_done.set()
            tui.stop()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    tui.run()
    worker_done.wait(timeout=5)

    result = str(worker_result.get("result") or "error")
    if result in ("restart", "success"):
        return RESTART_EXIT_CODE
    if result in ("failed", "error"):
        return 1
    return 0


def _run_selfdev_interactive(repo_root: Path, model: str | None, max_turns: int) -> int:
    """Run one selfdev cycle with live TUI streaming and optional user feedback turns."""
    import queue

    items = parse_backlog(repo_root)
    if not items:
        print("No unchecked backlog items. Nothing to do.")
        return 0

    item = items[0]
    done_count, total_count = _count_backlog_totals(repo_root)
    progress_current = done_count + 1

    worktree_path: Path | None = None
    branch_name = f"selfdev/{re.sub(r'[^a-z0-9]+', '-', item.text.lower())[:50].strip('-')}"

    prompts: queue.Queue[str] = queue.Queue()
    prompt_counter = [1]
    worker_done = threading.Event()
    worker_result: dict[str, object] = {"result": "error", "error": "unexpected"}
    stderr_thread: threading.Thread | None = None

    tui = TUI(on_submit=lambda _text: None, model=model, auto_spinner_on_submit=False)
    tui.set_phase("preparing")
    tui.set_progress(progress_current, total_count)
    tui.push_item(StatusItem(text=f"task: {item.text}"))
    tui.push_item(StatusItem(text="Type feedback to send another turn, /done to finish and validate."))

    def on_submit(text: str) -> None:
        stripped = text.strip()
        if stripped == "/done":
            prompts.put("__SELFDEV_DONE__")
            tui.push_item(StatusItem(text="finishing current run and validating..."))
            return
        if stripped == "/abort":
            prompts.put("__SELFDEV_ABORT__")
            tui.push_item(StatusItem(text="aborting selfdev run..."))
            return
        prompts.put(text)
        tui.push_item(StatusItem(text=f"queued feedback turn #{prompt_counter[0]}"))
        prompt_counter[0] += 1

    tui.on_submit = on_submit

    def worker() -> None:
        nonlocal worktree_path, stderr_thread
        proc = None
        try:
            tui.set_phase("creating worktree")
            worktree_path = create_worktree(repo_root, branch_name)

            prompt = build_prompt_for_task(item.text, repo_root, worktree_path)

            cmd = [
                sys.executable,
                "-m",
                "avoid_agent",
                "headless",
                "--auto-approve",
                "--no-session",
                "--max-turns",
                str(max_turns),
            ]
            if model:
                cmd.extend(["--model", model])

            env = os.environ.copy()
            env["PYTHONPATH"] = str(worktree_path)

            tui.set_phase("running agent")
            tui._start_spinner()  # pylint: disable=protected-access

            proc = subprocess.Popen(
                cmd,
                cwd=worktree_path,
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            tool_count = [0]
            stderr_thread = threading.Thread(
                target=_stream_selfdev_headless_stderr,
                args=(proc.stderr, tui, tool_count),
                daemon=True,
            )
            stderr_thread.start()

            assert proc.stdin is not None
            assert proc.stdout is not None

            def send_prompt(prompt_text: str) -> None:
                proc.stdin.write(json.dumps({"prompt": prompt_text}) + "\n")
                proc.stdin.flush()

            send_prompt(prompt)

            successful_turns = 0

            while True:
                line = proc.stdout.readline()
                if not line:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    result = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if not result.get("success"):
                    worker_result["result"] = "failed"
                    worker_result["error"] = result.get("error") or "headless run failed"
                    break

                successful_turns += 1
                tui.push_item(StatusItem(text=f"turn {successful_turns} completed"))

                tui.push_item(StatusItem(text="waiting for your feedback (/done to continue pipeline)..."))
                next_prompt = prompts.get()

                if next_prompt is None or next_prompt == "__SELFDEV_DONE__":
                    worker_result["result"] = "success"
                    break

                if next_prompt == "__SELFDEV_ABORT__":
                    worker_result["result"] = "failed"
                    worker_result["error"] = "aborted by user"
                    break

                send_prompt(next_prompt)
                tui.set_phase("running feedback turn")

            if proc.stdin:
                try:
                    proc.stdin.write(json.dumps({"command": "quit"}) + "\n")
                    proc.stdin.flush()
                except Exception:  # pylint: disable=broad-except
                    pass

            proc.wait(timeout=10)

            if worker_result.get("result") != "success":
                error = str(worker_result.get("error") or "unknown error")
                tui.report_error(error)
                if worktree_path:
                    commit_if_dirty(worktree_path, item.text)
                mark_backlog_item(repo_root, item, "failed", note=f"{error[:80]} | branch: {branch_name}")
                detach_worktree(repo_root, branch_name)
                return

            # Ensure there is a commit if agent only changed working tree files.
            diff_result = subprocess.run(
                ["git", "diff", "--stat", "main", "HEAD"],
                capture_output=True,
                text=True,
                cwd=worktree_path,
                check=False,
            )
            if not diff_result.stdout.strip() and not commit_if_dirty(worktree_path, item.text):
                mark_backlog_item(repo_root, item, "failed", note="no changes produced")
                cleanup_worktree(repo_root, branch_name)
                worker_result["result"] = "failed"
                worker_result["error"] = "no changes produced"
                tui.report_error("No changes were produced.")
                return

            tui.set_phase("validating")
            validation = validate_worktree(repo_root, worktree_path)
            if not validation.passed:
                mark_backlog_item(repo_root, item, "failed", note=f"validation failed | branch: {branch_name}")
                detach_worktree(repo_root, branch_name)
                worker_result["result"] = "failed"
                worker_result["error"] = "validation failed"
                tui.report_error("Validation failed. Branch preserved for review.")
                tui.push_item(AssistantItem(text=validation.summary))
                return

            tui.set_phase("merging")
            if not merge_worktree(repo_root, branch_name):
                mark_backlog_item(repo_root, item, "failed", note=f"merge conflict | branch: {branch_name}")
                detach_worktree(repo_root, branch_name)
                worker_result["result"] = "failed"
                worker_result["error"] = "merge conflict"
                tui.report_error("Merge failed. Branch preserved for review.")
                return

            mark_backlog_item(repo_root, item, "done")
            cleanup_worktree(repo_root, branch_name)
            worker_result["result"] = "restart"
            tui.set_phase("done")
            tui.push_item(StatusItem(text=f"Task completed and merged: {item.text}"))

        except Exception as e:  # pylint: disable=broad-except
            worker_result["result"] = "error"
            worker_result["error"] = str(e)
            tui.report_error(str(e))
            if worktree_path:
                try:
                    commit_if_dirty(worktree_path, item.text)
                    mark_backlog_item(repo_root, item, "failed", note=f"{str(e)[:80]} | branch: {branch_name}")
                    detach_worktree(repo_root, branch_name)
                except Exception:  # pylint: disable=broad-except
                    pass
        finally:
            if stderr_thread:
                stderr_thread.join(timeout=3)
            tui._stop_spinner()  # pylint: disable=protected-access
            worker_done.set()
            tui.stop()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    tui.run()
    worker_done.wait(timeout=5)

    result = str(worker_result.get("result") or "error")
    if result == "restart":
        return RESTART_EXIT_CODE
    if result == "success":
        return RESTART_EXIT_CODE
    if result in ("failed", "error"):
        return 1
    return 0


def _parse_rate_limit_wait(error_str: str) -> int | None:
    """Return seconds to wait if error_str is a 429 rate limit error, else None."""
    if "429" not in error_str:
        return None
    brace = error_str.find("{")
    if brace == -1:
        return 60
    try:
        body = json.loads(error_str[brace:])
        err = body.get("error", body)
        secs = int(err.get("resets_in_seconds", 0))
        return max(secs, 60)
    except Exception:  # pylint: disable=broad-except
        return 60


def _run_selfdev_operator(repo_root: Path, model: str | None, max_turns: int) -> int:
    """Run the operator agent with a read-only TUI for observation."""
    from avoid_agent.selfdev.operator import stream_operator_to_tui

    tui = TUI(on_submit=lambda _text: None, model=model, auto_spinner_on_submit=False, read_only=True)
    tui.set_phase("operator starting")
    tui.push_item(StatusItem(text="Operator agent managing selfdev workflow..."))

    worker_done = threading.Event()
    worker_result: dict[str, object] = {"success": False}

    _MAX_RATE_LIMIT_RETRIES = 5

    def worker() -> None:
        rate_limit_retries = 0
        try:
            tui._start_spinner()  # pylint: disable=protected-access
            while True:
                result = stream_operator_to_tui(
                    repo_root=repo_root,
                    tui=tui,
                    stderr_streamer=_stream_selfdev_headless_stderr,
                    model=model,
                    max_cycles=10,
                    max_turns_per_worker=max_turns,
                )
                worker_result.update(result)

                if result.get("success"):
                    tui.set_phase("operator done")
                    tui.push_item(StatusItem(text="Operator completed successfully."))
                    break

                error = str(result.get("error", "unknown error"))
                wait_secs = _parse_rate_limit_wait(error)

                if wait_secs is None or rate_limit_retries >= _MAX_RATE_LIMIT_RETRIES:
                    tui.report_error(f"Operator failed: {error}")
                    break

                rate_limit_retries += 1
                wait_with_buffer = wait_secs + 60
                resume_time = datetime.fromtimestamp(time.time() + wait_with_buffer)
                tui.set_phase("rate limited")
                tui.push_item(StatusItem(
                    text=(
                        f"Usage limit reached. Resuming at {resume_time.strftime('%H:%M:%S')}. "
                        f"Waiting {wait_with_buffer}s "
                        f"(attempt {rate_limit_retries}/{_MAX_RATE_LIMIT_RETRIES})..."
                    )
                ))

                remaining = wait_with_buffer
                while remaining > 0:
                    chunk = min(remaining, 60)
                    time.sleep(chunk)
                    remaining -= chunk
                    if remaining > 0:
                        tui.set_phase(f"resuming in {remaining}s")

                tui.set_phase("operator restarting")
                tui.push_item(StatusItem(text="Resuming operator after rate limit reset..."))
                tui._start_spinner()  # pylint: disable=protected-access
        except Exception as e:  # pylint: disable=broad-except
            worker_result["error"] = str(e)
            tui.report_error(str(e))
        finally:
            tui._stop_spinner()  # pylint: disable=protected-access
            worker_done.set()
            tui.stop()

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    tui.run()
    worker_done.wait(timeout=5)

    if worker_result.get("success"):
        return RESTART_EXIT_CODE
    return 1
