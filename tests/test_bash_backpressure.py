"""Tests for run_bash stdout backpressure behavior."""

import shlex
import sys

from avoid_agent.agent.tools.core import _BASH_QUIET_THRESHOLD, run_bash


def _python_c(code: str) -> str:
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def test_long_successful_output_is_suppressed():
    result = run_bash("seq 1 100")

    assert "Exit code: 0" in result.content
    assert "suppressed" in result.content


def test_short_successful_output_is_kept():
    result = run_bash("echo hello")

    assert "Exit code: 0" in result.content
    assert "hello" in result.content


def test_failed_command_never_suppresses_stdout():
    command = _python_c(
        "print(chr(10).join(str(i) for i in range(50))); import sys; sys.exit(1)"
    )

    result = run_bash(command)

    assert "Exit code: 1" in result.content
    assert "suppressed" not in result.content
    assert "STDOUT:" in result.content
    assert "0\n1\n2" in result.content
    assert "49" in result.content


def test_stderr_is_always_present_even_when_stdout_is_suppressed():
    command = _python_c(
        "print(chr(10).join(str(i) for i in range(30))); import sys; sys.stderr.write('warn\\n')"
    )

    result = run_bash(command)

    assert "Exit code: 0" in result.content
    assert "suppressed" in result.content
    assert "STDERR:" in result.content
    assert "warn" in result.content


def test_bash_quiet_threshold_is_20():
    assert _BASH_QUIET_THRESHOLD == 20
