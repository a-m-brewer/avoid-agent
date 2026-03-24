"""Tests for headless mode."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import avoid_agent.agent.tools.core  # noqa: F401  # registers built-in tools

from avoid_agent.__main__ import _run_headless
from avoid_agent.agent.tools.finder import find_available_tools
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    Provider,
    ProviderResponse,
    ProviderStream,
    ProviderToolCall,
    ToolChoice,
    ToolResultMessage,
)


class FakeStream(ProviderStream):
    def __init__(self, response: ProviderResponse):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def event_stream(self):
        return iter(())

    def get_final_message(self) -> ProviderResponse:
        return self._response


class FakeProvider(Provider):
    def __init__(self, responses: list[ProviderResponse]):
        super().__init__(system="test", model="fake", max_tokens=1024)
        self._responses = responses
        self.calls = 0

    def stream(self, messages: list[Message], tools, tool_choice: ToolChoice = "auto"):
        response = self._responses[self.calls]
        self.calls += 1
        return FakeStream(response)


def _response(message: AssistantMessage) -> ProviderResponse:
    return ProviderResponse(
        message=message,
        stop_reason=message.stop_reason,
        input_tokens=42,
    )


def _complete(evidence: list[str], summary: str = "Done.") -> AssistantMessage:
    return AssistantMessage(
        text=json.dumps({
            "plan": "Stop after verified execution.",
            "action": {
                "tool": "complete",
                "args": {"summary": summary, "evidence": evidence},
            },
        }),
        stop_reason="stop",
    )


def _blocker(reason: str) -> AssistantMessage:
    return AssistantMessage(
        text=json.dumps({
            "plan": "Cannot continue.",
            "action": {"tool": "blocker", "args": {"reason": reason}},
        }),
        stop_reason="stop",
    )


class _FakeArgs:
    """Mimics argparse namespace for _run_headless."""
    def __init__(self, **kwargs):
        self.prompt = kwargs.get("prompt")
        self.session = kwargs.get("session")
        self.auto_approve = kwargs.get("auto_approve", False)
        self.model = kwargs.get("model")
        self.max_turns = kwargs.get("max_turns", 20)
        self.context_strategy = kwargs.get("context_strategy", "compact+window")
        self.no_session = kwargs.get("no_session", True)


def test_headless_single_turn_blocker(tmp_path: Path, capsys, monkeypatch):
    """Single-turn mode with a blocker response produces valid JSON on stdout."""
    fake_provider = FakeProvider([
        _response(_blocker("Need more info.")),
    ])

    monkeypatch.setenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4-6")
    monkeypatch.chdir(tmp_path)

    with patch("avoid_agent.__main__.providers.get_provider", return_value=fake_provider), \
         patch("avoid_agent.__main__.load_allowed", return_value=set()), \
         patch("avoid_agent.__main__.gather_initial_context", return_value=(str(tmp_path), "", "")), \
         patch("avoid_agent.__main__.build_system_prompt", return_value="test system prompt"):
        try:
            _run_headless(_FakeArgs(prompt="hello"))
        except SystemExit as e:
            assert e.code == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert result["turn"] == 1
    assert result["assistant_text"] is not None
    assert result["error"] is None


def test_headless_single_turn_tool_call(tmp_path: Path, capsys, monkeypatch):
    """Single-turn with a tool call and completion produces valid output."""
    target = tmp_path / "test.txt"
    fake_provider = FakeProvider([
        _response(AssistantMessage(
            tool_calls=[
                ProviderToolCall(
                    id="call_1",
                    name="write_file",
                    arguments={"path": str(target), "content": "hello"},
                )
            ],
            stop_reason="tool_use",
        )),
        _response(_complete(["call_1"], summary="Wrote the file.")),
    ])

    monkeypatch.chdir(tmp_path)

    with patch("avoid_agent.__main__.providers.get_provider", return_value=fake_provider), \
         patch("avoid_agent.__main__.load_allowed", return_value=set()), \
         patch("avoid_agent.__main__.gather_initial_context", return_value=(str(tmp_path), "", "")), \
         patch("avoid_agent.__main__.build_system_prompt", return_value="test system prompt"):
        try:
            _run_headless(_FakeArgs(prompt="write a file"))
        except SystemExit as e:
            assert e.code == 0

    assert target.read_text() == "hello"

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    assert result["assistant_text"] == "Wrote the file."
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["name"] == "write_file"


def test_headless_auto_approve_bash(tmp_path: Path, capsys, monkeypatch):
    """With --auto-approve, bash commands are allowed."""
    fake_provider = FakeProvider([
        _response(AssistantMessage(
            tool_calls=[
                ProviderToolCall(
                    id="call_1",
                    name="run_bash",
                    arguments={"command": "echo hi"},
                )
            ],
            stop_reason="tool_use",
        )),
        _response(_complete(["call_1"])),
    ])

    monkeypatch.chdir(tmp_path)

    with patch("avoid_agent.__main__.providers.get_provider", return_value=fake_provider), \
         patch("avoid_agent.__main__.load_allowed", return_value=set()), \
         patch("avoid_agent.__main__.gather_initial_context", return_value=(str(tmp_path), "", "")), \
         patch("avoid_agent.__main__.build_system_prompt", return_value="test system prompt"):
        try:
            _run_headless(_FakeArgs(prompt="run echo", auto_approve=True))
        except SystemExit as e:
            assert e.code == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["success"] is True
    # Check stderr for auto-approve event
    assert "permission_auto_approved" in captured.err


def test_headless_no_auto_approve_denies_bash(tmp_path: Path, capsys, monkeypatch):
    """Without --auto-approve, unknown bash commands are denied."""
    fake_provider = FakeProvider([
        _response(AssistantMessage(
            tool_calls=[
                ProviderToolCall(
                    id="call_1",
                    name="run_bash",
                    arguments={"command": "rm -rf /"},
                )
            ],
            stop_reason="tool_use",
        )),
        _response(_blocker("Command denied.")),
    ])

    monkeypatch.chdir(tmp_path)

    with patch("avoid_agent.__main__.providers.get_provider", return_value=fake_provider), \
         patch("avoid_agent.__main__.load_allowed", return_value=set()), \
         patch("avoid_agent.__main__.gather_initial_context", return_value=(str(tmp_path), "", "")), \
         patch("avoid_agent.__main__.build_system_prompt", return_value="test system prompt"):
        try:
            _run_headless(_FakeArgs(prompt="delete everything", auto_approve=False))
        except SystemExit as e:
            assert e.code == 0

    captured = capsys.readouterr()
    assert "permission_denied" in captured.err


def test_headless_multi_turn_stdin(tmp_path: Path, capsys, monkeypatch):
    """Multi-turn mode reads JSONL from stdin and outputs per-turn results."""
    fake_provider = FakeProvider([
        _response(_blocker("Got it.")),
        _response(_blocker("Understood.")),
    ])

    stdin_data = (
        '{"prompt": "first message"}\n'
        '{"prompt": "second message"}\n'
        '{"command": "quit"}\n'
    )

    monkeypatch.chdir(tmp_path)

    import io
    fake_stdin = io.StringIO(stdin_data)
    fake_stdin.isatty = lambda: False

    with patch("avoid_agent.__main__.providers.get_provider", return_value=fake_provider), \
         patch("avoid_agent.__main__.load_allowed", return_value=set()), \
         patch("avoid_agent.__main__.gather_initial_context", return_value=(str(tmp_path), "", "")), \
         patch("avoid_agent.__main__.build_system_prompt", return_value="test system prompt"), \
         patch("avoid_agent.__main__.sys") as mock_sys:
        mock_sys.stdin = fake_stdin
        mock_sys.stdin.isatty = fake_stdin.isatty

        stdout_lines = []
        stderr_lines = []
        mock_sys.stdout.write = lambda s: stdout_lines.append(s)
        mock_sys.stdout.flush = lambda: None
        mock_sys.stderr.write = lambda s: stderr_lines.append(s)
        mock_sys.stderr.flush = lambda: None
        mock_sys.exit = sys.exit

        _run_headless(_FakeArgs(prompt=None))

    # Should have 2 result JSONs on stdout
    results = [json.loads(line) for line in stdout_lines if line.strip()]
    assert len(results) == 2
    assert results[0]["turn"] == 1
    assert results[1]["turn"] == 2


def test_headless_max_turns_exceeded(tmp_path: Path, capsys, monkeypatch):
    """Multi-turn mode stops when max_turns is exceeded."""
    fake_provider = FakeProvider([
        _response(_blocker("Ok.")),
        _response(_blocker("Ok.")),
        _response(_blocker("Ok.")),
    ])

    stdin_data = (
        '{"prompt": "one"}\n'
        '{"prompt": "two"}\n'
        '{"prompt": "three"}\n'
    )

    monkeypatch.chdir(tmp_path)

    import io
    fake_stdin = io.StringIO(stdin_data)
    fake_stdin.isatty = lambda: False

    with patch("avoid_agent.__main__.providers.get_provider", return_value=fake_provider), \
         patch("avoid_agent.__main__.load_allowed", return_value=set()), \
         patch("avoid_agent.__main__.gather_initial_context", return_value=(str(tmp_path), "", "")), \
         patch("avoid_agent.__main__.build_system_prompt", return_value="test system prompt"), \
         patch("avoid_agent.__main__.sys") as mock_sys:
        mock_sys.stdin = fake_stdin
        mock_sys.stdin.isatty = fake_stdin.isatty

        stdout_lines = []
        stderr_lines = []
        mock_sys.stdout.write = lambda s: stdout_lines.append(s)
        mock_sys.stdout.flush = lambda: None
        mock_sys.stderr.write = lambda s: stderr_lines.append(s)
        mock_sys.stderr.flush = lambda: None
        mock_sys.exit = sys.exit

        _run_headless(_FakeArgs(prompt=None, max_turns=2))

    results = [json.loads(line) for line in stdout_lines if line.strip()]
    assert len(results) == 3  # 2 successes + 1 error
    assert results[2]["success"] is False
    assert "Max turns" in results[2]["error"]
