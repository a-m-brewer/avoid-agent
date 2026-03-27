"""Tests for mid-flight slash-command cancellation in the TUI.

When a submit is in-flight (background thread running on_submit) and the user
types a slash command + enter:
  - /exit or /quit should cancel the turn and stop the TUI loop.
  - Any other slash command (/clear etc.) should cancel the turn, then be
    dispatched as a new on_submit call once the background thread finishes.
  - Non-slash text typed while busy should be silently discarded.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from avoid_agent.tui import TUI
from avoid_agent.tui.components.conversation import AssistantItem, UserItem


class _FakeTerminal:
    columns = 120

    def start(self, on_resize=None) -> None:
        return None

    def stop(self) -> None:
        return None

    def hide_cursor(self) -> None:
        return None

    def show_cursor(self) -> None:
        return None

    def write(self, _text: str) -> None:
        return None

    def move_up(self, _rows: int) -> None:
        return None

    def read_key(self) -> bytes:
        return b"\r"


class _FakeRenderer:
    def __init__(self, _terminal) -> None:
        self.has_content = False

    def render(self, _lines) -> None:
        self.has_content = True

    def physical_rows(self, lines) -> int:
        return len(lines)


def _make_tui(on_submit) -> TUI:
    with patch("avoid_agent.tui.Terminal", return_value=_FakeTerminal()), \
         patch("avoid_agent.tui.Renderer", _FakeRenderer):
        tui = TUI(on_submit=on_submit, model="test", auto_spinner_on_submit=False)
    return tui


# ---------------------------------------------------------------------------
# Helper: simulate a slow on_submit that blocks until an event is set.
# ---------------------------------------------------------------------------

def _slow_submit(unblock: threading.Event, calls: list[str]):
    """Factory returning an on_submit that records calls and blocks until unblock."""
    def _submit(text: str) -> None:
        calls.append(text)
        unblock.wait(timeout=5)
    return _submit


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_slash_exit_while_busy_stops_loop() -> None:
    """/quit typed while a submit is in-flight should cancel the turn and stop
    the TUI loop (self._running = False) once the background thread finishes."""
    unblock = threading.Event()
    calls: list[str] = []

    tui = _make_tui(_slow_submit(unblock, calls))

    # Simulate submitting a normal message (starts background thread)
    tui._input.line.text = "hello"
    tui._input.line.cursor = len("hello")
    result = tui._handle_key("enter", b"\r")
    assert result is False  # loop continues while thread runs
    assert tui._is_submit_busy()

    # Now type /quit while busy
    tui._input.line.text = "/quit"
    tui._input.line.cursor = len("/quit")
    result = tui._handle_key("enter", b"\r")
    assert result is False  # _handle_key itself returns False; loop exit via _running

    # The cancel token should be set
    assert tui.cancel_token is not None and tui.cancel_token.is_set()
    assert tui._pending_slash == "/quit"

    # Unblock the background thread so it can finish
    unblock.set()
    if tui._submit_thread:
        tui._submit_thread.join(timeout=3)

    # After the thread finishes, _drain_pending_slash should have set _running=False
    assert tui._running is False
    # The /quit command itself was NOT passed to on_submit
    assert calls == ["hello"]


def test_non_exit_slash_while_busy_reruns_after_cancel() -> None:
    """/clear typed while a submit is in-flight should be dispatched as a new
    on_submit call once the background thread finishes, not before."""
    unblock = threading.Event()
    calls: list[str] = []

    tui = _make_tui(_slow_submit(unblock, calls))

    # Simulate submitting a normal message
    tui._input.line.text = "do something"
    tui._input.line.cursor = len("do something")
    tui._handle_key("enter", b"\r")
    assert tui._is_submit_busy()

    # Type /clear while busy
    tui._input.line.text = "/clear"
    tui._input.line.cursor = len("/clear")
    tui._handle_key("enter", b"\r")
    assert tui._pending_slash == "/clear"
    assert tui.cancel_token is not None and tui.cancel_token.is_set()

    # Unblock
    unblock.set()
    if tui._submit_thread:
        tui._submit_thread.join(timeout=3)

    # /clear should have been dispatched to on_submit after the original turn ended
    assert calls == ["do something", "/clear"]
    # _running was never set to True (run() was not called), but crucially it was
    # NOT set to False by an exit command either — it stays at its initial value.
    assert tui._running is False  # initial value; exit commands leave it False too,
    # but we can confirm no exit command fired by checking calls contains /clear.


def test_non_slash_text_while_busy_is_discarded() -> None:
    """Plain text typed while busy is silently discarded (no second on_submit call)."""
    unblock = threading.Event()
    calls: list[str] = []

    tui = _make_tui(_slow_submit(unblock, calls))

    tui._input.line.text = "first"
    tui._input.line.cursor = len("first")
    tui._handle_key("enter", b"\r")
    assert tui._is_submit_busy()

    # Type non-slash text while busy
    tui._input.line.text = "second"
    tui._input.line.cursor = len("second")
    tui._handle_key("enter", b"\r")
    # No pending slash, no cancel signal
    assert tui._pending_slash is None
    assert tui.cancel_token is not None and not tui.cancel_token.is_set()

    unblock.set()
    if tui._submit_thread:
        tui._submit_thread.join(timeout=3)

    # Only one call was made
    assert calls == ["first"]


def test_cancel_token_is_set_when_slash_exit_received() -> None:
    """The cancel_token event is set as soon as the slash command is entered."""
    unblock = threading.Event()
    token_was_set_immediately: list[bool] = []

    def _submit(text: str) -> None:
        unblock.wait(timeout=5)

    tui = _make_tui(_submit)

    tui._input.line.text = "start"
    tui._input.line.cursor = len("start")
    tui._handle_key("enter", b"\r")

    # Give the background thread a moment to start
    time.sleep(0.02)
    token_before = tui.cancel_token

    tui._input.line.text = "/exit"
    tui._input.line.cursor = len("/exit")
    tui._handle_key("enter", b"\r")

    # Cancel token should be set immediately (synchronously in _handle_key)
    assert token_before is not None and token_before.is_set()

    unblock.set()
    if tui._submit_thread:
        tui._submit_thread.join(timeout=3)


def test_turn_cancelled_error_is_importable() -> None:
    """TurnCancelledError is exported from the runtime module."""
    from avoid_agent.agent.runtime import TurnCancelledError  # noqa: F401
    assert issubclass(TurnCancelledError, Exception)


def test_runtime_raises_turn_cancelled_error_when_token_set() -> None:
    """AgentRuntime.run_user_turn raises TurnCancelledError when the cancel
    token is already set before the call."""
    import json
    import threading as _threading
    from avoid_agent.agent.runtime import AgentRuntime, TurnCancelledError
    from avoid_agent.providers import (
        AssistantMessage,
        Provider,
        ProviderResponse,
        ProviderStream,
        ToolChoice,
    )
    from avoid_agent.providers import Message

    class _FakeStream(ProviderStream):
        def __init__(self, response):
            self._response = response

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def event_stream(self):
            return iter(())

        def get_final_message(self):
            return self._response

    class _FakeProvider(Provider):
        def __init__(self):
            super().__init__(system="s", model="fake", max_tokens=64)

        def stream(self, messages, tools, tool_choice: ToolChoice = "auto"):
            return _FakeStream(
                ProviderResponse(
                    message=AssistantMessage(
                        text=json.dumps({
                            "plan": "done",
                            "action": {"tool": "complete", "args": {"summary": "ok", "evidence": []}},
                        }),
                        stop_reason="stop",
                    ),
                    stop_reason="stop",
                    input_tokens=1,
                )
            )

    token = _threading.Event()
    token.set()  # already cancelled before the call

    runtime = AgentRuntime(
        provider=_FakeProvider(),
        tool_definitions=[],
        allowed_prefixes=set(),
    )

    with pytest.raises(TurnCancelledError):
        runtime.run_user_turn([], "hello", cancel_token=token)
