"""Tests for slash-prefixed exit commands in the TUI."""

import pytest

import avoid_agent.tui as tui_module
from avoid_agent.tui.components.conversation import UserItem


class _FakeTerminal:
    columns = 120

    def start(self) -> None:
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


@pytest.mark.parametrize("text", ["exit", "quit"])
def test_plain_exit_words_are_submitted_as_messages(monkeypatch, text: str) -> None:
    monkeypatch.setattr(tui_module, "Terminal", _FakeTerminal)
    monkeypatch.setattr(tui_module, "Renderer", _FakeRenderer)

    submitted: list[str] = []
    tui = tui_module.TUI(on_submit=submitted.append, model="test")
    monkeypatch.setattr(tui, "_start_spinner", lambda: None)
    monkeypatch.setattr(tui, "_stop_spinner", lambda: None)

    tui._input.line.text = text
    tui._input.line.cursor = len(text)

    should_exit = tui._handle_key("enter", b"\r")

    assert should_exit is False
    assert submitted == [text]
    assert isinstance(tui._conversation.items[-1], UserItem)
    assert tui._conversation.items[-1].text == text


@pytest.mark.parametrize("text", ["/exit", "/quit"])
def test_slash_exit_commands_terminate_without_submitting(monkeypatch, text: str) -> None:
    monkeypatch.setattr(tui_module, "Terminal", _FakeTerminal)
    monkeypatch.setattr(tui_module, "Renderer", _FakeRenderer)

    submitted: list[str] = []
    tui = tui_module.TUI(on_submit=submitted.append, model="test")

    tui._input.line.text = text
    tui._input.line.cursor = len(text)

    should_exit = tui._handle_key("enter", b"\r")

    assert should_exit is True
    assert submitted == []
    assert tui._conversation.items == []
