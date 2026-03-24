"""Tests for clearing TUI conversation state."""

import avoid_agent.tui as tui_module


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


def test_clear_conversation_resets_tokens_and_messages(monkeypatch) -> None:
    monkeypatch.setattr(tui_module, "Terminal", _FakeTerminal)
    monkeypatch.setattr(tui_module, "Renderer", _FakeRenderer)

    tui = tui_module.TUI(on_submit=lambda _text: None, model="test")

    tui.update_tokens(123)
    tui._status.messages = 4

    tui.clear_conversation()

    assert tui._status.tokens == 0
    assert tui._status.messages == 0


def test_reset_spinner_message_restores_thinking_label(monkeypatch) -> None:
    monkeypatch.setattr(tui_module, "Terminal", _FakeTerminal)
    monkeypatch.setattr(tui_module, "Renderer", _FakeRenderer)

    tui = tui_module.TUI(on_submit=lambda _text: None, model="test")
    tui.set_spinner_message("running tool: run_bash")

    tui.reset_spinner_message()

    assert tui._spinner.message == "thinking..."
