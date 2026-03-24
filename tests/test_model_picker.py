"""Tests for searchable model picker behavior."""

import avoid_agent.tui as tui_module
from avoid_agent.providers import list_available_models


class _FakeTerminal:
    columns = 120

    def __init__(self, keys: list[bytes] | None = None) -> None:
        self._keys = keys or []

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
        if not self._keys:
            return b"\r"
        return self._keys.pop(0)


class _FakeRenderer:
    def __init__(self, _terminal) -> None:
        self.has_content = False
        self.calls = 0

    def render(self, _lines) -> None:
        self.has_content = True
        self.calls += 1

    def physical_rows(self, lines) -> int:
        return len(lines)


def test_list_available_models_includes_provider_prefix() -> None:
    models = list_available_models()
    assert len(models) > 0, "Expected at least one model"
    assert all("/" in m for m in models), "All models should have provider/model format"


def test_picker_can_filter_and_select(monkeypatch) -> None:
    keys = [b"g", b"p", b"t", b"\r"]

    def make_terminal():
        return _FakeTerminal(keys=keys)

    monkeypatch.setattr(tui_module, "Terminal", make_terminal)
    monkeypatch.setattr(tui_module, "Renderer", _FakeRenderer)

    tui = tui_module.TUI(on_submit=lambda _text: None, model="anthropic/old")
    selected = tui.pick_from_list("Select model", ["anthropic/claude", "openai/gpt-5"])

    assert selected == "openai/gpt-5"


def test_picker_does_not_append_to_conversation_on_navigation(monkeypatch) -> None:
    keys = [b"\x1b[B", b"\x1b[A", b"\r"]

    def make_terminal():
        return _FakeTerminal(keys=keys)

    monkeypatch.setattr(tui_module, "Terminal", make_terminal)
    monkeypatch.setattr(tui_module, "Renderer", _FakeRenderer)

    tui = tui_module.TUI(on_submit=lambda _text: None, model="anthropic/old")
    selected = tui.pick_from_list("Select model", ["anthropic/claude", "openai/gpt-5"])

    assert selected == "anthropic/claude"
    assert tui._conversation.items == []
