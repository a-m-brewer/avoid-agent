"""Tests for on-the-fly model switching command behavior."""

from avoid_agent.tui import TUI


def test_tui_set_model_updates_status(monkeypatch) -> None:
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

    monkeypatch.setattr(tui_module, "Terminal", _FakeTerminal)
    monkeypatch.setattr(tui_module, "Renderer", _FakeRenderer)

    tui = TUI(on_submit=lambda _text: None, model="anthropic/old")
    tui.set_model("openai/gpt-5")

    assert tui._status.model == "openai/gpt-5"
