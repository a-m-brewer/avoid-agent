"""Tests for clearing TUI conversation state."""

from avoid_agent.__main__ import _clear_learning_session_files
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


def test_replace_last_assistant_updates_existing_item(monkeypatch) -> None:
    monkeypatch.setattr(tui_module, "Terminal", _FakeTerminal)
    monkeypatch.setattr(tui_module, "Renderer", _FakeRenderer)

    tui = tui_module.TUI(on_submit=lambda _text: None, model="test")
    tui.report_info("raw json")

    tui.replace_last_assistant("formatted message")

    assert tui._conversation.items[-1].text == "formatted message"


def test_clear_learning_session_files_returns_removed_markdown_count(tmp_path) -> None:
    learnings_dir = tmp_path / ".learnings" / "sessions"
    learnings_dir.mkdir(parents=True)

    (learnings_dir / "session-a.md").write_text("a", encoding="utf-8")
    (learnings_dir / "session-b.md").write_text("b", encoding="utf-8")
    (learnings_dir / "ignore.txt").write_text("c", encoding="utf-8")

    removed = _clear_learning_session_files(learnings_dir)

    assert removed == 2
    assert not (learnings_dir / "session-a.md").exists()
    assert not (learnings_dir / "session-b.md").exists()
    assert (learnings_dir / "ignore.txt").exists()
