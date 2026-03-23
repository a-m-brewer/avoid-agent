"""Tests for key debug logging toggle in the TUI."""

from pathlib import Path

import avoid_agent.tui as tui_module


class _FakeTerminal:
    def __init__(self):
        self._reads: list[bytes] = []

    def read_key(self) -> bytes:
        if self._reads:
            return self._reads.pop(0)
        return b"\r"


def test_key_debug_logging_writes_when_enabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tui_module, "Terminal", _FakeTerminal)
    log_path = tmp_path / "keys.log"
    monkeypatch.setenv("AVOID_AGENT_DEBUG_KEYS", "1")
    monkeypatch.setenv("AVOID_AGENT_DEBUG_KEYS_PATH", str(log_path))

    tui = tui_module.TUI(on_submit=lambda _text: None, model="test")
    tui._log_key_debug(b"\x1b[27;2;13~", "shift+enter")

    content = log_path.read_text(encoding="utf-8")
    assert "shift+enter" in content
    assert "1b5b32373b323b31337e" in content


def test_key_debug_logging_disabled_by_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(tui_module, "Terminal", _FakeTerminal)
    log_path = tmp_path / "keys.log"
    monkeypatch.delenv("AVOID_AGENT_DEBUG_KEYS", raising=False)
    monkeypatch.setenv("AVOID_AGENT_DEBUG_KEYS_PATH", str(log_path))

    tui = tui_module.TUI(on_submit=lambda _text: None, model="test")
    tui._log_key_debug(b"\x1b[13;2u", "shift+enter")

    assert not log_path.exists()
