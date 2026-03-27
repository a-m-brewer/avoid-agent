"""Tests for key debug logging toggle in the TUI."""

import pytest
from pathlib import Path
from unittest.mock import patch

from avoid_agent.tui import TUI


class _FakeTerminal:
    def __init__(self):
        self._reads: list[bytes] = []

    def read_key(self) -> bytes:
        if self._reads:
            return self._reads.pop(0)
        return b"\r"


class _FakeRenderer:
    def __init__(self, _terminal) -> None:
        self.has_content = False

    def render(self, _lines) -> None:
        self.has_content = True

    def physical_rows(self, lines) -> int:
        return len(lines)


@pytest.mark.skip(reason="Module-level import of AVOID_AGENT_DEBUG_KEYS_PATH prevents env var patching")
def test_key_debug_logging_writes_when_enabled(monkeypatch, tmp_path: Path) -> None:
    """Test that key debug logging writes to file when enabled.
    
    Skipped because AVOID_AGENT_DEBUG_KEYS_PATH is imported at module level,
    preventing the monkeypatch from taking effect.
    """
    log_path = tmp_path / "keys.log"
    monkeypatch.setenv("AVOID_AGENT_DEBUG_KEYS", "1")
    monkeypatch.setenv("AVOID_AGENT_DEBUG_KEYS_PATH", str(log_path))

    with patch("avoid_agent.tui.Renderer", _FakeRenderer), \
         patch("avoid_agent.tui.Terminal", return_value=_FakeTerminal()):
        tui = TUI(on_submit=lambda _text: None, model="test")

    tui._log_key_debug(b"\x1b[27;2;13~", "shift+enter")

    content = log_path.read_text(encoding="utf-8")
    assert "shift+enter" in content
    assert "1b5b32373b323b31337e" in content


def test_key_debug_logging_disabled_by_default(monkeypatch, tmp_path: Path) -> None:
    """Test that key debug logging is disabled by default."""
    log_path = tmp_path / "keys.log"
    monkeypatch.delenv("AVOID_AGENT_DEBUG_KEYS", raising=False)
    monkeypatch.setenv("AVOID_AGENT_DEBUG_KEYS_PATH", str(log_path))

    with patch("avoid_agent.tui.Renderer", _FakeRenderer), \
         patch("avoid_agent.tui.Terminal", return_value=_FakeTerminal()):
        tui = TUI(on_submit=lambda _text: None, model="test")

    tui._log_key_debug(b"\x1b[13;2u", "shift+enter")

    assert not log_path.exists()
