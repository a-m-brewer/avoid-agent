"""Tests for terminal input buffering and mode toggles."""

import avoid_agent.tui.terminal as terminal_module
from avoid_agent.tui.terminal import Terminal


def test_read_key_collects_chunked_escape_sequence(monkeypatch) -> None:
    class FakeStdin:
        @staticmethod
        def fileno() -> int:
            return 99

    monkeypatch.setattr(terminal_module.sys, "stdin", FakeStdin())

    terminal = Terminal()

    chunks = [bytes([byte]) for byte in b"\x1b[27;2;13~"]

    def fake_read(fd: int, _size: int) -> bytes:
        assert fd == 99
        return chunks.pop(0)

    def fake_select(_read, _write, _except, _timeout):
        return ([99], [], []) if chunks else ([], [], [])

    monkeypatch.setattr(terminal_module.os, "read", fake_read)
    monkeypatch.setattr(terminal_module.select, "select", fake_select)

    assert terminal.read_key() == b"\x1b[27;2;13~"


def test_start_and_stop_toggle_input_modes(monkeypatch) -> None:
    class FakeStdin:
        @staticmethod
        def fileno() -> int:
            return 7

    monkeypatch.setattr(terminal_module.sys, "stdin", FakeStdin())

    terminal = Terminal()
    writes: list[str] = []

    monkeypatch.setattr(terminal_module.termios, "tcgetattr", lambda _fd: [1, 2, 3])
    monkeypatch.setattr(terminal_module.tty, "setraw", lambda _fd: None)
    monkeypatch.setattr(terminal_module.termios, "tcsetattr", lambda _fd, _when, _settings: None)
    monkeypatch.setattr(terminal, "write", lambda data: writes.append(data))

    terminal.start()
    terminal.stop()

    assert "\x1b[?2004h" in writes
    assert "\x1b[>4;2m" in writes
    assert "\x1b[>4;0m" in writes
    assert "\x1b[?2004l" in writes
