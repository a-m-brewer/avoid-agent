"""Tests for terminal key parsing."""

from avoid_agent.tui.keys import parse_key


def test_parse_key_shift_enter_csi_u_sequence() -> None:
    assert parse_key(b"\x1b[13;2u") == "shift+enter"


def test_parse_key_shift_enter_xterm_modified_sequence() -> None:
    assert parse_key(b"\x1b[27;2;13~") == "shift+enter"


def test_parse_key_shift_enter_csi_13_2z_sequence() -> None:
    assert parse_key(b"\x1b[13;2z") == "shift+enter"


def test_parse_key_shift_enter_csi_13_2m_sequence() -> None:
    assert parse_key(b"\x1b[13;2M") == "shift+enter"
