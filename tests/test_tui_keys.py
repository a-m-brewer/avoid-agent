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


# modifyOtherKeys arrow key sequences (modifier code 1 = no modifier)
def test_parse_key_modifyotherkeys_up_no_modifier() -> None:
    """\\x1b[1;1A is sent by some terminals for plain Up in modifyOtherKeys mode."""
    assert parse_key(b"\x1b[1;1A") == "up"


def test_parse_key_modifyotherkeys_down_no_modifier() -> None:
    """\\x1b[1;1B is sent by some terminals for plain Down in modifyOtherKeys mode."""
    assert parse_key(b"\x1b[1;1B") == "down"


def test_parse_key_modifyotherkeys_right_no_modifier() -> None:
    assert parse_key(b"\x1b[1;1C") == "right"


def test_parse_key_modifyotherkeys_left_no_modifier() -> None:
    assert parse_key(b"\x1b[1;1D") == "left"


def test_parse_key_modifyotherkeys_up_shift() -> None:
    assert parse_key(b"\x1b[1;2A") == "shift+up"


def test_parse_key_modifyotherkeys_up_alt() -> None:
    assert parse_key(b"\x1b[1;3A") == "alt+up"


def test_parse_key_modifyotherkeys_up_ctrl() -> None:
    assert parse_key(b"\x1b[1;5A") == "ctrl+up"


def test_parse_key_plain_arrow_sequences_unaffected() -> None:
    """Standard arrow escape sequences must still parse correctly."""
    assert parse_key(b"\x1b[A") == "up"
    assert parse_key(b"\x1b[B") == "down"
    assert parse_key(b"\x1b[C") == "right"
    assert parse_key(b"\x1b[D") == "left"


def test_parse_key_standalone_bracket_is_printable() -> None:
    """A bare '[' byte must remain a printable character, not an arrow key."""
    assert parse_key(b"[") == "["
