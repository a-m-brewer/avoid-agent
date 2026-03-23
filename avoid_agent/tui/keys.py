"""Key parsing for terminal input."""

import re

ESCAPE_SEQUENCES: dict[bytes, str] = {
    b"\x1b[A": "up",
    b"\x1b[B": "down",
    b"\x1b[C": "right",
    b"\x1b[D": "left",
    b"\x1b[H": "home",
    b"\x1b[F": "end",
    b"\x1b[2~": "insert",
    b"\x1b[3~": "delete",
    b"\x1b[5~": "pageup",
    b"\x1b[6~": "pagedown",
    b"\x1bOP": "f1",
    b"\x1bOQ": "f2",
    b"\x1bOR": "f3",
    b"\x1bOS": "f4",
    b"\x1b[200~": "paste_start",
    b"\x1b[201~": "paste_end",
}

CTRL_KEYS: dict[bytes, str] = {
    b"\x01": "ctrl+a",
    b"\x02": "ctrl+b",
    b"\x03": "ctrl+c",
    b"\x04": "ctrl+d",
    b"\x05": "ctrl+e",
    b"\x06": "ctrl+f",
    b"\x07": "ctrl+g",
    b"\x08": "ctrl+h",
    b"\x09": "tab",
    b"\x0a": "enter",
    b"\x0b": "ctrl+k",
    b"\x0c": "ctrl+l",
    b"\x0d": "enter",
    b"\x0e": "ctrl+n",
    b"\x0f": "ctrl+o",
    b"\x10": "ctrl+p",
    b"\x11": "ctrl+q",
    b"\x12": "ctrl+r",
    b"\x13": "ctrl+s",
    b"\x14": "ctrl+t",
    b"\x15": "ctrl+u",
    b"\x16": "ctrl+v",
    b"\x17": "ctrl+w",
    b"\x18": "ctrl+x",
    b"\x19": "ctrl+y",
    b"\x1a": "ctrl+z",
    b"\x1b": "escape",
    b"\x7f": "backspace",
}

_CSI_U_ENTER_RE = re.compile(r"^\x1b\[13;(\d+)u$")
_XTERM_MOD_ENTER_RE = re.compile(r"^\x1b\[27;(\d+);13~$")
_CSI_MOD_ENTER_Z_RE = re.compile(r"^\x1b\[13;(\d+)z$")
_CSI_MOD_ENTER_M_RE = re.compile(r"^\x1b\[13;(\d+)M$")


def _mod_code_to_prefix(code: int) -> str:
    """Translate xterm/kitty modifier code to key prefix.

    xterm/kitty modified key sequences encode modifiers as 1 + bitmask:
      Shift=1, Alt=2, Ctrl=4, Super=8
    So code 2 => Shift, 5 => Ctrl, 6 => Ctrl+Shift, etc.
    """
    mask = code - 1
    parts: list[str] = []
    if mask & 4:
        parts.append("ctrl")
    if mask & 2:
        parts.append("alt")
    if mask & 1:
        parts.append("shift")
    if mask & 8:
        parts.append("super")
    return "+".join(parts)


def _parse_modified_enter(data: bytes) -> str | None:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None

    for pattern in (
        _CSI_U_ENTER_RE,
        _XTERM_MOD_ENTER_RE,
        _CSI_MOD_ENTER_Z_RE,
        _CSI_MOD_ENTER_M_RE,
    ):
        match = pattern.match(text)
        if not match:
            continue
        mod_code = int(match.group(1))
        if mod_code <= 1:
            return "enter"
        prefix = _mod_code_to_prefix(mod_code)
        return f"{prefix}+enter" if prefix else "enter"

    return None


def parse_key(data: bytes) -> str:
    if data in ESCAPE_SEQUENCES:
        return ESCAPE_SEQUENCES[data]

    modified_enter = _parse_modified_enter(data)
    if modified_enter is not None:
        return modified_enter

    if data in CTRL_KEYS:
        return CTRL_KEYS[data]

    try:
        char = data.decode("utf-8")
        if char.isprintable():
            return char
    except UnicodeDecodeError:
        pass

    return repr(data)
