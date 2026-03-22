"""Key parsing for terminal input."""

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

def parse_key(data: bytes) -> str:
    if data in ESCAPE_SEQUENCES:
        return ESCAPE_SEQUENCES[data]

    if data in CTRL_KEYS:
        return CTRL_KEYS[data]

    try:
        char = data.decode("utf-8")
        if char.isprintable():
            return char
    except UnicodeDecodeError:
        pass

    return repr(data)
