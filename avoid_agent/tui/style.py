"""
ANSI escape codes for styling text in the terminal. """

def bold(text: str) -> str:
    return f"\x1b[1m{text}\x1b[0m"

def dim(text: str) -> str:
    return f"\x1b[2m{text}\x1b[0m"

def cyan(text: str) -> str:
    return f"\x1b[36m{text}\x1b[0m"

def yellow(text: str) -> str:
    return f"\x1b[33m{text}\x1b[0m"

def gray(text: str) -> str:
    return f"\x1b[90m{text}\x1b[0m"

def bg_dark(text: str) -> str:
    return f"\x1b[48;5;236m{text}\x1b[0m"

def bg_status(text: str) -> str:
    return f"\x1b[48;5;234m{text}\x1b[0m"

def bg_user(text: str) -> str:
    return f"\x1b[48;5;24m\x1b[97m{text}\x1b[0m"
