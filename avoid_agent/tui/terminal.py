"""Terminal handling for raw input and resize events."""

import os
import select
import shutil
import signal
import sys
import termios
import time
import tty
from typing import Callable

_ESCAPE_SEQUENCE_TIMEOUT_S = 0.12


class Terminal:
    """Handles raw terminal input and resize events."""

    def __init__(self):
        # File descriptor for the terminal input (stdin)
        self._fd = sys.stdin.fileno()
        self._old_settings = None
        self._resize_cb: Callable[[], None] | None = None
        self._old_sigwinch = None

    @property
    def columns(self) -> int:
        return shutil.get_terminal_size().columns

    @property
    def rows(self) -> int:
        return shutil.get_terminal_size().rows

    def start(self, on_resize: Callable[[], None] | None = None) -> None:
        """Start the terminal in raw mode and set up resize handling."""
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)

        # Enable bracketed paste mode so pasted multiline text can be handled as one message.
        self.write("\x1b[?2004h")
        # Enable xterm modifyOtherKeys so modified Enter (Shift+Enter, etc.) is distinguishable.
        self.write("\x1b[>4;2m")

        if on_resize:
            self._resize_cb = on_resize
            self._old_sigwinch = signal.signal(signal.SIGWINCH, self._handle_sigwinch)

    def stop(self) -> None:
        # Disable input modes before restoring terminal settings.
        self.write("\x1b[>4;0m")
        self.write("\x1b[?2004l")

        if self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None

        if self._old_sigwinch is not None:
            signal.signal(signal.SIGWINCH, self._old_sigwinch)
            self._old_sigwinch = None

        self._resize_cb = None

    def _handle_sigwinch(self, _signum, _frame):
        if self._resize_cb:
            self._resize_cb()

    def _is_complete_escape_sequence(self, data: bytes) -> bool:
        """Return True when an escape sequence is complete enough to parse."""
        if data == b"\x1b":
            return False

        if not data.startswith(b"\x1b"):
            return True

        # CSI sequence: ESC [ ... <final-byte>
        if data.startswith(b"\x1b["):
            if len(data) < 3:
                return False
            last = data[-1]
            return 0x40 <= last <= 0x7E

        # SS3 sequence: ESC O <char>
        if data.startswith(b"\x1bO"):
            return len(data) >= 3

        # Meta/Alt sequence (ESC + key)
        return len(data) >= 2

    def _read_escape_sequence(self) -> bytes:
        data = b"\x1b"
        deadline = time.monotonic() + _ESCAPE_SEQUENCE_TIMEOUT_S

        while time.monotonic() < deadline:
            if self._is_complete_escape_sequence(data):
                break

            timeout = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self._fd], [], [], timeout)
            if not ready:
                break

            data += os.read(self._fd, 1)

        return data

    def read_key(self) -> bytes:
        data = os.read(self._fd, 1)

        if data == b"\x1b":
            data = self._read_escape_sequence()

        elif data[0] & 0x80:
            # Multi-byte UTF-8: first byte tells us how many more to read
            if data[0] & 0xE0 == 0xC0:
                remaining = 1
            elif data[0] & 0xF0 == 0xE0:
                remaining = 2
            elif data[0] & 0xF8 == 0xF0:
                remaining = 3
            else:
                remaining = 0
            for _ in range(remaining):
                data += os.read(self._fd, 1)

        return data

    def write(self, data: str) -> None:
        sys.stdout.write(data)
        sys.stdout.flush()

    def move_up(self, n: int) -> None:
        if n > 0:
            self.write(f"\x1b[{n}A")

    def hide_cursor(self) -> None:
        self.write("\x1b[?25l")

    def show_cursor(self) -> None:
        self.write("\x1b[?25h")
