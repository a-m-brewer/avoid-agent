"""Terminal handling for raw input and resize events."""

import os
import select
import shutil
from signal import signal
import sys
import termios
import tty
from typing import Callable


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
        if on_resize:
            self._resize_cb = on_resize
            self._old_sigwinch = signal.signal(signal.SIGWINCH, self._handle_sigwinch)

    def stop(self) -> None:
        # Disable bracketed paste mode before restoring terminal settings.
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

    def read_key(self) -> bytes:
        data = os.read(self._fd, 1)

        if data == b"\x1b":
            # Could be ESC alone, or start of a sequence like \x1b[A
            r, _, _ = select.select([self._fd], [], [], 0.05)
            if r:
                data += os.read(self._fd, 32)

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
