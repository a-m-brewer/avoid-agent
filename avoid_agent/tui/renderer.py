""""Renderer for the TUI output."""
import re

from avoid_agent.tui.terminal import Terminal

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class Renderer:
    """Handles rendering the agent's output to the terminal."""
    def __init__(self, terminal: Terminal):
        self._terminal = terminal
        self._last_lines: list[str] = []

    @property
    def has_content(self) -> bool:
        return bool(self._last_lines)

    def render(self, lines: list[str]) -> None:
        if not self._last_lines:
            self._first_render(lines)
        else:
            self._update(lines)
        self._last_lines = list(lines)

    def clear(self) -> None:
        self._last_lines = []

    def _first_render(self, lines: list[str]) -> None:
        for line in lines:
            self._terminal.write(line + "\r\n")

    def physical_rows(self, lines: list[str]) -> int:
        """Count physical terminal rows occupied by a list of logical lines."""
        width = self._terminal.columns
        total = 0
        for line in lines:
            plain = _ANSI_RE.sub("", line)
            total += max(1, -(-len(plain) // width))  # ceiling division
        return total

    def _update(self, lines: list[str]) -> None:
        first_change = self._first_changed(lines)
        if first_change is None:
            return

        self._terminal.write("\x1b[?2026h")  # begin synchronized update

        lines_to_move_up = self.physical_rows(self._last_lines[first_change:])
        if lines_to_move_up > 0:
            self._terminal.move_up(lines_to_move_up)

        self._terminal.write("\x1b[J")

        for line in lines[first_change:]:
            self._terminal.write(line + "\r\n")

        self._terminal.write("\x1b[?2026l")  # end synchronized update

    def _first_changed(self, lines: list[str]) -> int | None:
        for i in range(min(len(lines), len(self._last_lines))):
            if lines[i] != self._last_lines[i]:
                return i
        if len(lines) != len(self._last_lines):
            return min(len(lines), len(self._last_lines))
        return None
