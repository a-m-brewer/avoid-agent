"""History management for the TUI input line."""

class History:
    """Manages the history of user inputs in the TUI, allowing navigation with up/down keys."""
    def __init__(self):
        self._entries: list[str] = []
        self._index: int | None = None
        self._draft: str = ""

    def push(self, text: str) -> None:
        if text and (not self._entries or self._entries[-1] != text):
            self._entries.append(text)
        self._index = None
        self._draft = ""

    def up(self, current: str) -> str | None:
        if not self._entries:
            return None
        if self._index is None:
            self._draft = current
            self._index = len(self._entries) - 1
        elif self._index > 0:
            self._index -= 1
        return self._entries[self._index]

    def down(self) -> str | None:
        if self._index is None:
            return None
        if self._index < len(self._entries) - 1:
            self._index += 1
            return self._entries[self._index]
        else:
            self._index = None
            return self._draft
