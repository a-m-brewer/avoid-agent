"""Component for rendering the user input line and cursor."""

from avoid_agent.tui.input_line import InputLine
from avoid_agent.tui.style import bg_dark, cyan


class InputComponent:
    """Component for rendering the user input line and cursor.

    ``pending_images`` is a list of :class:`~avoid_agent.tui.clipboard.ClipboardImage`
    objects queued for the next submission.  The component renders a compact
    indicator line beneath the text input when images are pending.
    """

    def __init__(self, prompt: str = "You: "):
        self.prompt = prompt
        self.line = InputLine()
        # ClipboardImage objects waiting to be attached to the next message.
        self.pending_images: list = []

    def _prefix(self) -> str:
        return " " + self.prompt

    def _plain_logical_lines(self) -> list[str]:
        """Return logical input lines with prompt/continuation prefixes applied."""
        prefix = self._prefix()
        raw_lines = self.line.text.split("\n")

        if not raw_lines:
            raw_lines = [""]

        rendered: list[str] = []
        continuation_prefix = " " * len(prefix)
        for index, raw in enumerate(raw_lines):
            if index == 0:
                rendered.append(prefix + raw)
            else:
                rendered.append(continuation_prefix + raw)

        return rendered

    @staticmethod
    def _wrap_count(length: int, width: int) -> int:
        if width <= 0:
            return 1
        if length <= 0:
            return 1
        return (length + width - 1) // width

    def _image_indicator_line(self, width: int) -> str | None:
        """Return a styled image-count indicator, or None if no images are queued."""
        if not self.pending_images:
            return None
        from avoid_agent.tui.clipboard import format_size
        count = len(self.pending_images)
        total_bytes = sum(img.size_bytes for img in self.pending_images)
        label = (
            f" [image: {format_size(total_bytes)}]"
            if count == 1
            else f" [{count} images: {format_size(total_bytes)}]"
        )
        padded = label + " " * max(0, width - len(label))
        return cyan(padded[:width])

    def render(self, width: int) -> list[str]:
        plain_lines = self._plain_logical_lines()

        if width <= 0:
            # Defensive fallback for tiny/unknown terminal sizes.
            return [bg_dark(plain_lines[0] if plain_lines else "")]

        lines: list[str] = []
        for logical_line in plain_lines:
            if logical_line == "":
                lines.append(bg_dark(" " * width))
                continue

            for i in range(0, len(logical_line), width):
                segment = logical_line[i : i + width]
                lines.append(bg_dark(segment + " " * (width - len(segment))))

        if not lines:
            lines.append(bg_dark(" " * width))

        # Append image indicator line below the text input when images are queued.
        indicator = self._image_indicator_line(width)
        if indicator is not None:
            lines.append(indicator)

        return lines

    def cursor_position(self, width: int) -> tuple[int, int]:
        """Return cursor (row, col) within rendered input lines.

        The image indicator line (if present) is rendered *below* the text
        input so it does not affect cursor positioning within the input itself.
        """
        if width <= 0:
            return (0, 0)

        prefix_len = len(self._prefix())
        before_cursor = self.line.text[: self.line.cursor]
        before_lines = before_cursor.split("\n")

        cursor_logical_line = len(before_lines) - 1
        cursor_col_in_logical = len(before_lines[-1]) if before_lines else 0

        row = 0

        all_raw_lines = self.line.text.split("\n")
        for index in range(cursor_logical_line):
            # Full prior logical lines consume wrapped rows.
            logical_len = prefix_len + len(all_raw_lines[index])
            row += self._wrap_count(logical_len, width)

        cursor_visual_col = prefix_len + cursor_col_in_logical
        row += cursor_visual_col // width
        col = cursor_visual_col % width

        return (row, col)
