"""Component for rendering the user input line and cursor."""
from avoid_agent.tui.input_line import InputLine


class InputComponent:
    """Component for rendering the user input line and cursor."""
    def __init__(self, prompt: str = "You: "):
        self.prompt = prompt
        self.line = InputLine()

    def render(self, width: int) -> list[str]:
        content = self.prompt + self.line.text
        # Pad to full width so the background colour covers the whole line later
        padded = content + " " * max(0, width - len(content))
        return [padded]

    @property
    def cursor_col(self) -> int:
        return len(self.prompt) + self.line.cursor
