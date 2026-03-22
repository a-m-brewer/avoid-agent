"""Component for rendering the user input line and cursor."""
from avoid_agent.tui.input_line import InputLine
from avoid_agent.tui.style import bg_dark


class InputComponent:
    """Component for rendering the user input line and cursor."""
    def __init__(self, prompt: str = "You: "):
        self.prompt = prompt
        self.line = InputLine()

    def render(self, width: int) -> list[str]:
        content = " " + self.prompt + self.line.text
        padded = content + " " * max(0, width - len(content))
        return [bg_dark(padded)]

    @property
    def cursor_col(self) -> int:
        return 1 + len(self.prompt) + self.line.cursor
