class SpinnerComponent:
    """Component for rendering a spinner animation while the agent is thinking."""
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        self._frame = 0
        self.message = "thinking..."

    def tick(self) -> None:
        self._frame = (self._frame + 1) % len(self.FRAMES)

    def set_message(self, message: str) -> None:
        self.message = message

    def render(self, width: int) -> list[str]:
        line = f" {self.FRAMES[self._frame]} {self.message}"
        return [line[:width]]
