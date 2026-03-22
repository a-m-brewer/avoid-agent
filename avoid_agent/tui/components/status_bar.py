from avoid_agent.tui.style import dim


class StatusBarComponent:
    """"Component for rendering the status bar at the bottom of the TUI, showing model and token count."""
    def __init__(self, model: str):
        self.model = model
        self.tokens = 0

    def render(self, width: int) -> list[str]:
        left = f" {self.model}"
        right = f"{self.tokens} tokens "
        gap = width - len(left) - len(right)
        if gap < 0:
            line = (left + right)[:width]
        else:
            line = left + " " * gap + right
        return [dim(line)]
