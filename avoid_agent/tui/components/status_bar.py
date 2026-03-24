from avoid_agent.tui.style import dim


class StatusBarComponent:
    """"Component for rendering the status bar at the bottom of the TUI, showing model and token count."""
    def __init__(self, model: str):
        self.model = model
        self.tokens = 0
        self.messages = 0
        self.thinking_enabled: bool = False
        self.effort: str = "high"

    def render(self, width: int) -> list[str]:
        extras: list[str] = []
        if self.thinking_enabled:
            extras.append("thinking: on")
        if self.effort:
            extras.append(f"effort: {self.effort}")
        extras_text = (" | " + " | ".join(extras)) if extras else ""

        left = f" {self.model}{extras_text}"
        right = f"{self.messages} msgs | {self.tokens} tokens "
        gap = width - len(left) - len(right)
        if gap < 0:
            line = (left + right)[:width]
        else:
            line = left + " " * gap + right
        return [dim(line)]
