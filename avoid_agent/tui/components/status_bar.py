from avoid_agent.tui.style import dim


class StatusBarComponent:
    """"Component for rendering the status bar at the bottom of the TUI, showing model and token count."""
    def __init__(self, model: str):
        self.model = model
        self.tokens = 0
        self.messages = 0
        self.thinking_enabled: bool = False
        self.effort: str = "high"
        self.warning: str | None = None
        self.phase: str | None = None
        self.progress_current: int = 0
        self.progress_total: int = 0
        self.vision_enabled: bool = True

    @staticmethod
    def _progress_bar(current: int, total: int, width: int = 10) -> str:
        if total <= 0:
            return ""
        current = max(0, min(current, total))
        filled = int((current / total) * width)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {current}/{total}"

    def render(self, width: int) -> list[str]:
        extras: list[str] = []
        if self.phase:
            extras.append(f"phase: {self.phase}")
        progress = self._progress_bar(self.progress_current, self.progress_total)
        if progress:
            extras.append(progress)
        if self.thinking_enabled:
            extras.append("thinking: on")
        if self.effort:
            extras.append(f"effort: {self.effort}")
        if not self.vision_enabled:
            extras.append("no vision")
        extras_text = (" | " + " | ".join(extras)) if extras else ""

        left = f" {self.model}{extras_text}"
        right = f"{self.messages} msgs | {self.tokens} tokens"
        if self.warning:
            right += f" | {self.warning}"
        right += " "
        gap = width - len(left) - len(right)
        if gap < 0:
            line = (left + right)[:width]
        else:
            line = left + " " * gap + right
        return [dim(line)]
