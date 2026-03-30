from dataclasses import dataclass, field
from typing import Union

from avoid_agent.tui.style import bg_user, bold, cyan, yellow, gray, magenta


@dataclass
class UserItem:
    """Represents a message from the user."""

    text: str
    images: list = field(default_factory=list)


@dataclass
class AssistantItem:
    """Represents a message from the assistant."""

    text: str = ""


@dataclass
class ToolCallItem:
    """Represents a call to a tool."""

    id: str | None
    name: str
    arguments: dict
    status: str = "pending"  # pending | running | done | failed


@dataclass
class ToolResultItem:
    """Represents the result of a tool call."""

    id: str | None
    name: str
    content: str
    status: str = "done"


@dataclass
class StatusItem:
    """Represents provider status/reasoning breadcrumbs."""

    text: str


@dataclass
class PermissionItem:
    """Represents a pending or resolved permission prompt for run_bash."""

    command: str
    result: str = ""  # "" = waiting, "allowed" | "saved" | "denied"


ConversationItem = Union[UserItem, AssistantItem, ToolCallItem, ToolResultItem, PermissionItem, StatusItem]


class ConversationComponent:
    """Manages the conversation history and rendering for the TUI."""

    def __init__(self):
        self.items: list[ConversationItem] = []

    def render(self, width: int) -> list[str]:
        lines = []
        for item in self.items:
            lines.extend(self._render_item(item, width))
        return lines

    def _render_item(self, item: ConversationItem, width: int) -> list[str]:
        if isinstance(item, UserItem):
            lines = self._wrap(f"  You: {item.text}", width)
            padded = [l + " " * max(0, width - len(l)) for l in lines]
            return [bg_user(l) for l in padded] + [""]  # trailing gap

        elif isinstance(item, AssistantItem):
            lines = self._wrap(item.text, width) if item.text else []
            return lines + [""]  # trailing gap

        elif isinstance(item, ToolCallItem):
            args = ", ".join(f"{k}={repr(v)[:20]}" for k, v in item.arguments.items())
            lines = self._wrap(f"  > [{item.status}] {item.name}({args})", width)
            return [yellow(l) for l in lines]
        elif isinstance(item, ToolResultItem):
            first_line = item.content.splitlines()[0] if item.content else ""
            lines = self._wrap(f"  < [{item.status}] {item.name}: {first_line}", width)
            return [gray(l) for l in lines]
        elif isinstance(item, StatusItem):
            lines = self._wrap(f"  · {item.text}", width)
            return [cyan(l) for l in lines]
        elif isinstance(item, PermissionItem):
            cmd_lines = self._wrap(f"  ? run_bash: {item.command}", width)
            rendered = [magenta(l) for l in cmd_lines]
            if item.result:
                rendered.append(gray(f"  {item.result}"))
            else:
                rendered.append(magenta("  Allow? [y] once  [s] save  [n] deny"))
            return rendered
        return []

    def _wrap(self, text: str, width: int) -> list[str]:
        if not text:
            return []
        lines = []
        for paragraph in text.splitlines():
            if not paragraph:
                lines.append("")
                continue
            while len(paragraph) > width:
                lines.append(paragraph[:width])
                paragraph = paragraph[width:]
            lines.append(paragraph)
        return lines
