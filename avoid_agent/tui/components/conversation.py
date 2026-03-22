from dataclasses import dataclass
from typing import Union

from avoid_agent.tui.style import bold, cyan, yellow, gray


@dataclass
class UserItem:
    """Represents a message from the user."""

    text: str


@dataclass
class AssistantItem:
    """Represents a message from the assistant."""

    text: str = ""


@dataclass
class ToolCallItem:
    """Represents a call to a tool."""

    name: str
    arguments: dict


@dataclass
class ToolResultItem:
    """Represents the result of a tool call."""

    name: str
    content: str


ConversationItem = Union[UserItem, AssistantItem, ToolCallItem, ToolResultItem]


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
            lines = self._wrap(f"You: {item.text}", width)
            return [bold(cyan(l)) for l in lines]
        elif isinstance(item, AssistantItem):
            return self._wrap(item.text, width) if item.text else []
        elif isinstance(item, ToolCallItem):
            args = ", ".join(f"{k}={repr(v)[:20]}" for k, v in item.arguments.items())
            lines = self._wrap(f"  > {item.name}({args})", width)
            return [yellow(l) for l in lines]
        elif isinstance(item, ToolResultItem):
            first_line = item.content.splitlines()[0] if item.content else ""
            lines = self._wrap(f"  < {item.name}: {first_line}", width)
            return [gray(l) for l in lines]
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
