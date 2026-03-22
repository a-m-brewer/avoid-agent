""""Defines the Component protocol for TUI components."""
from typing import Protocol


class Component(Protocol):
    """Protocol for TUI components."""
    def render(self, width: int) -> list[str]:
        ...
