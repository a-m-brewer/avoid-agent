"""Text-based user interface for interacting with the agent in the terminal."""

import sys

from avoid_agent.tui.components.conversation import AssistantItem, ConversationComponent, UserItem
from avoid_agent.tui.components.input_component import InputComponent
from avoid_agent.tui.history import History
from avoid_agent.tui.keys import parse_key
from avoid_agent.tui.renderer import Renderer
from avoid_agent.tui.terminal import Terminal


class TUI:
    """A simple text-based user interface for interacting with the agent in the terminal."""

    def __init__(self, on_submit, prompt: str = "You: "):
        self._terminal = Terminal()
        self._input = InputComponent(prompt=prompt)
        self._history = History()
        self._renderer = Renderer(self._terminal)
        self._conversation = ConversationComponent()
        self.on_submit = on_submit

    def run(self) -> None:
        self._terminal.start()
        try:
            self._render()
            while True:
                data = self._terminal.read_key()
                key = parse_key(data)
                if self._handle_key(key, data):
                    break
        finally:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._terminal.stop()

    def push_item(self, item) -> None:
        self._conversation.items.append(item)
        self._render()

    def append_chunk(self, text: str) -> None:
        if self._conversation.items and isinstance(
            self._conversation.items[-1], AssistantItem
        ):
            self._conversation.items[-1].text += text
        else:
            self._conversation.items.append(AssistantItem(text=text))
        self._render()

    def clear_conversation(self) -> None:
        self._conversation.items.clear()
        self._render()

    def _render(self) -> None:
        width = self._terminal.columns
        lines = self._conversation.render(width) + self._input.render(width)
        if self._renderer.has_content:
            self._terminal.write("\x1b8")
        self._renderer.render(lines)
        self._terminal.write("\x1b7")
        self._terminal.move_up(1)
        self._terminal.write("\r")
        if self._input.cursor_col > 0:
            self._terminal.write(f"\x1b[{self._input.cursor_col}C")

    def _handle_key(self, key: str, data: bytes) -> bool:
        """Returns True if the loop should exit."""
        if key == "ctrl+c" or key == "ctrl+d":
            return True
        elif key == "enter":
            text = self._input.line.clear()
            if text.strip():
                self._history.push(text)
                if text.strip() in ("exit", "quit"):
                    self._render()
                    return True
                self._conversation.items.append(UserItem(text=text))
                self._render()
                self.on_submit(text)
        elif key == "backspace":
            self._input.line.backspace()
        elif key == "delete":
            self._input.line.delete_forward()
        elif key == "left":
            self._input.line.move_left()
        elif key == "right":
            self._input.line.move_right()
        elif key in ("ctrl+a", "home"):
            self._input.line.move_home()
        elif key in ("ctrl+e", "end"):
            self._input.line.move_end()
        elif key == "ctrl+k":
            self._input.line.kill_to_end()
        elif key == "ctrl+u":
            self._input.line.move_home()
            self._input.line.kill_to_end()
        elif key == "up":
            result = self._history.up(self._input.line.text)
            if result is not None:
                self._input.line.text = result
                self._input.line.move_end()
        elif key == "down":
            result = self._history.down()
            if result is not None:
                self._input.line.text = result
                self._input.line.move_end()
        elif len(key) == 1:
            self._input.line.insert(key)

        self._render()
        return False
