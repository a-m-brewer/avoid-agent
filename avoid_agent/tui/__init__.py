"""Text-based user interface for interacting with the agent in the terminal."""

import sys
import threading
import time

from avoid_agent.tui.components.conversation import AssistantItem, ConversationComponent, PermissionItem, UserItem
from avoid_agent.tui.components.input_component import InputComponent
from avoid_agent.tui.components.spinner import SpinnerComponent
from avoid_agent.tui.components.status_bar import StatusBarComponent
from avoid_agent.tui.history import History
from avoid_agent.tui.keys import parse_key
from avoid_agent.tui.renderer import Renderer
from avoid_agent.tui.terminal import Terminal


class TUI:
    """A simple text-based user interface for interacting with the agent in the terminal."""

    def __init__(self,
                 on_submit,
                 model: str,
                 prompt: str = "You: "):
        self._terminal = Terminal()
        self._input = InputComponent(prompt=prompt)
        self._history = History()
        self._renderer = Renderer(self._terminal)
        self._conversation = ConversationComponent()
        self._status = StatusBarComponent(model=model)
        self.on_submit = on_submit

        # Spinner
        self._spinner = SpinnerComponent()
        self._busy = False
        self._lock = threading.Lock()
        self._spinner_thread: threading.Thread | None = None


    def run(self) -> None:
        self._terminal.start()
        try:
            self._safe_render()
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
        self._safe_render()

    def update_tokens(self, tokens: int) -> None:
        self._status.tokens = tokens
        self._safe_render()

    def append_chunk(self, text: str) -> None:
        if self._conversation.items and isinstance(
            self._conversation.items[-1], AssistantItem
        ):
            self._conversation.items[-1].text += text
        else:
            self._conversation.items.append(AssistantItem(text=text))
        self._safe_render()

    def clear_conversation(self) -> None:
        self._conversation.items.clear()
        self._safe_render()

    def report_error(self, message: str) -> None:
        """Append an error to the last assistant response, or push a new one."""
        if self._conversation.items and isinstance(self._conversation.items[-1], AssistantItem):
            self._conversation.items[-1].text += f"\n\n[Error: {message}]"
        else:
            self._conversation.items.append(AssistantItem(text=f"[Error: {message}]"))
        self._safe_render()

    def _safe_render(self) -> None:
        with self._lock:
            self._render()

    def _render(self) -> None:
        self._terminal.hide_cursor()
        width = self._terminal.columns
        lines = (
            self._conversation.render(width)
            + (self._spinner.render(width) if self._busy else [])
            + self._input.render(width)
            + self._status.render(width)
        )
        if self._renderer.has_content:
            self._terminal.write("\x1b8")
        self._renderer.render(lines)
        self._terminal.write("\x1b7")

        # Move up to the start of the input line, accounting for physical wrapping
        input_lines = self._input.render(width)
        status_lines = self._status.render(width)
        rows_up = self._renderer.physical_rows(input_lines + status_lines)
        self._terminal.move_up(rows_up)
        self._terminal.write("\r")

        # Position cursor within the (possibly wrapped) input line
        cursor_col = self._input.cursor_col
        row_in_input = cursor_col // width
        col_in_row = cursor_col % width
        if row_in_input > 0:
            self._terminal.write(f"\x1b[{row_in_input}B")
        if col_in_row > 0:
            self._terminal.write(f"\x1b[{col_in_row}C")
        self._terminal.show_cursor()

    def _handle_key(self, key: str, data: bytes) -> bool:
        """Returns True if the loop should exit."""
        if key == "ctrl+c" or key == "ctrl+d":
            return True
        elif key == "enter":
            text = self._input.line.clear()
            if text.strip():
                self._history.push(text)
                if text.strip() in ("exit", "quit"):
                    self._safe_render()
                    return True
                self._conversation.items.append(UserItem(text=text))
                self._safe_render()
                self._start_spinner()
                self.on_submit(text)
                self._stop_spinner()

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

        self._safe_render()
        return False

    def ask_permission(self, command: str) -> str:
        """Show an inline permission prompt and return 'allow', 'save', or 'deny'."""
        self._stop_spinner()
        item = PermissionItem(command=command)
        self._conversation.items.append(item)
        self._safe_render()

        while True:
            data = self._terminal.read_key()
            key = parse_key(data)
            if key in ("y", "Y"):
                item.result = "allowed once"
                self._safe_render()
                self._start_spinner()
                return "allow"
            if key in ("s", "S"):
                item.result = "allowed (saved)"
                self._safe_render()
                self._start_spinner()
                return "save"
            if key in ("n", "N", "ctrl+c", "ctrl+d"):
                item.result = "denied"
                self._safe_render()
                self._start_spinner()
                return "deny"

    # Spinner
    def _start_spinner(self) -> None:
        self._busy = True
        self._spinner_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spinner_thread.start()

    def _stop_spinner(self) -> None:
        self._busy = False
        if self._spinner_thread:
            self._spinner_thread.join()
            self._spinner_thread = None
        self._safe_render()

    def _spin_loop(self) -> None:
        while self._busy:
            self._spinner.tick()
            self._safe_render()
            time.sleep(0.1)
