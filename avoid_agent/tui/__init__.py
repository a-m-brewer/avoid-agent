"""Text-based user interface for interacting with the agent in the terminal."""

import os
import sys
import threading
import time

from avoid_agent.infra.config import env_flag, AVOID_AGENT_DEBUG_KEYS_PATH
from avoid_agent.tui.components.conversation import AssistantItem, ConversationComponent, PermissionItem, ToolCallItem, UserItem
from avoid_agent.tui.components.input_component import InputComponent
from avoid_agent.tui.components.spinner import SpinnerComponent
from avoid_agent.tui.components.status_bar import StatusBarComponent
from avoid_agent.tui.history import History
from avoid_agent.tui.keys import parse_key
from avoid_agent.tui.renderer import Renderer
from avoid_agent.tui.terminal import Terminal

# Slash commands that exit the app.
_EXIT_COMMANDS = frozenset({"/exit", "/quit"})


class TUI:
    """A simple text-based user interface for interacting with the agent in the terminal."""

    def __init__(self,
                 on_submit,
                 model: str,
                 prompt: str = "You: ",
                 auto_spinner_on_submit: bool = True,
                 read_only: bool = False):
        self._terminal = Terminal()
        self._input = InputComponent(prompt=prompt)
        self._history = History()
        self._renderer = Renderer(self._terminal)
        self._conversation = ConversationComponent()
        self._status = StatusBarComponent(model=model)
        self.on_submit = on_submit
        self._read_only = read_only

        # Spinner
        self._spinner = SpinnerComponent()
        self._busy = False
        self._displayed_tool_ids: set[str] = set()
        self._lock = threading.Lock()
        self._spinner_thread: threading.Thread | None = None
        self._in_paste = False
        self._running = False
        self._auto_spinner_on_submit = auto_spinner_on_submit

        # Background submit thread and mid-flight slash-command support.
        self._submit_thread: threading.Thread | None = None
        # cancel_token is created fresh for each submit; the background thread
        # receives it and passes it down to the AgentRuntime so it can exit
        # cleanly between steps.
        self.cancel_token: threading.Event | None = None
        # Slash command typed while a submit is in-flight is stored here so it
        # can be executed once the background thread has finished.
        self._pending_slash: str | None = None

        self._debug_keys = env_flag("AVOID_AGENT_DEBUG_KEYS")
        self._debug_keys_path = AVOID_AGENT_DEBUG_KEYS_PATH

    def _log_key_debug(self, data: bytes, key: str) -> None:
        if not self._debug_keys:
            return

        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} key={key!r} raw={data!r} hex={data.hex()}\n"
        try:
            parent = os.path.dirname(self._debug_keys_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._debug_keys_path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            # Best-effort debugging only.
            pass

    def _read_parsed_key(self) -> tuple[bytes, str]:
        data = self._terminal.read_key()
        key = parse_key(data)
        self._log_key_debug(data, key)
        return data, key

    def _on_resize(self) -> None:
        """Handle terminal resize events."""
        self._safe_render()

    def run(self) -> None:
        self._terminal.start(on_resize=self._on_resize)
        try:
            self._running = True
            self._safe_render()
            if self._read_only:
                # In read-only mode, only handle ctrl+c to exit.
                while self._running:
                    data, key = self._read_parsed_key()
                    if key in ("ctrl+c", "ctrl+d"):
                        break
            else:
                while self._running:
                    data, key = self._read_parsed_key()
                    if self._handle_key(key, data):
                        break
        finally:
            self._running = False
            # Wait for any in-flight submit thread before tearing down.
            if self._submit_thread is not None and self._submit_thread.is_alive():
                if self.cancel_token is not None:
                    self.cancel_token.set()
                self._submit_thread.join()
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._terminal.stop()

    def stop(self) -> None:
        self._running = False

    def push_item(self, item) -> None:
        if isinstance(item, ToolCallItem) and item.id:
            if item.id in self._displayed_tool_ids:
                return
            self._displayed_tool_ids.add(item.id)
        self._conversation.items.append(item)
        if isinstance(item, (UserItem, AssistantItem)):
            self._status.messages = sum(
                1 for i in self._conversation.items
                if isinstance(i, (UserItem, AssistantItem))
            )
        self._safe_render()

    def update_tool_status(self, tool_call_id: str, status: str) -> None:
        for item in self._conversation.items:
            if isinstance(item, ToolCallItem) and item.id == tool_call_id:
                item.status = status
                self._safe_render()
                return

    def update_tokens(self, tokens: int) -> None:
        self._status.tokens = tokens
        self._safe_render()

    def set_model(self, model: str) -> None:
        self._status.model = model
        self._safe_render()

    def set_thinking_enabled(self, enabled: bool) -> None:
        self._status.thinking_enabled = bool(enabled)
        self._safe_render()

    def set_effort(self, effort: str) -> None:
        self._status.effort = effort
        self._safe_render()

    def set_warning(self, text: str | None) -> None:
        self._status.warning = text
        self._safe_render()

    def set_phase(self, phase: str | None) -> None:
        self._status.phase = phase
        self._safe_render()

    def set_progress(self, current: int, total: int) -> None:
        self._status.progress_current = current
        self._status.progress_total = total
        self._safe_render()

    def append_chunk(self, text: str) -> None:
        if self._conversation.items and isinstance(
            self._conversation.items[-1], AssistantItem
        ):
            self._conversation.items[-1].text += text
        else:
            self._conversation.items.append(AssistantItem(text=text))
            self._status.messages = sum(
                1 for i in self._conversation.items
                if isinstance(i, (UserItem, AssistantItem))
            )
        self._safe_render()

    def clear_conversation(self) -> None:
        self._conversation.items.clear()
        self._displayed_tool_ids.clear()
        self._status.messages = 0
        self._status.tokens = 0
        self._safe_render()

    def report_error(self, message: str) -> None:
        """Append an error to the last assistant response, or push a new one."""
        if self._conversation.items and isinstance(self._conversation.items[-1], AssistantItem):
            self._conversation.items[-1].text += f"\n\n[Error: {message}]"
        else:
            self._conversation.items.append(AssistantItem(text=f"[Error: {message}]"))
        self._safe_render()

    def report_info(self, message: str) -> None:
        """Append an informational message to the conversation."""
        if self._conversation.items and isinstance(self._conversation.items[-1], AssistantItem):
            self._conversation.items[-1].text += f"\n\n{message}"
        else:
            self._conversation.items.append(AssistantItem(text=message))
        self._safe_render()

    def replace_last_assistant(self, text: str) -> None:
        """Replace the most recent assistant item, or append one if absent."""
        if self._conversation.items and isinstance(self._conversation.items[-1], AssistantItem):
            self._conversation.items[-1].text = text
        else:
            self._conversation.items.append(AssistantItem(text=text))
            self._status.messages = sum(
                1 for i in self._conversation.items
                if isinstance(i, (UserItem, AssistantItem))
            )
        self._safe_render()

    def _safe_render(self) -> None:
        with self._lock:
            self._render()

    def _render(self) -> None:
        self._terminal.hide_cursor()
        width = self._terminal.columns
        if self._read_only:
            lines = (
                self._conversation.render(width)
                + (self._spinner.render(width) if self._busy else [])
                + self._status.render(width)
            )
            if self._renderer.has_content:
                self._terminal.write("\x1b8")
            self._renderer.render(lines)
            self._terminal.write("\x1b7")

            # Position cursor at bottom of status bar
            status_lines = self._status.render(width)
            rows_up = self._renderer.physical_rows(status_lines)
            self._terminal.move_up(rows_up)
            self._terminal.write("\r")
            return

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

        # Position cursor within the rendered (possibly multiline, wrapped) input.
        row_in_input, col_in_row = self._input.cursor_position(width)
        if row_in_input > 0:
            self._terminal.write(f"\x1b[{row_in_input}B")
        if col_in_row > 0:
            self._terminal.write(f"\x1b[{col_in_row}C")
        self._terminal.show_cursor()

    def _is_submit_busy(self) -> bool:
        """Return True if a background submit thread is currently running."""
        return self._submit_thread is not None and self._submit_thread.is_alive()

    def _handle_key(self, key: str, data: bytes) -> bool:
        """Returns True if the loop should exit."""
        if key == "ctrl+c" or key == "ctrl+d":
            if self._is_submit_busy():
                # Cancel the in-flight turn and wait for the thread to stop
                # before exiting, so the app shuts down cleanly.
                if self.cancel_token is not None:
                    self.cancel_token.set()
                if self._submit_thread is not None:
                    self._submit_thread.join()
            return True
        elif key == "paste_start":
            self._in_paste = True
        elif key == "paste_end":
            self._in_paste = False
        elif key == "ctrl+v":
            # Ctrl+V: attempt to capture an image from the system clipboard.
            # Text paste via Ctrl+V is not handled here — the terminal's
            # bracketed-paste mode (paste_start / paste_end) covers that.
            self._try_capture_clipboard_image()
        elif key == "enter":
            if self._in_paste:
                self._input.line.insert("\n")
            else:
                text = self._input.line.clear()
                has_images = bool(self._input.pending_images)
                if text.strip() or has_images:
                    if text.strip():
                        self._history.push(text)
                    stripped = text.strip()

                    if self._is_submit_busy():
                        # A turn is in-flight.  Only slash commands are accepted;
                        # everything else is silently ignored so the user doesn't
                        # accidentally queue up a second request.
                        if stripped.startswith("/"):
                            # Store the slash command and signal cancellation.
                            self._pending_slash = stripped
                            if self.cancel_token is not None:
                                self.cancel_token.set()
                            self.report_info(
                                f"Cancelling current turn, then running: {stripped}"
                            )
                        self._safe_render()
                        return False

                    # Not busy — dispatch normally.
                    if stripped in _EXIT_COMMANDS:
                        self._safe_render()
                        return True

                    # Handle /paste-image slash command
                    if stripped == "/paste-image":
                        self._try_capture_clipboard_image()
                        return False

                    # Drain pending images and submit.
                    images = list(self._input.pending_images)
                    self._input.pending_images.clear()

                    self._conversation.items.append(UserItem(text=text, images=images))
                    self._safe_render()
                    self._dispatch_submit(text, images=images)
                    return False
        elif key == "shift+enter":
            self._input.line.insert("\n")
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

    def _dispatch_submit(self, text: str) -> None:
        """Run on_submit in a background thread so the key loop stays live.

        When the thread finishes (normally or after cancellation) any pending
        slash command is dispatched.
        """
        token = threading.Event()
        self.cancel_token = token

        def _run() -> None:
            if self._auto_spinner_on_submit:
                self._start_spinner()
            try:
                self.on_submit(text)
            finally:
                if self._auto_spinner_on_submit:
                    self._stop_spinner()
                # Dispatch any slash command that arrived mid-flight.
                self._drain_pending_slash()

        t = threading.Thread(target=_run, daemon=True)
        self._submit_thread = t
        t.start()

    def _drain_pending_slash(self) -> None:
        """Called from the background thread after it completes.

        Executes the slash command that was stashed while the turn was in-flight.
        Exit commands set self._running = False so the main loop exits cleanly.
        """
        cmd = self._pending_slash
        self._pending_slash = None
        self.cancel_token = None
        if cmd is None:
            return

        if cmd in _EXIT_COMMANDS:
            self._running = False
            return

        # For non-exit slash commands, submit them as a new turn.  We call
        # on_submit directly here (we are already on a background thread so
        # the spinner start/stop is safe).
        if self._auto_spinner_on_submit:
            self._start_spinner()
        try:
            self.on_submit(cmd)
        finally:
            if self._auto_spinner_on_submit:
                self._stop_spinner()

    def ask_permission(self, command: str) -> str:
        """Show an inline permission prompt and return 'allow', 'save', or 'deny'."""
        self._stop_spinner()
        item = PermissionItem(command=command)
        self._conversation.items.append(item)
        self._safe_render()

        while True:
            _, key = self._read_parsed_key()
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
        self._spinner.set_message("thinking...")
        self._spinner_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spinner_thread.start()

    def set_spinner_message(self, message: str) -> None:
        self._spinner.set_message(message)
        if self._busy:
            self._safe_render()

    def reset_spinner_message(self) -> None:
        self._spinner.set_message("thinking...")
        if self._busy:
            self._safe_render()

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


    def pick_from_list(self, title: str, options: list[str]) -> str | None:
        """Interactive searchable picker. Returns selected option or None if cancelled."""
        if not options:
            self.report_error("No options available.")
            return None

        query = ""
        selected = 0
        view_offset = 0
        window_size = 15

        while True:
            filtered = [opt for opt in options if query.lower() in opt.lower()]
            if selected >= len(filtered):
                selected = max(0, len(filtered) - 1)

            max_offset = max(0, len(filtered) - window_size)
            view_offset = min(view_offset, max_offset)
            if selected < view_offset:
                view_offset = selected
            elif selected >= view_offset + window_size:
                view_offset = selected - window_size + 1

            width = self._terminal.columns
            border = "─" * max(1, width)

            lines = self._conversation.render(width)
            if lines:
                lines.append("")
            lines.append(border)
            lines.append(title)
            lines.append(f"Search: {query}")
            lines.append("(type to filter, ↑/↓ to move, Enter to select, Esc/Ctrl+C to cancel)")
            preview = filtered[view_offset:view_offset + window_size]
            for i, opt in enumerate(preview):
                marker = ">" if i + view_offset == selected else " "
                lines.append(f"{marker} {opt}")
            remaining_below = len(filtered) - (view_offset + len(preview))
            if remaining_below > 0:
                lines.append(f"... and {remaining_below} more")
            if not filtered:
                lines.append("(no matches)")
            lines.append(border)

            self._terminal.hide_cursor()
            self._renderer.render(lines)
            self._terminal.show_cursor()

            data, key = self._read_parsed_key()
            if key in ("ctrl+c", "ctrl+d", "esc", "escape"):
                return None
            if key == "up":
                if filtered:
                    selected = max(0, selected - 1)
                    if selected < view_offset:
                        view_offset = selected
                continue
            if key == "down":
                if filtered:
                    selected = min(len(filtered) - 1, selected + 1)
                    if selected >= view_offset + window_size:
                        view_offset = selected - window_size + 1
                continue
            if key == "enter":
                if filtered:
                    return filtered[selected]
                continue
            if key == "backspace":
                if query:
                    query = query[:-1]
                continue
            if len(key) == 1 and key.isprintable():
                query += key
                selected = 0
                continue
