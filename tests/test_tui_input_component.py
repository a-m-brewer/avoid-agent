"""Tests for multiline input rendering and cursor positioning."""

from avoid_agent.tui.components.input_component import InputComponent


def _strip_bg(lines: list[str]) -> list[str]:
    # bg_dark wraps lines with ANSI sequences; keep the visible text only.
    # Current style adds '\x1b[48;5;236m' prefix and '\x1b[0m' suffix.
    cleaned: list[str] = []
    for line in lines:
        line = line.replace("\x1b[48;5;236m", "")
        line = line.replace("\x1b[0m", "")
        cleaned.append(line)
    return cleaned


def test_multiline_input_does_not_embed_newline_chars_in_rendered_lines() -> None:
    component = InputComponent(prompt="You: ")
    component.line.text = "hello\nworld"
    component.line.cursor = len(component.line.text)

    rendered = _strip_bg(component.render(width=20))

    assert all("\n" not in line for line in rendered)
    assert rendered[0].startswith(" You: hello")
    assert rendered[1].startswith("      world")


def test_cursor_position_tracks_explicit_newlines_and_wrapping() -> None:
    component = InputComponent(prompt="You: ")
    component.line.text = "abc\ndef"
    component.line.cursor = len(component.line.text)

    # Prefix is " You: " (6 chars). On second logical line, cursor is after "def".
    row, col = component.cursor_position(width=10)

    assert row == 1
    assert col == 9
