"""Tests for display formatting of structured terminal actions."""

import json

from avoid_agent.cli.tui import (
    _display_text_for_assistant_message,
    _extract_recent_learning_errors,
    _format_learning_suggestions,
    messages_to_items,
)
from avoid_agent.providers import AssistantMessage
from avoid_agent.tui.components.conversation import AssistantItem


def test_display_text_for_complete_uses_summary() -> None:
    message = AssistantMessage(
        text=json.dumps(
            {
                "plan": "Finish after verified execution.",
                "action": {
                    "tool": "complete",
                    "args": {
                        "summary": "Verification passed.",
                        "evidence": ["call_1"],
                    },
                },
            }
        ),
        stop_reason="stop",
    )

    assert _display_text_for_assistant_message(message) == "Verification passed."


def test_messages_to_items_renders_blocker_reason_not_raw_json() -> None:
    message = AssistantMessage(
        text=json.dumps(
            {
                "plan": "Cannot continue without input.",
                "action": {
                    "tool": "blocker",
                    "args": {"reason": "Need the target file path."},
                },
            }
        ),
        stop_reason="stop",
    )

    items = messages_to_items([message])

    assert len(items) == 1
    assert isinstance(items[0], AssistantItem)
    assert items[0].text == "Need the target file path."


def test_display_text_for_tool_action_uses_plan() -> None:
    message = AssistantMessage(
        text=json.dumps(
            {
                "plan": "Run the targeted test command.",
                "action": {
                    "tool": "run_bash",
                    "args": {"command": ".venv/bin/pytest -q tests/test_model_picker.py"},
                },
            }
        ),
        stop_reason="stop",
    )

    assert _display_text_for_assistant_message(message) == "Run the targeted test command."


def test_extract_recent_learning_errors_limits_to_three(tmp_path) -> None:
    newest = tmp_path / "newest.md"
    newest.write_text(
        "# Session\n## Errors\nfirst error\nsecond error\n## Notes\nignored\n",
        encoding="utf-8",
    )
    older = tmp_path / "older.md"
    older.write_text(
        "# Session\n## errors\nthird error\nfourth error\n",
        encoding="utf-8",
    )

    errors = _extract_recent_learning_errors([newest, older], limit=3)

    assert errors == ["first error", "second error", "third error"]


def test_format_learning_suggestions_handles_empty_and_populated() -> None:
    assert _format_learning_suggestions([]) == "No suggestions yet."
    assert _format_learning_suggestions(["Retry with stricter input validation", "Add a smoke test"]) == (
        "Suggestions:\n- Retry with stricter input validation\n- Add a smoke test"
    )
