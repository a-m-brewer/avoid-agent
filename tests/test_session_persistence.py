"""Tests for session persistence and migration."""

import json

from avoid_agent import session
from avoid_agent.providers import AssistantMessage, ProviderToolCall, ToolResultMessage, UserMessage


def test_save_and_load_round_trip_with_structured_messages(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "_SESSIONS_DIR", tmp_path)
    messages = [
        UserMessage(text="hello", timestamp=1),
        AssistantMessage(
            text="working",
            tool_calls=[ProviderToolCall(id="call_1", name="edit_file", arguments={"path": "x"})],
            stop_reason="tool_use",
            timestamp=2,
        ),
        ToolResultMessage(
            tool_call_id="call_1",
            tool_name="edit_file",
            content="Edit applied to x",
            timestamp=3,
        ),
    ]

    session.save_session("/repo", messages, "test")
    loaded = session.load_session("/repo", "test")

    assert loaded is not None
    assert loaded[0].timestamp == 1
    assert loaded[1].stop_reason == "tool_use"
    assert loaded[1].tool_calls[0].name == "edit_file"
    assert loaded[2].tool_name == "edit_file"


def test_load_session_migrates_legacy_flat_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "_SESSIONS_DIR", tmp_path)
    path = session.session_name_path("/repo", "default")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cwd": "/repo",
                "name": "default",
                "messages": [
                    {"type": "user", "text": "hello"},
                    {
                        "type": "assistant",
                        "text": "done",
                        "text_id": "msg_1",
                        "reasoning_items": [],
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "name": "edit_file",
                                "arguments": {"path": "x"},
                                "item_id": "fc_1",
                            }
                        ],
                    },
                    {"type": "tool_result", "tool_call_id": "call_1", "content": "Edit applied"},
                ],
            }
        )
    )

    loaded = session.load_session("/repo", "default")

    assert loaded is not None
    assert loaded[0].text == "hello"
    assert loaded[1].text == "done"
    assert loaded[1].tool_calls[0].item_id == "fc_1"
    assert loaded[1].stop_reason == "stop"
    assert loaded[2].content == "Edit applied"
