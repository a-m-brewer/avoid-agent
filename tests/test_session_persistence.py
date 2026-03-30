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
            details={
                "plan": "Apply the edit.",
                "action": {"tool": "edit_file", "args": {"path": "x"}},
                "proof": {"kind": "file_change", "path": "x", "changed": True},
                "verification": {"status": "verified", "message": "Edited x."},
            },
        ),
    ]

    session.save_session("/repo", messages, "test")
    loaded = session.load_session("/repo", "test")

    assert loaded is not None
    assert loaded[0].timestamp == 1
    assert loaded[1].stop_reason == "tool_use"
    assert loaded[1].tool_calls[0].name == "edit_file"
    assert loaded[1].provider_state == {}
    assert loaded[2].tool_name == "edit_file"
    assert loaded[2].details["verification"]["status"] == "verified"


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
    assert loaded[1].provider_state == {}
    assert loaded[2].content == "Edit applied"


def test_save_session_compacts_large_tool_result_when_artifact_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "_SESSIONS_DIR", tmp_path)
    artifact_path = tmp_path / "artifact.txt"
    large_content = "x" * 3001
    artifact_path.write_text(large_content, encoding="utf-8")
    messages = [
        ToolResultMessage(
            tool_call_id="call_1",
            tool_name="read_file",
            content=large_content,
            details={
                "artifact": {
                    "path": str(artifact_path),
                    "chars": len(large_content),
                }
            },
        )
    ]

    session.save_session("/repo", messages, "test")
    raw = json.loads(session.session_name_path("/repo", "test").read_text(encoding="utf-8"))

    assert raw["messages"][0]["stored_externally"] is True
    assert "[tool result stored in artifact]" in raw["messages"][0]["content"]
    assert raw["messages"][0]["content"] != large_content


def test_load_session_restores_tool_result_content_from_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "_SESSIONS_DIR", tmp_path)
    artifact_path = tmp_path / "artifact.txt"
    large_content = "x" * 3001
    artifact_path.write_text(large_content, encoding="utf-8")
    path = session.session_name_path("/repo", "test")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 2,
                "cwd": "/repo",
                "name": "test",
                "messages": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_1",
                        "tool_name": "read_file",
                        "content": "[tool result stored in artifact]",
                        "stored_externally": True,
                        "details": {
                            "artifact": {
                                "path": str(artifact_path),
                                "chars": len(large_content),
                            }
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = session.load_session("/repo", "test")

    assert loaded is not None
    assert loaded[0].content == large_content


def test_save_and_load_round_trip_preserves_assistant_provider_state(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "_SESSIONS_DIR", tmp_path)
    messages = [
        AssistantMessage(
            text="done",
            provider_state={"response_id": "resp_123"},
        )
    ]

    session.save_session("/repo", messages, "test")
    loaded = session.load_session("/repo", "test")

    assert loaded is not None
    assert loaded[0].provider_state == {"response_id": "resp_123"}
