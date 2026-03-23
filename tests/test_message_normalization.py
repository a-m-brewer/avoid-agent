"""Tests for provider message normalization."""

from avoid_agent.providers import (
    AssistantMessage,
    ProviderToolCall,
    ToolResultMessage,
    UserMessage,
    normalize_messages,
)


def test_normalize_messages_inserts_synthetic_tool_result_for_orphaned_tool_call():
    tool_call = ProviderToolCall(id="call_1", name="edit_file", arguments={"path": "x"})
    messages = [
        UserMessage(text="fix it"),
        AssistantMessage(tool_calls=[tool_call], stop_reason="tool_use"),
        UserMessage(text="actually never mind"),
    ]

    normalized = normalize_messages(messages)

    synthetic_results = [
        message
        for message in normalized
        if isinstance(message, ToolResultMessage) and message.tool_call_id == "call_1"
    ]
    assert len(synthetic_results) == 1
    assert synthetic_results[0].is_error is True
    assert synthetic_results[0].content == "No result provided"


def test_normalize_messages_skips_errored_assistant_messages():
    messages = [
        UserMessage(text="fix it"),
        AssistantMessage(text="partial", stop_reason="error", error_message="boom"),
        UserMessage(text="try again"),
    ]

    normalized = normalize_messages(messages)

    assert len(normalized) == 2
    assert isinstance(normalized[0], UserMessage)
    assert isinstance(normalized[1], UserMessage)
