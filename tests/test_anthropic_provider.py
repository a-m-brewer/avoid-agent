"""Tests for Anthropic provider helpers."""

from unittest.mock import Mock

from avoid_agent.providers.anthropic import AnthropicProvider
from avoid_agent.providers import UserMessage


def test_add_conversation_cache_control_marks_recent_user_breakpoints():
    messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "tool output"},
                {"type": "text", "text": "state"},
            ],
        },
        {"role": "assistant", "content": "reply 2"},
        {"role": "user", "content": "latest"},
    ]

    result = AnthropicProvider._add_conversation_cache_control(messages)

    assert result[4]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert result[2]["content"][1]["cache_control"] == {"type": "ephemeral"}
    assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_request_metrics_reports_wire_size_and_cache_breakpoints():
    provider = AnthropicProvider(
        system="sys",
        model="claude-sonnet-4-6",
        max_tokens=1024,
        api_key="test-key",
    )
    provider._client = Mock()

    metrics = provider.request_metrics([UserMessage(text="hello")], tools=[])

    assert metrics["provider"] == "anthropic"
    assert metrics["wire_chars"] > 0
    assert metrics["messages"] == 1
    assert metrics["cache_breakpoints"] == 1
