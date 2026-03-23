"""Focused tests for provider stop-reason mapping."""

from avoid_agent.providers.anthropic import _map_stop_reason as map_anthropic_stop_reason
from avoid_agent.providers.openai import _map_stop_reason as map_openai_stop_reason


def test_openai_stop_reason_mapping():
    assert map_openai_stop_reason("tool_calls") == "tool_use"
    assert map_openai_stop_reason("length") == "length"
    assert map_openai_stop_reason("stop") == "stop"
    assert map_openai_stop_reason(None) == "stop"


def test_anthropic_stop_reason_mapping():
    assert map_anthropic_stop_reason("tool_use") == "tool_use"
    assert map_anthropic_stop_reason("max_tokens") == "length"
    assert map_anthropic_stop_reason("end_turn") == "stop"
    assert map_anthropic_stop_reason(None) == "stop"
