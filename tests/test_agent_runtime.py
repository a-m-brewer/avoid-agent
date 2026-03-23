"""Tests for the structured agent runtime loop."""

from pathlib import Path

import avoid_agent.agent.tools.core  # noqa: F401  # registers built-in tools

from avoid_agent.agent.runtime import AgentRuntime, RuntimeEvent
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    Provider,
    ProviderResponse,
    ProviderStream,
    ProviderToolCall,
)


class FakeStream(ProviderStream):
    def __init__(self, response: ProviderResponse):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def event_stream(self):
        return iter(())

    def get_final_message(self) -> ProviderResponse:
        return self._response


class FakeProvider(Provider):
    def __init__(self, responses: list[ProviderResponse]):
        super().__init__(system="test", model="fake", max_tokens=1024)
        self._responses = responses
        self.calls = 0

    def stream(self, messages: list[Message], tools):
        response = self._responses[self.calls]
        self.calls += 1
        return FakeStream(response)

    def compact(self, messages: list[Message], keep_last: int = 6) -> list[Message]:
        return messages


def _response(message: AssistantMessage) -> ProviderResponse:
    return ProviderResponse(
        message=message,
        stop_reason=message.stop_reason,
        input_tokens=123,
    )


def test_runtime_retries_and_marks_failure_for_prose_only_mutation_completion():
    provider = FakeProvider(
        responses=[
            _response(AssistantMessage(text="I updated the file.", stop_reason="stop")),
            _response(AssistantMessage(text="I changed the implementation.", stop_reason="stop")),
            _response(AssistantMessage(text="Implemented the fix.", stop_reason="stop")),
        ]
    )
    events: list[RuntimeEvent] = []
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=[],
        allowed_prefixes=set(),
        on_event=events.append,
    )

    result = runtime.run_user_turn([], "please update the implementation")

    assert provider.calls == 3
    assert result.messages[-1].stop_reason == "error"
    assert "mutation-capable tool" in result.messages[-1].error_message
    validation_events = [event for event in events if event.type == "validation_error"]
    assert len(validation_events) == 3


def test_runtime_executes_tool_calls_and_continues_until_final_assistant_message(tmp_path: Path):
    target = tmp_path / "runtime-write.txt"
    provider = FakeProvider(
        responses=[
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_1",
                            name="write_file",
                            arguments={"path": str(target), "content": "hello"},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            _response(AssistantMessage(text="Done.", stop_reason="stop")),
        ]
    )
    events: list[RuntimeEvent] = []
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=[],
        allowed_prefixes=set(),
        on_event=events.append,
    )

    result = runtime.run_user_turn([], "please write the file")

    assert provider.calls == 2
    assert target.read_text() == "hello"
    assert any(event.type == "tool_result" for event in events)
    assert result.messages[-1].text == "Done."
    assert result.messages[-1].stop_reason == "stop"


def test_runtime_allows_clarifying_question_without_tools():
    provider = FakeProvider(
        responses=[
            _response(AssistantMessage(text="Which file should I edit?", stop_reason="stop")),
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=[],
        allowed_prefixes=set(),
    )

    result = runtime.run_user_turn([], "please edit the file")

    assert provider.calls == 1
    assert result.messages[-1].text == "Which file should I edit?"
    assert result.messages[-1].stop_reason == "stop"
