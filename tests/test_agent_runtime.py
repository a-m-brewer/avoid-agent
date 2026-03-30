"""Tests for the agent runtime execution controller."""

import json
from pathlib import Path

import avoid_agent.agent.tools.core  # noqa: F401  # registers built-in tools

from avoid_agent.agent.runtime import AgentRuntime, RuntimeEvent
from avoid_agent.agent.tools.finder import find_available_tools
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    Provider,
    ProviderResponse,
    ProviderStream,
    ProviderToolCall,
    ToolChoice,
    ToolResultMessage,
    Usage,
    UserMessage,
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
        self.last_tool_choice: ToolChoice | None = None
        self.received_messages: list[list[Message]] = []

    def stream(self, messages: list[Message], tools, tool_choice: ToolChoice = "auto"):
        self.last_tool_choice = tool_choice
        self.received_messages.append(list(messages))
        response = self._responses[self.calls]
        self.calls += 1
        return FakeStream(response)


def _response(message: AssistantMessage) -> ProviderResponse:
    return ProviderResponse(
        message=message,
        stop_reason=message.stop_reason,
        input_tokens=123,
    )


def _complete(evidence: list[str], summary: str = "Verified work is complete.") -> AssistantMessage:
    return AssistantMessage(
        text=json.dumps(
            {
                "plan": "Stop after verified execution.",
                "action": {
                    "tool": "complete",
                    "args": {
                        "summary": summary,
                        "evidence": evidence,
                    },
                },
            }
        ),
        stop_reason="stop",
    )


def _blocker(reason: str) -> AssistantMessage:
    return AssistantMessage(
        text=json.dumps(
            {
                "plan": "Cannot continue until the blocker is resolved.",
                "action": {
                    "tool": "blocker",
                    "args": {"reason": reason},
                },
            }
        ),
        stop_reason="stop",
    )


def test_runtime_executes_tool_calls_records_proof_and_accepts_structured_completion(
    tmp_path: Path,
):
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
            _response(_complete(["call_1"])),
        ]
    )
    events: list[RuntimeEvent] = []
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
        on_event=events.append,
    )

    result = runtime.run_user_turn([], "please write the file")

    assert provider.calls == 2
    assert target.read_text() == "hello"
    tool_results = [message for message in result.messages if isinstance(message, ToolResultMessage)]
    assert len(tool_results) == 1
    assert tool_results[0].details["proof"]["kind"] == "file_change"
    assert tool_results[0].details["verification"]["status"] == "verified"
    assert any(event.type == "tool_result" for event in events)
    assert result.messages[-1].text == _complete(["call_1"]).text


def test_runtime_verifies_ranged_read_file_proof(tmp_path: Path):
    target = tmp_path / "runtime-read.txt"
    target.write_text("one\ntwo\nthree\nfour\n", encoding="utf-8")
    provider = FakeProvider(
        responses=[
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_1",
                            name="read_file",
                            arguments={"path": str(target), "start_line": 2, "limit": 2},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            _response(_complete(["call_1"])),
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
    )

    result = runtime.run_user_turn([], "read the middle of the file")

    tool_results = [message for message in result.messages if isinstance(message, ToolResultMessage)]
    assert len(tool_results) == 1
    assert tool_results[0].content == "two\nthree\n"
    assert tool_results[0].details["proof"]["start_line"] == 2
    assert tool_results[0].details["proof"]["end_line"] == 3
    assert tool_results[0].details["verification"]["status"] == "verified"


def test_runtime_verifies_line_range_edit_proof(tmp_path: Path):
    target = tmp_path / "runtime-edit.txt"
    target.write_text("one\ntwo\nthree\n", encoding="utf-8")
    provider = FakeProvider(
        responses=[
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_1",
                            name="edit_file",
                            arguments={
                                "path": str(target),
                                "start_line": 2,
                                "end_line": 2,
                                "replacement": "updated\n",
                            },
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            _response(_complete(["call_1"])),
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
    )

    result = runtime.run_user_turn([], "update one line")

    assert target.read_text(encoding="utf-8") == "one\nupdated\nthree\n"
    tool_results = [message for message in result.messages if isinstance(message, ToolResultMessage)]
    assert len(tool_results) == 1
    assert tool_results[0].details["proof"]["edit_mode"] == "line_range"
    assert tool_results[0].details["verification"]["status"] == "verified"


def test_runtime_summarizes_older_large_tool_results_for_replay(tmp_path: Path):
    target = tmp_path / "runtime-read.txt"
    target.write_text("".join(f"line {i}\n" for i in range(1, 401)), encoding="utf-8")
    provider = FakeProvider(
        responses=[
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_1",
                            name="read_file",
                            arguments={"path": str(target)},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_2",
                            name="run_bash",
                            arguments={"command": "echo ok"},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            _response(_complete(["call_1", "call_2"])),
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes={"echo"},
    )

    runtime.run_user_turn([], "read a large file, then verify")

    third_request_messages = provider.received_messages[2]
    replayed_results = [
        message for message in third_request_messages if isinstance(message, ToolResultMessage)
    ]
    assert len(replayed_results) == 2
    assert "Full content suppressed for replay" in replayed_results[0].content
    assert "Artifact:" in replayed_results[0].content
    assert replayed_results[0].details["replay_backpressure"]["applied"] is True
    assert replayed_results[1].content.startswith("Exit code: 0")


def test_runtime_stores_large_tool_results_as_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("avoid_agent.agent.runtime._artifacts_dir", lambda: tmp_path / "artifacts")
    target = tmp_path / "runtime-read.txt"
    target.write_text("".join(f"line {i}\n" for i in range(1, 401)), encoding="utf-8")
    provider = FakeProvider(
        responses=[
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_1",
                            name="read_file",
                            arguments={"path": str(target)},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            _response(_complete(["call_1"])),
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
    )

    result = runtime.run_user_turn([], "read the file")

    tool_results = [message for message in result.messages if isinstance(message, ToolResultMessage)]
    assert len(tool_results) == 1
    artifact = tool_results[0].details["artifact"]
    artifact_path = Path(artifact["path"])
    assert artifact_path.exists()
    assert artifact_path.read_text(encoding="utf-8") == tool_results[0].content


def test_runtime_emits_provider_request_and_usage_status_events(tmp_path: Path):
    target = tmp_path / "runtime-read.txt"
    target.write_text("hello\n", encoding="utf-8")
    provider = FakeProvider(
        responses=[
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_1",
                            name="read_file",
                            arguments={"path": str(target)},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            ProviderResponse(
                message=AssistantMessage(
                    text=_complete(["call_1"]).text,
                    stop_reason="stop",
                    usage=Usage(
                        input_tokens=123,
                        output_tokens=45,
                        total_tokens=168,
                        cache_read_input_tokens=10,
                        cache_creation_input_tokens=20,
                    ),
                ),
                stop_reason="stop",
                input_tokens=123,
            ),
        ]
    )
    events: list[RuntimeEvent] = []
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
        on_event=events.append,
    )

    runtime.run_user_turn([], "read the file")

    status_events = [
        event.provider_event.status
        for event in events
        if event.type == "provider_event"
        and event.provider_event is not None
        and event.provider_event.type == "status"
    ]
    assert any(status.startswith("Provider request:") for status in status_events if status)
    assert any("cache-read 10" in status for status in status_events if status)
    assert any("cache-write 20" in status for status in status_events if status)


def test_runtime_emits_provider_specific_metrics_status(tmp_path: Path):
    class MetricsProvider(FakeProvider):
        def request_metrics(self, messages, tools, tool_choice="auto") -> dict:
            return {
                "provider": "anthropic",
                "wire_chars": 4321,
                "cache_breakpoints": 2,
                "messages": 3,
            }

    target = tmp_path / "runtime-read.txt"
    target.write_text("hello\n", encoding="utf-8")
    provider = MetricsProvider(
        responses=[
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_1",
                            name="read_file",
                            arguments={"path": str(target)},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            _response(_complete(["call_1"])),
        ]
    )
    events: list[RuntimeEvent] = []
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
        on_event=events.append,
    )

    runtime.run_user_turn([], "read the file")

    status_events = [
        event.provider_event.status
        for event in events
        if event.type == "provider_event"
        and event.provider_event is not None
        and event.provider_event.type == "status"
    ]
    assert any(
        status == "Provider metrics: provider=anthropic, wire_chars=4321, cache_breakpoints=2, provider_messages=3"
        for status in status_events
        if status
    )


def test_runtime_summarizes_medium_read_results_earlier_than_other_tools(tmp_path: Path):
    target = tmp_path / "runtime-read.txt"
    target.write_text("x" * 1300, encoding="utf-8")
    provider = FakeProvider(
        responses=[
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_1",
                            name="read_file",
                            arguments={"path": str(target)},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_2",
                            name="run_bash",
                            arguments={"command": "printf '%s' '" + ("y" * 1300) + "'"},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            _response(_complete(["call_1", "call_2"])),
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes={"printf"},
    )

    runtime.run_user_turn([], "read medium file then print medium output")

    third_request_messages = provider.received_messages[2]
    replayed_results = [
        message for message in third_request_messages if isinstance(message, ToolResultMessage)
    ]
    assert len(replayed_results) == 2
    assert "Full content suppressed for replay" in replayed_results[0].content
    assert replayed_results[1].content == "Exit code: 0\n\nSTDOUT:\n" + ("y" * 1300)


def test_runtime_retries_and_fails_on_freeform_completion_without_tool_use():
    provider = FakeProvider(
        responses=[
            _response(AssistantMessage(text="Implemented the fix.", stop_reason="stop")),
            _response(AssistantMessage(text="Done.", stop_reason="stop")),
            _response(AssistantMessage(text="Fixed it.", stop_reason="stop")),
        ]
    )
    events: list[RuntimeEvent] = []
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
        on_event=events.append,
    )

    result = runtime.run_user_turn([], "please update the implementation")

    assert provider.calls == 3
    assert result.messages[-1].stop_reason == "error"
    assert "success was claimed without a real tool execution" in result.messages[-1].error_message
    validation_events = [event for event in events if event.type == "validation_error"]
    assert len(validation_events) == 3


def test_runtime_rejects_completion_without_verified_evidence_then_accepts_retry(
    tmp_path: Path,
):
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
            _response(_complete(["missing"])),
            _response(_complete(["call_1"])),
        ]
    )
    events: list[RuntimeEvent] = []
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
        on_event=events.append,
    )

    result = runtime.run_user_turn([], "please write the file")

    assert provider.calls == 3
    assert result.messages[-1].text == _complete(["call_1"]).text
    validation_events = [event for event in events if event.type == "validation_error"]
    assert len(validation_events) == 1
    assert "Unknown or unverified: missing" in validation_events[0].message


def test_runtime_accepts_completion_wrapped_in_extra_text_after_real_tool_use(
    tmp_path: Path,
):
    target = tmp_path / "runtime-write.txt"
    wrapped_complete = AssistantMessage(
        text=(
            "completed\n"
            "```json\n"
            f'{_complete(["call_1"]).text}\n'
            "```\n"
            "Only the JSON object above matters."
        ),
        stop_reason="stop",
    )
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
            _response(wrapped_complete),
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
    )

    result = runtime.run_user_turn([], "please write the file")

    assert provider.calls == 2
    assert result.messages[-1].stop_reason == "stop"
    assert target.read_text() == "hello"


def test_runtime_recovers_truncated_structured_completion_after_real_tool_use(
    tmp_path: Path,
):
    target = tmp_path / "runtime-write.txt"
    truncated_complete = AssistantMessage(
        text=(
            '{"plan":"Stop after verified execution.","action":{"tool":"complete",'
            '"args":{"summary":"ok","evidence":["call_1"]}}'
        ),
        stop_reason="stop",
    )
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
            _response(truncated_complete),
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
    )

    result = runtime.run_user_turn([], "please write the file")

    assert provider.calls == 2
    assert result.messages[-1].stop_reason == "stop"
    assert target.read_text() == "hello"


def test_runtime_reports_irrecoverably_malformed_completion_json(tmp_path: Path):
    target = tmp_path / "runtime-write.txt"
    malformed_complete = AssistantMessage(
        text=(
            '{"plan":"Stop after verified execution.","action":{"tool":"complete"'
            '"args":{"summary":"ok","evidence":["call_1"]}}}'
        ),
        stop_reason="stop",
    )
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
            _response(malformed_complete),
            _response(_complete(["call_1"])),
        ]
    )
    events: list[RuntimeEvent] = []
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
        on_event=events.append,
    )

    result = runtime.run_user_turn([], "please write the file")

    assert provider.calls == 3
    assert result.messages[-1].stop_reason == "stop"
    validation_events = [event for event in events if event.type == "validation_error"]
    assert len(validation_events) == 1
    assert "valid JSON object" in validation_events[0].message


def test_runtime_executes_structured_json_tool_action_for_real_tool(tmp_path: Path):
    target = tmp_path / "runtime-write.txt"
    provider = FakeProvider(
        responses=[
            _response(
                AssistantMessage(
                    text=json.dumps(
                        {
                            "plan": "Write the requested file.",
                            "action": {
                                "tool": "write_file",
                                "args": {"path": str(target), "content": "hello"},
                            },
                        }
                    ),
                    stop_reason="stop",
                )
            ),
            _response(_complete(["json_action_1"])),
        ]
    )
    events: list[RuntimeEvent] = []
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
        on_event=events.append,
    )

    result = runtime.run_user_turn([], "please write the file")

    assert provider.calls == 2
    assert target.read_text() == "hello"
    tool_results = [message for message in result.messages if isinstance(message, ToolResultMessage)]
    assert len(tool_results) == 1
    assert tool_results[0].tool_call_id == "json_action_1"
    structured_events = [event for event in events if event.type == "structured_action"]
    assert [event.message for event in structured_events] == [
        "Write the requested file.",
        "Verified work is complete.",
    ]


def test_runtime_rejects_false_no_tool_access_blocker_and_retries_with_real_tool(
    tmp_path: Path,
):
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
            _response(
                _blocker(
                    "I cannot run the required verification tools because this turn did not "
                    "include tool access. Please allow me to run tests."
                )
            ),
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_2",
                            name="run_bash",
                            arguments={"command": "echo ok"},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            _response(_complete(["call_1", "call_2"])),
        ]
    )
    events: list[RuntimeEvent] = []
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes={"echo"},
        on_event=events.append,
    )

    result = runtime.run_user_turn([], "please write the file and verify it")

    assert provider.calls == 4
    assert result.messages[-1].text == _complete(["call_1", "call_2"]).text
    validation_events = [event for event in events if event.type == "validation_error"]
    assert len(validation_events) == 1
    assert "tool access is available in this turn" in validation_events[0].message


def test_runtime_allows_structured_blocker_without_tool_calls():
    provider = FakeProvider(
        responses=[_response(_blocker("Need the target file path before editing."))]
    )
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
    )

    result = runtime.run_user_turn([], "edit the file")

    assert provider.calls == 1
    assert result.messages[-1].text == _blocker("Need the target file path before editing.").text


def test_runtime_passes_tool_choice_to_provider():
    provider = FakeProvider(
        responses=[
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_1",
                            name="run_bash",
                            arguments={"command": "echo hi"},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            _response(_complete(["call_1"])),
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes={"echo"},
        tool_choice="required",
    )

    runtime.run_user_turn([], "do something")
    assert provider.last_tool_choice == "required"


def test_runtime_rejects_unknown_tool():
    provider = FakeProvider(
        responses=[
            _response(
                AssistantMessage(
                    tool_calls=[
                        ProviderToolCall(
                            id="call_1",
                            name="nonexistent_tool",
                            arguments={},
                        )
                    ],
                    stop_reason="tool_use",
                )
            ),
            _response(_blocker("Unknown tool call must be corrected.")),
        ]
    )
    events: list[RuntimeEvent] = []
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
        on_event=events.append,
    )

    runtime.run_user_turn([], "do something")

    tool_results = [event for event in events if event.type == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0].tool_result.is_error
    assert "Unknown tool" in tool_results[0].tool_result.content


def test_runtime_regrounds_follow_up_steps_with_execution_state(tmp_path: Path):
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
            _response(_complete(["call_1"])),
        ]
    )
    runtime = AgentRuntime(
        provider=provider,
        tool_definitions=find_available_tools(),
        allowed_prefixes=set(),
    )

    runtime.run_user_turn([], "please write the file")

    second_request_messages = provider.received_messages[1]
    state_messages = [
        message
        for message in second_request_messages
        if isinstance(message, UserMessage)
        and message.text.startswith("[EXECUTION STATE]")
    ]
    assert len(state_messages) == 1
    assert "call_1" in state_messages[0].text
    assert "Verified IDs: call_1" in state_messages[0].text
