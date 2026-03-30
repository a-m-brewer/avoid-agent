"""Tests for the OpenAI Codex provider parser."""

import json
from urllib.request import Request

from avoid_agent.providers import AssistantMessage, ProviderToolCall, ToolResultMessage, UserMessage
from avoid_agent.providers.openai_codex import CodexStream
from avoid_agent.providers.openai_codex import OpenAICodexProvider


class FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0

    def read(self, size: int) -> bytes:
        if self._offset >= len(self._payload):
            return b""
        chunk = self._payload[self._offset:self._offset + size]
        self._offset += size
        return chunk

    def close(self) -> None:
        return None


class OpenAfterTerminalResponse:
    """Simulates an SSE connection that stays open after response.completed."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0
        self._read_after_payload = False

    def read(self, size: int) -> bytes:
        if self._offset < len(self._payload):
            chunk = self._payload[self._offset:self._offset + size]
            self._offset += size
            return chunk
        if self._read_after_payload:
            raise RuntimeError("stream kept reading after terminal event")
        self._read_after_payload = True
        raise RuntimeError("stream kept reading after terminal event")

    def close(self) -> None:
        return None


def test_codex_stream_parses_tool_call_and_sets_tool_use_stop_reason(monkeypatch):
    sse = "\n\n".join(
        [
            'data: {"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"edit_file","arguments":""}}',
            'data: {"type":"response.function_call_arguments.delta","delta":"{\\"path\\":\\"file.py\\",\\"old_string\\":\\"a\\",\\"new_string\\":\\"b\\"}"}',
            'data: {"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"edit_file","arguments":"{\\"path\\":\\"file.py\\",\\"old_string\\":\\"a\\",\\"new_string\\":\\"b\\"}"}}',
            'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}',
            "",
        ]
    ).encode()

    monkeypatch.setattr(
        "avoid_agent.providers.openai_codex.urlopen",
        lambda request, timeout: FakeHTTPResponse(sse),
    )

    with CodexStream(lambda: Request("https://example.com")) as stream:
        events = list(stream.event_stream())
        final = stream.get_final_message()

    tool_events = [event for event in events if event.type == "tool_call_detected"]
    assert len(tool_events) == 1
    assert final.stop_reason == "tool_use"
    assert final.message.tool_calls[0].arguments["path"] == "file.py"
    assert final.message.content[0].type == "tool_call"


def test_codex_stream_finishes_after_response_completed_even_if_connection_stays_open(monkeypatch):
    sse = "\n\n".join(
        [
            'data: {"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"write_file","arguments":""}}',
            'data: {"type":"response.function_call_arguments.done","arguments":"{\\"path\\":\\"file.py\\",\\"content\\":\\"hello\\"}"}',
            'data: {"type":"response.output_item.done","item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"write_file","arguments":"{\\"path\\":\\"file.py\\",\\"content\\":\\"hello\\"}"}}',
            'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}',
            "",
        ]
    ).encode()

    monkeypatch.setattr(
        "avoid_agent.providers.openai_codex.urlopen",
        lambda request, timeout: OpenAfterTerminalResponse(sse),
    )

    with CodexStream(lambda: Request("https://example.com")) as stream:
        events = list(stream.event_stream())
        final = stream.get_final_message()

    assert any(event.type == "tool_call_detected" for event in events)
    assert any(event.type == "status" for event in events)
    assert final.stop_reason == "tool_use"
    assert final.message.tool_calls[0].name == "write_file"


def test_codex_stream_recovers_missing_text_suffix_from_final_message_item(monkeypatch):
    json_text = (
        '{"plan":"Stop after verified execution.","action":{"tool":"complete",'
        '"args":{"summary":"ok","evidence":["call_1"]}}}'
    )
    truncated_delta = json_text[:-1]
    sse = "\n\n".join(
        [
            'data: {"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}',
            f'data: {{"type":"response.output_text.delta","delta":{json.dumps(truncated_delta)}}}',
            f'data: {{"type":"response.output_item.done","item":{{"type":"message","id":"msg_1","content":[{{"type":"output_text","text":{json.dumps(json_text)}}}]}}}}',
            f'data: {{"type":"response.completed","response":{{"status":"completed","usage":{{"input_tokens":5,"output_tokens":3,"total_tokens":8}},"output":[{{"type":"message","id":"msg_1","content":[{{"type":"output_text","text":{json.dumps(json_text)}}}]}}]}}}}',
            "",
        ]
    ).encode()

    monkeypatch.setattr(
        "avoid_agent.providers.openai_codex.urlopen",
        lambda request, timeout: FakeHTTPResponse(sse),
    )

    with CodexStream(lambda: Request("https://example.com")) as stream:
        events = list(stream.event_stream())
        final = stream.get_final_message()

    streamed_text = "".join(event.text or "" for event in events if event.type == "text_delta")
    assert streamed_text == json_text
    assert final.message.text == json_text


def test_codex_stream_captures_response_id_from_completed_event(monkeypatch):
    sse = "\n\n".join(
        [
            'data: {"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}',
            'data: {"type":"response.output_text.delta","delta":"hi"}',
            'data: {"type":"response.completed","response":{"id":"resp_123","status":"completed","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8},"output":[{"type":"message","id":"msg_1","content":[{"type":"output_text","text":"hi"}]}]}}',
            "",
        ]
    ).encode()

    monkeypatch.setattr(
        "avoid_agent.providers.openai_codex.urlopen",
        lambda request, timeout: FakeHTTPResponse(sse),
    )

    with CodexStream(lambda: Request("https://example.com")) as stream:
        list(stream.event_stream())
        final = stream.get_final_message()

    assert final.message.provider_state["response_id"] == "resp_123"


def test_codex_provider_builds_delta_request_from_previous_response_id():
    provider = OpenAICodexProvider(
        system="sys",
        model="codex-mini-latest",
        max_tokens=1024,
        credentials={"access": "token", "account_id": "acct"},
    )
    messages = [
        UserMessage(text="first"),
        AssistantMessage(
            text="working",
            tool_calls=[ProviderToolCall(id="call_1", name="read_file", arguments={"path": "x"})],
            stop_reason="tool_use",
            provider_state={"response_id": "resp_prev"},
        ),
        ToolResultMessage(tool_call_id="call_1", tool_name="read_file", content="hello"),
        UserMessage(text="continue"),
    ]

    body = provider._build_body(messages, tools=[], tool_choice="auto")

    assert body["store"] is True
    assert body["previous_response_id"] == "resp_prev"
    assert body["input"] == [
        {"type": "function_call_output", "call_id": "call_1", "output": "hello"},
        {"role": "user", "content": "continue"},
    ]


def test_codex_provider_falls_back_to_full_input_without_response_id():
    provider = OpenAICodexProvider(
        system="sys",
        model="codex-mini-latest",
        max_tokens=1024,
        credentials={"access": "token", "account_id": "acct"},
    )
    messages = [UserMessage(text="first")]

    body = provider._build_body(messages, tools=[], tool_choice="auto")

    assert "previous_response_id" not in body
    assert body["input"] == [{"role": "user", "content": "first"}]
