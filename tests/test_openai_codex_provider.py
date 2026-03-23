"""Tests for the OpenAI Codex provider parser."""

from urllib.request import Request

from avoid_agent.providers.openai_codex import CodexStream


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
