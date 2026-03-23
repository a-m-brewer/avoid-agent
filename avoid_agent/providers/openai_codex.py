"""OpenAI Codex provider — uses ChatGPT Plus/Pro OAuth subscription."""

import json
import os
import platform
from typing import Callable, Iterator
from urllib.request import Request, urlopen
from urllib.error import HTTPError

_DEBUG_LOG = os.getenv("DEBUG_CODEX_EVENTS")  # set to a file path to log all SSE events


def _debug_log(event: dict) -> None:
    if not _DEBUG_LOG:
        return
    with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

from avoid_agent.agent.tools import ToolDefinition
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    Provider,
    ProviderEvent,
    ProviderResponse,
    ProviderStream,
    ProviderToolCall,
    ToolResultMessage,
    UserMessage,
)

CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
# Time to wait for data before treating the connection as dead.
# Long enough for deep reasoning pauses; short enough to recover after sleep/disconnect.
_STREAM_TIMEOUT = 120


def _parse_sse(response) -> Iterator[dict]:
    """Parse SSE events from a streaming HTTP response."""
    buffer = b""
    while True:
        chunk = response.read(4096)
        if not chunk:
            break
        buffer += chunk
        while b"\n\n" in buffer:
            event_bytes, buffer = buffer.split(b"\n\n", 1)
            data_parts = []
            for line in event_bytes.split(b"\n"):
                if line.startswith(b"data:"):
                    data_parts.append(line[5:].strip())
            if data_parts:
                data = b"".join(data_parts)
                if data and data != b"[DONE]":
                    try:
                        yield json.loads(data)
                    except json.JSONDecodeError:
                        pass


class CodexStream(ProviderStream):
    """Streaming context manager for the OpenAI Codex Responses API."""

    def __init__(self, make_request: Callable[[], Request], on_401: Callable[[], None] | None = None):
        self._make_request = make_request
        self._on_401 = on_401
        self._response = None
        self._text = ""
        self._text_id: str | None = None
        self._tool_calls: list[ProviderToolCall] = []
        self._reasoning_items: list[dict] = []
        self._stop = True
        self._input_tokens = 0
        # Per-item streaming state
        self._current_item_type: str | None = None
        self._pending_args = ""

    def _open(self) -> None:
        req = self._make_request()
        try:
            self._response = urlopen(req, timeout=_STREAM_TIMEOUT)
        except HTTPError as e:
            if e.code == 401 and self._on_401:
                self._on_401()
                req = self._make_request()
                self._response = urlopen(req, timeout=_STREAM_TIMEOUT)
            else:
                raise RuntimeError(f"HTTP {e.code}: {e.reason}") from e

    def __enter__(self):
        self._open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._response:
            self._response.close()

    def event_stream(self) -> Iterator[ProviderEvent]:
        for event in _parse_sse(self._response):
            _debug_log(event)
            event_type = event.get("type", "")

            if event_type == "response.output_item.added":
                item = event.get("item", {})
                self._current_item_type = item.get("type")
                self._pending_args = ""
                if self._current_item_type == "function_call":
                    tool_call = ProviderToolCall(
                        id=item.get("call_id", item.get("id", "")),
                        name=item.get("name", ""),
                        arguments={},
                        item_id=item.get("id"),
                    )
                    self._tool_calls.append(tool_call)
                    yield ProviderEvent(type="tool_call_detected", tool_call=tool_call)
                continue

            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    self._text += delta
                    yield ProviderEvent(type="text_delta", text=delta)
                continue

            if event_type == "response.function_call_arguments.delta":
                self._pending_args += event.get("delta", "")
                continue

            if event_type == "response.function_call_arguments.done":
                self._pending_args = event.get("arguments", self._pending_args)
                continue

            if event_type == "response.output_item.done":
                item = event.get("item", {})
                itype = item.get("type")
                if itype == "reasoning":
                    self._reasoning_items.append(item)
                    yield ProviderEvent(type="reasoning_item", reasoning_item=item)
                elif itype == "message":
                    self._text_id = item.get("id")
                elif itype == "function_call" and self._tool_calls:
                    try:
                        args = json.loads(self._pending_args or item.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    self._tool_calls[-1].arguments = args
                self._current_item_type = None
                continue

            if event_type in ("response.completed", "response.done"):
                response_obj = event.get("response", {})
                usage = response_obj.get("usage", {})
                self._input_tokens = usage.get("input_tokens", 0)
                if not self._tool_calls:
                    for item in response_obj.get("output", []):
                        if item.get("type") == "function_call":
                            try:
                                args = json.loads(item.get("arguments", "{}"))
                            except json.JSONDecodeError:
                                args = {}
                            tool_call = ProviderToolCall(
                                id=item.get("call_id", item.get("id", "")),
                                name=item.get("name", ""),
                                arguments=args,
                                item_id=item.get("id"),
                            )
                            self._tool_calls.append(tool_call)
                            yield ProviderEvent(type="tool_call_detected", tool_call=tool_call)
                self._stop = len(self._tool_calls) == 0
                yield ProviderEvent(type="status", status="completed")
                continue

            if event_type == "error":
                msg = event.get("message") or event.get("code") or json.dumps(event)
                yield ProviderEvent(type="error", error=msg)
                raise RuntimeError(f"Codex error: {msg}")

            yield ProviderEvent(type="raw", raw_event=event)

    def get_final_message(self) -> ProviderResponse:
        return ProviderResponse(
            message=AssistantMessage(
                text=self._text or None,
                tool_calls=self._tool_calls,
                text_id=self._text_id,
                reasoning_items=self._reasoning_items,
            ),
            stop=self._stop,
            input_tokens=self._input_tokens,
        )


class OpenAICodexProvider(Provider):
    """Provider for OpenAI Codex using ChatGPT Plus/Pro OAuth credentials."""

    def __init__(self, system: str, model: str, max_tokens: int, credentials: dict):
        super().__init__(system, model, max_tokens)
        self._credentials = credentials

    def _refresh_auth(self) -> None:
        from avoid_agent.providers.openai_codex_oauth import refresh_credentials  # pylint: disable=import-outside-toplevel
        self._credentials = refresh_credentials(self._credentials)

    def _build_headers(self) -> dict:
        ua = f"avoid-agent ({platform.system()} {platform.release()}; {platform.machine()})"
        return {
            "Authorization": f"Bearer {self._credentials['access']}",
            "chatgpt-account-id": self._credentials['account_id'],
            "originator": "avoid-agent",
            "User-Agent": ua,
            "OpenAI-Beta": "responses=experimental",
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        }

    def _build_body(self, messages: list[Message], tools: list[ToolDefinition]) -> dict:
        body: dict = {
            "model": self.model,
            "store": False,
            "stream": True,
            "instructions": self.system,
            "input": self._convert_messages(messages),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
        }
        if tools:
            body["tools"] = [self._convert_tool(t) for t in tools]
        return body

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        result = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                result.append({"role": "user", "content": msg.text})

            elif isinstance(msg, AssistantMessage):
                # Replay reasoning items first so the model has its prior thinking context
                for reasoning_item in msg.reasoning_items:
                    result.append(reasoning_item)

                # Text portion: include id and status for proper Responses API history format
                if msg.text:
                    text_item: dict = {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": msg.text}],
                    }
                    if msg.text_id:
                        text_item["id"] = msg.text_id
                    result.append(text_item)

                # Tool calls: use item_id for "id" and call_id (tc.id) for "call_id"
                for tc in msg.tool_calls:
                    fc_item: dict = {
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    }
                    if tc.item_id:
                        fc_item["id"] = tc.item_id
                    result.append(fc_item)

            elif isinstance(msg, ToolResultMessage):
                result.append({
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": msg.content,
                })

        return result

    def _convert_tool(self, tool: ToolDefinition) -> dict:
        properties = {}
        for param in tool.parameters:
            properties[param.name] = {
                "type": param.type.value,
                "description": param.description,
            }
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [p.name for p in tool.parameters if p.required],
            },
        }

    def stream(self, messages: list[Message], tools: list[ToolDefinition]) -> ProviderStream:
        body = self._build_body(messages, tools)
        body_bytes = json.dumps(body).encode()
        return CodexStream(
            make_request=lambda: Request(CODEX_URL, data=body_bytes, headers=self._build_headers(), method="POST"),
            on_401=self._refresh_auth,
        )

    def compact(self, messages: list[Message], keep_last: int = 6) -> list[Message]:
        """Summarise older messages via a streaming Codex request."""
        to_summarize = self._convert_messages(messages[:-keep_last])
        to_summarize.append({
            "role": "user",
            "content": (
                "Summarize this conversation so far. Include: what the user "
                "asked for, what was explored, what changes were made, and "
                "any important findings. Be concise but complete."
            ),
        })

        body = {
            "model": self.model,
            "store": False,
            "stream": True,
            "instructions": self.system,
            "input": to_summarize,
        }
        body_bytes = json.dumps(body).encode()

        req = Request(
            CODEX_URL,
            data=body_bytes,
            headers=self._build_headers(),
            method="POST",
        )
        try:
            with urlopen(req, timeout=_STREAM_TIMEOUT) as resp:
                summary_parts: list[str] = []
                for event in _parse_sse(resp):
                    etype = event.get("type", "")
                    if etype == "response.output_text.delta":
                        summary_parts.append(event.get("delta", ""))
                summary_text = "".join(summary_parts)
        except HTTPError as e:
            error_body = e.read().decode(errors="replace") if e.fp else ""
            raise RuntimeError(
                f"Codex compact failed ({e.code}): {error_body[:500]}"
            ) from e

        if not summary_text.strip():
            raise RuntimeError("Codex compact returned empty summary")

        return [
            UserMessage(text=f"[Conversation summary]\n{summary_text}"),
            AssistantMessage(text="Understood.", tool_calls=[]),
            *messages[-keep_last:],
        ]
