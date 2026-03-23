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
    AssistantTextBlock,
    AssistantThinkingBlock,
    Message,
    Provider,
    ProviderEvent,
    ProviderResponse,
    ProviderStream,
    ProviderToolCall,
    StopReason,
    ToolResultMessage,
    UserMessage,
    Usage,
    normalize_messages,
)

CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
# Time to wait for data before treating the connection as dead.
# Long enough for deep reasoning pauses; short enough to recover after sleep/disconnect.
_STREAM_TIMEOUT = 120


def _extract_reasoning_text(item: dict) -> str:
    summary = item.get("summary")
    if isinstance(summary, list):
        return "\n\n".join(
            str(part.get("text", "")).strip()
            for part in summary
            if isinstance(part, dict) and str(part.get("text", "")).strip()
        )
    if isinstance(summary, str):
        return summary
    return ""


def _map_stop_reason(status: str | None) -> StopReason:
    if status == "incomplete":
        return "length"
    if status in ("failed", "cancelled"):
        return "error"
    if status in ("completed", "queued", "in_progress", None):
        return "stop"
    return "error"


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
        self._content: list[AssistantTextBlock | AssistantThinkingBlock | ProviderToolCall] = []
        self._tool_calls: list[ProviderToolCall] = []
        self._reasoning_items: list[dict] = []
        self._usage = Usage()
        self._stop_reason: StopReason = "stop"
        # Per-item streaming state
        self._current_item_type: str | None = None
        self._pending_args = ""
        self._current_text_block: AssistantTextBlock | None = None
        self._current_tool_call: ProviderToolCall | None = None
        self._current_thinking_block: AssistantThinkingBlock | None = None

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
                if self._current_item_type == "message":
                    self._current_text_block = AssistantTextBlock(
                        text="",
                        item_id=item.get("id"),
                    )
                    self._content.append(self._current_text_block)
                elif self._current_item_type == "reasoning":
                    self._current_thinking_block = AssistantThinkingBlock(text="")
                    self._content.append(self._current_thinking_block)
                elif self._current_item_type == "function_call":
                    self._current_tool_call = ProviderToolCall(
                        id=item.get("call_id", item.get("id", "")),
                        name=item.get("name", ""),
                        arguments={},
                        item_id=item.get("id"),
                    )
                    self._tool_calls.append(self._current_tool_call)
                    self._content.append(self._current_tool_call)
                    yield ProviderEvent(
                        type="tool_call_detected",
                        tool_call=self._current_tool_call,
                    )
                continue

            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    if self._current_text_block is not None:
                        self._current_text_block.text += delta
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
                if itype == "reasoning" and self._current_thinking_block is not None:
                    self._current_thinking_block.text = _extract_reasoning_text(item)
                    self._current_thinking_block.raw_item = item
                    self._reasoning_items.append(item)
                    yield ProviderEvent(type="reasoning_item", reasoning_item=item)
                    self._current_thinking_block = None
                elif itype == "message" and self._current_text_block is not None:
                    self._current_text_block.item_id = item.get("id")
                    if not self._current_text_block.text:
                        content = item.get("content", [])
                        self._current_text_block.text = "".join(
                            str(part.get("text", ""))
                            for part in content
                            if isinstance(part, dict) and part.get("type") == "output_text"
                        )
                    self._current_text_block = None
                elif itype == "function_call" and self._current_tool_call is not None:
                    try:
                        args = json.loads(self._pending_args or item.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {}
                    self._current_tool_call.arguments = args
                    self._current_tool_call = None
                self._current_item_type = None
                continue

            if event_type in ("response.completed", "response.done", "response.incomplete"):
                response_obj = event.get("response", {})
                usage = response_obj.get("usage", {})
                self._usage = Usage(
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                )
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
                            self._content.append(tool_call)
                            yield ProviderEvent(type="tool_call_detected", tool_call=tool_call)
                self._stop_reason = _map_stop_reason(response_obj.get("status"))
                if self._tool_calls and self._stop_reason == "stop":
                    self._stop_reason = "tool_use"
                yield ProviderEvent(
                    type="status",
                    status=response_obj.get("status", "completed"),
                )
                continue

            if event_type == "error":
                msg = event.get("message") or event.get("code") or json.dumps(event)
                yield ProviderEvent(type="error", error=msg)
                raise RuntimeError(f"Codex error: {msg}")

            yield ProviderEvent(type="raw", raw_event=event)

    def get_final_message(self) -> ProviderResponse:
        text_blocks = [
            block for block in self._content if isinstance(block, AssistantTextBlock)
        ]
        return ProviderResponse(
            message=AssistantMessage(
                text="".join(block.text for block in text_blocks) or None,
                tool_calls=self._tool_calls,
                text_id=text_blocks[0].item_id if len(text_blocks) == 1 else None,
                reasoning_items=self._reasoning_items,
                content=self._content,
                stop_reason=self._stop_reason,
                usage=self._usage,
            ),
            stop_reason=self._stop_reason,
            input_tokens=self._usage.input_tokens,
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
            "input": self._convert_messages(normalize_messages(messages)),
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
                if msg.content:
                    for block in msg.content:
                        if isinstance(block, AssistantThinkingBlock):
                            if block.raw_item:
                                result.append(block.raw_item)
                        elif isinstance(block, AssistantTextBlock):
                            text_item: dict = {
                                "type": "message",
                                "role": "assistant",
                                "status": "completed",
                                "content": [{"type": "output_text", "text": block.text}],
                            }
                            if block.item_id:
                                text_item["id"] = block.item_id
                            result.append(text_item)
                        elif isinstance(block, ProviderToolCall):
                            fc_item: dict = {
                                "type": "function_call",
                                "call_id": block.id,
                                "name": block.name,
                                "arguments": json.dumps(block.arguments),
                            }
                            if block.item_id:
                                fc_item["id"] = block.item_id
                            result.append(fc_item)
                else:
                    for reasoning_item in msg.reasoning_items:
                        result.append(reasoning_item)

                    if msg.text:
                        text_item = {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": msg.text}],
                        }
                        if msg.text_id:
                            text_item["id"] = msg.text_id
                        result.append(text_item)

                    for tc in msg.tool_calls:
                        fc_item = {
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
        to_summarize = self._convert_messages(normalize_messages(messages[:-keep_last]))
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
