"""OpenAI Codex provider — uses ChatGPT Plus/Pro OAuth subscription."""

import json
import platform
from typing import Callable, Iterator
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from avoid_agent.agent.tools import ToolDefinition
from avoid_agent.providers import (
    AssistantMessage,
    Message,
    Provider,
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
        self._tool_calls: list[ProviderToolCall] = []
        self._stop = True
        self._input_tokens = 0

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

    def text_stream(self) -> Iterator[str]:
        for event in _parse_sse(self._response):
            event_type = event.get("type", "")

            if event_type == "response.output_text.delta":
                delta = event.get("delta", "")
                if delta:
                    self._text += delta
                    yield delta

            elif event_type in ("response.completed", "response.done"):
                response_obj = event.get("response", {})
                usage = response_obj.get("usage", {})
                self._input_tokens = usage.get("input_tokens", 0)
                for item in response_obj.get("output", []):
                    if item.get("type") == "function_call":
                        try:
                            args = json.loads(item.get("arguments", "{}"))
                        except json.JSONDecodeError:
                            args = {}
                        self._tool_calls.append(
                            ProviderToolCall(
                                id=item.get("id", ""),
                                name=item.get("name", ""),
                                arguments=args,
                            )
                        )
                self._stop = len(self._tool_calls) == 0

            elif event_type == "error":
                msg = event.get("message") or event.get("code") or json.dumps(event)
                raise RuntimeError(f"Codex error: {msg}")

    def get_final_message(self) -> ProviderResponse:
        return ProviderResponse(
            message=AssistantMessage(
                text=self._text or None,
                tool_calls=self._tool_calls,
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
                # Text portion (if any)
                if msg.text:
                    result.append({
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": msg.text}],
                    })
                # Tool calls as separate top-level items
                for tc in msg.tool_calls:
                    result.append({
                        "type": "function_call",
                        "id": tc.id,
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    })

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
        """Summarise older messages using the Codex API (non-streaming)."""
        to_summarize = self._convert_messages(messages[:-keep_last])
        to_summarize.append({
            "role": "user",
            "content": (
                "Summarize this conversation so far. Include: what the user asked for, "
                "what was explored, what changes were made, and any important findings. "
                "Be concise but complete."
            ),
        })

        body = {
            "model": self.model,
            "store": False,
            "stream": False,
            "instructions": self.system,
            "input": to_summarize,
        }
        headers = self._build_headers()
        headers["Accept"] = "application/json"
        del headers["OpenAI-Beta"]

        req = Request(
            CODEX_URL,
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(req) as resp:
                result = json.loads(resp.read())
        except HTTPError as e:
            raise RuntimeError(f"Codex compact request failed: {e.code}") from e

        # Extract text from output
        summary_text = ""
        for item in result.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        summary_text += part.get("text", "")

        return [
            UserMessage(text=f"[Conversation summary]\n{summary_text}"),
            AssistantMessage(text="Understood.", tool_calls=[]),
            *messages[-keep_last:],
        ]
