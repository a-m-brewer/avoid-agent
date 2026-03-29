"""OpenAI provider implementation for Avoid Agent."""

import json
from typing import Iterator

from openai import OpenAI


def list_models(api_key: str, base_url: str | None = None) -> list[str]:
    """List available model IDs via OpenAI-compatible models API."""
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.models.list()
    ids = [getattr(model, "id", "") for model in response.data]
    return sorted(model_id for model_id in ids if model_id)

from avoid_agent.agent.tools import ToolDefinition
from avoid_agent.providers import (
    AssistantMessage,
    AssistantTextBlock,
    Message,
    Provider,
    ProviderEvent,
    ProviderResponse,
    ProviderStream,
    ProviderToolCall,
    ToolChoice,
    ToolResultMessage,
    UserMessage,
    Usage,
    normalize_messages,
)


def _map_stop_reason(finish_reason: str | None) -> str:
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "length"
    if finish_reason == "stop":
        return "stop"
    if finish_reason is None:
        return "stop"
    return "error"


class OpenAIStream(ProviderStream):
    """Context manager for streaming responses from the OpenAI provider."""

    def __init__(self, ctx):
        super().__init__()
        self._ctx = ctx
        self._stream = None

    def __enter__(self):
        self._stream = self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)

    def event_stream(self) -> Iterator[ProviderEvent]:
        for event in self._stream:
            if event.type == 'content.delta' and event.delta:
                yield ProviderEvent(type="text_delta", text=event.delta)
            else:
                yield ProviderEvent(type="raw", raw_event={"type": event.type})

    def get_final_message(self) -> ProviderResponse:
        final = self._stream.get_final_completion()
        choice0 = final.choices[0]
        finish_reason = choice0.finish_reason

        tool_calls = (
            []
            if finish_reason != "tool_calls"
            else [
                ProviderToolCall(
                    id=block.id,
                    name=block.function.name,
                    arguments=json.loads(block.function.arguments),
                )
                for block in choice0.message.tool_calls
            ]
        )

        stop_reason = _map_stop_reason(finish_reason)
        if tool_calls and stop_reason == "stop":
            stop_reason = "tool_use"

        content = []
        if choice0.message.content:
            content.append(AssistantTextBlock(text=choice0.message.content))
        content.extend(tool_calls)

        return ProviderResponse(
            message=AssistantMessage(
                text=choice0.message.content,
                tool_calls=tool_calls,
                content=content,
                stop_reason=stop_reason,
                usage=Usage(
                    input_tokens=final.usage.prompt_tokens if final.usage else 0,
                    output_tokens=final.usage.completion_tokens if final.usage else 0,
                    total_tokens=final.usage.total_tokens if final.usage else 0,
                ),
            ),
            stop_reason=stop_reason,
            input_tokens=final.usage.prompt_tokens if final.usage else 0,
        )


class OpenAIProvider(Provider):
    """Provider for OpenAI models."""

    def __init__(
        self,
        system: str,
        model: str,
        max_tokens: int,
        api_key: str,
        base_url: str | None,
        *,
        thinking_enabled: bool | None = None,
        effort: str | None = None,
    ):
        """Initialize the OpenAI provider with system instructions, model name, and max tokens."""
        super().__init__(
            system,
            model,
            max_tokens,
            thinking_enabled=thinking_enabled,
            effort=effort,
        )
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        tool_choice: ToolChoice = "auto",
    ) -> ProviderStream:
        provider_messages = self.__get_provider_messages(normalize_messages(messages))
        provider_tools = self.__get_provider_tools(tools)

        kwargs: dict = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": provider_messages,
            "tools": provider_tools,
        }
        if tool_choice != "auto" and provider_tools:
            kwargs["tool_choice"] = tool_choice
        # Optional reasoning effort control (supported by some models/APIs)
        if self.thinking_enabled and getattr(self, "effort", None):
            kwargs["reasoning"] = {"effort": self.effort}

        openai_stream = self._client.chat.completions.stream(**kwargs)
        return OpenAIStream(openai_stream)

    def __get_provider_messages(self, messages: list[Message]) -> list[dict]:
        return [
            {"role": "system", "content": self.system},
            *[self.__convert_to_provider_message(msg) for msg in messages],
        ]

    def __convert_to_provider_message(self, msg: Message) -> dict:
        if isinstance(msg, UserMessage):
            # If there are attached images, build a multipart content array.
            if msg.images:
                content: list[dict] = []
                for img in msg.images:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{img.media_type};base64,{img.data}",
                        },
                    })
                if msg.text:
                    content.append({"type": "text", "text": msg.text})
                return {"role": "user", "content": content}
            return {"role": "user", "content": msg.text}

        if isinstance(msg, AssistantMessage):
            am = {"role": "assistant", "content": msg.text or ""}

            if not msg.tool_calls:
                return am

            am["tool_calls"] = []
            for tool_call in msg.tool_calls:
                am["tool_calls"].append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                )
            return am

        if isinstance(msg, ToolResultMessage):
            return {
                "role": "tool",
                "tool_call_id": msg.tool_call_id,
                "content": msg.content,
            }

        raise ValueError(f"Unknown message type: {type(msg)}")

    def __get_provider_tools(self, tools: list[ToolDefinition]) -> list[dict]:
        return [self.__convert_to_provider_tool(tool) for tool in tools]

    def __convert_to_provider_tool(self, tool: ToolDefinition) -> dict:
        properties = {}
        for param in tool.parameters:
            properties[param.name] = {
                "type": param.type.value,
                "description": param.description,
            }

        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": [
                        required_param.name
                        for required_param in tool.parameters
                        if required_param.required
                    ],
                },
            },
        }
