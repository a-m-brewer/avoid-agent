"""OpenAI provider implementation for Avoid Agent."""

import json
from typing import Iterator

from openai import OpenAI

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

        tool_calls = (
            []
            if choice0.finish_reason != "tool_calls"
            else [
                ProviderToolCall(
                    id=block.id,
                    name=block.function.name,
                    arguments=json.loads(block.function.arguments),
                )
                for block in choice0.message.tool_calls
            ]
        )

        return ProviderResponse(
            message=AssistantMessage(
                text=choice0.message.content, tool_calls=tool_calls
            ),
            stop=choice0.finish_reason == "stop",
            input_tokens=final.usage.prompt_tokens,
        )


class OpenAIProvider(Provider):
    """Provider for OpenAI models."""

    def __init__(self,
                 system: str,
                 model: str,
                 max_tokens: int,
                 api_key: str,
                 base_url: str | None):
        """Initialize the OpenAI provider with system instructions, model name, and max tokens."""
        super().__init__(system, model, max_tokens)
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def stream(
        self, messages: list[Message], tools: list[ToolDefinition]
    ) -> ProviderStream:
        provider_messages = self.__get_provider_messages(messages)
        provider_tools = self.__get_provider_tools(tools)

        openai_stream = self._client.chat.completions.stream(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=provider_messages,
            tools=provider_tools,
        )

        return OpenAIStream(openai_stream)

    def compact(self, messages: list[Message], keep_last: int = 6) -> list[Message]:
        """Compacts messages by keeping the last N and summarizing the rest."""

        to_summarize = self.__get_provider_messages(messages[:-keep_last])
        recent = messages[-keep_last:]

        summary_response = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[
                *to_summarize,
                {
                    "role": "user",
                    "content": "Summarize this conversation so far. Include: what the user asked for, what was explored, what changes were made, and any important findings. Be concise but complete.",
                },
            ],
        )

        summary_text = summary_response.choices[0].message.content.strip()

        return [
            UserMessage(text=f"[Conversation summary]\n{summary_text}"),
            AssistantMessage(text="Understood.", tool_calls=[]),
            *recent,
        ]

    def __get_provider_messages(self, messages: list[Message]) -> list[dict]:
        return [
            {"role": "system", "content": self.system},
            *[self.__convert_to_provider_message(msg) for msg in messages],
        ]

    def __convert_to_provider_message(self, msg: Message) -> dict:
        if isinstance(msg, UserMessage):
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
