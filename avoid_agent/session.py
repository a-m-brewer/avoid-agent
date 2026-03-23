"""Session persistence — save and restore message history per working directory."""

import hashlib
import json
from pathlib import Path

from avoid_agent.providers import (
    AssistantMessage,
    Message,
    ProviderToolCall,
    ToolResultMessage,
    UserMessage,
)

_SESSIONS_DIR = Path.home() / ".avoid-agent" / "sessions"


def session_path(cwd: str) -> Path:
    key = hashlib.sha256(cwd.encode()).hexdigest()[:16]
    return _SESSIONS_DIR / f"{key}.json"


def save_session(cwd: str, messages: list[Message]) -> None:
    path = session_path(cwd)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"cwd": cwd, "messages": [_serialize(m) for m in messages]}, indent=2)
    )


def load_session(cwd: str) -> list[Message] | None:
    path = session_path(cwd)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
        return [_deserialize(m) for m in raw["messages"]]
    except Exception:  # pylint: disable=broad-except
        return None


def delete_session(cwd: str) -> None:
    path = session_path(cwd)
    if path.exists():
        path.unlink()


def _serialize(msg: Message) -> dict:
    if isinstance(msg, UserMessage):
        return {"type": "user", "text": msg.text}
    if isinstance(msg, AssistantMessage):
        return {
            "type": "assistant",
            "text": msg.text,
            "text_id": msg.text_id,
            "reasoning_items": msg.reasoning_items,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments, "item_id": tc.item_id}
                for tc in msg.tool_calls
            ],
        }
    if isinstance(msg, ToolResultMessage):
        return {"type": "tool_result", "tool_call_id": msg.tool_call_id, "content": msg.content}
    raise ValueError(f"Unknown message type: {type(msg)}")


def _deserialize(data: dict) -> Message:
    t = data["type"]
    if t == "user":
        return UserMessage(text=data["text"])
    if t == "assistant":
        return AssistantMessage(
            text=data.get("text"),
            text_id=data.get("text_id"),
            reasoning_items=data.get("reasoning_items", []),
            tool_calls=[
                ProviderToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                    item_id=tc.get("item_id"),
                )
                for tc in data.get("tool_calls", [])
            ],
        )
    if t == "tool_result":
        return ToolResultMessage(tool_call_id=data["tool_call_id"], content=data["content"])
    raise ValueError(f"Unknown message type: {t}")
