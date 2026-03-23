"""Session persistence — save and restore message history per working directory."""

import hashlib
import json
from pathlib import Path

from avoid_agent.providers import (
    AssistantMessage,
    AssistantTextBlock,
    AssistantThinkingBlock,
    Message,
    ProviderToolCall,
    ToolResultMessage,
    UserMessage,
    Usage,
)

_SESSIONS_DIR = Path.home() / ".avoid-agent" / "sessions"
_SESSION_VERSION = 2


def _cwd_key(cwd: str) -> str:
    return hashlib.sha256(cwd.encode()).hexdigest()[:16]


def _repo_dir(cwd: str) -> Path:
    return _SESSIONS_DIR / _cwd_key(cwd)


def session_path(cwd: str) -> Path:
    """Default session file path for backward compatibility."""
    return _repo_dir(cwd) / "default.json"


def session_name_path(cwd: str, name: str) -> Path:
    safe = _sanitize_session_name(name)
    return _repo_dir(cwd) / f"{safe}.json"


def list_sessions(cwd: str) -> list[str]:
    repo = _repo_dir(cwd)
    if not repo.exists():
        return []
    return sorted(p.stem for p in repo.glob("*.json") if p.is_file())


def save_session(cwd: str, messages: list[Message], name: str = "default") -> None:
    path = session_name_path(cwd, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": _SESSION_VERSION,
                "cwd": cwd,
                "name": name,
                "messages": [_serialize(m) for m in messages],
            },
            indent=2,
        )
    )


def load_session(cwd: str, name: str = "default") -> list[Message] | None:
    path = session_name_path(cwd, name)
    if not path.exists() and name == "default":
        legacy = _SESSIONS_DIR / f"{_cwd_key(cwd)}.json"
        if legacy.exists():
            path = legacy
        else:
            return None
    elif not path.exists():
        return None

    try:
        raw = json.loads(path.read_text())
        return [_deserialize(m) for m in raw["messages"]]
    except Exception:  # pylint: disable=broad-except
        return None


def delete_session(cwd: str, name: str = "default") -> None:
    path = session_name_path(cwd, name)
    if path.exists():
        path.unlink()


def _sanitize_session_name(name: str) -> str:
    cleaned = "".join(ch for ch in name.strip().lower() if ch.isalnum() or ch in ("-", "_"))
    return cleaned or "default"


def _serialize(msg: Message) -> dict:
    if isinstance(msg, UserMessage):
        return {"type": "user", "text": msg.text, "timestamp": msg.timestamp}
    if isinstance(msg, AssistantMessage):
        return {
            "type": "assistant",
            "text": msg.text,
            "text_id": msg.text_id,
            "reasoning_items": msg.reasoning_items,
            "content": [_serialize_content_block(block) for block in msg.content],
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments, "item_id": tc.item_id}
                for tc in msg.tool_calls
            ],
            "stop_reason": msg.stop_reason,
            "timestamp": msg.timestamp,
            "usage": {
                "input_tokens": msg.usage.input_tokens,
                "output_tokens": msg.usage.output_tokens,
                "total_tokens": msg.usage.total_tokens,
            },
            "error_message": msg.error_message,
        }
    if isinstance(msg, ToolResultMessage):
        return {
            "type": "tool_result",
            "tool_call_id": msg.tool_call_id,
            "content": msg.content,
            "tool_name": msg.tool_name,
            "is_error": msg.is_error,
            "timestamp": msg.timestamp,
            "details": msg.details,
        }
    raise ValueError(f"Unknown message type: {type(msg)}")


def _deserialize(data: dict) -> Message:
    t = data["type"]
    if t == "user":
        return UserMessage(
            text=data["text"],
            timestamp=data.get("timestamp", 0),
        )
    if t == "assistant":
        return AssistantMessage(
            text=data.get("text"),
            text_id=data.get("text_id"),
            reasoning_items=data.get("reasoning_items", []),
            content=[
                _deserialize_content_block(block)
                for block in data.get("content", [])
            ],
            tool_calls=[
                ProviderToolCall(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"],
                    item_id=tc.get("item_id"),
                )
                for tc in data.get("tool_calls", [])
            ],
            stop_reason=data.get("stop_reason", "stop"),
            timestamp=data.get("timestamp", 0),
            usage=Usage(
                input_tokens=data.get("usage", {}).get("input_tokens", 0),
                output_tokens=data.get("usage", {}).get("output_tokens", 0),
                total_tokens=data.get("usage", {}).get("total_tokens", 0),
            ),
            error_message=data.get("error_message"),
        )
    if t == "tool_result":
        return ToolResultMessage(
            tool_call_id=data["tool_call_id"],
            content=data["content"],
            tool_name=data.get("tool_name"),
            is_error=data.get("is_error", False),
            timestamp=data.get("timestamp", 0),
            details=data.get("details", {}),
        )
    raise ValueError(f"Unknown message type: {t}")


def _serialize_content_block(block: AssistantTextBlock | AssistantThinkingBlock | ProviderToolCall) -> dict:
    if isinstance(block, AssistantTextBlock):
        return {"type": "text", "text": block.text, "item_id": block.item_id}
    if isinstance(block, AssistantThinkingBlock):
        return {"type": "thinking", "text": block.text, "raw_item": block.raw_item}
    if isinstance(block, ProviderToolCall):
        return {
            "type": "tool_call",
            "id": block.id,
            "name": block.name,
            "arguments": block.arguments,
            "item_id": block.item_id,
        }
    raise ValueError(f"Unknown content block type: {type(block)}")


def _deserialize_content_block(data: dict) -> AssistantTextBlock | AssistantThinkingBlock | ProviderToolCall:
    block_type = data["type"]
    if block_type == "text":
        return AssistantTextBlock(text=data["text"], item_id=data.get("item_id"))
    if block_type == "thinking":
        return AssistantThinkingBlock(
            text=data.get("text", ""),
            raw_item=data.get("raw_item"),
        )
    if block_type == "tool_call":
        return ProviderToolCall(
            id=data["id"],
            name=data["name"],
            arguments=data.get("arguments", {}),
            item_id=data.get("item_id"),
        )
    raise ValueError(f"Unknown content block type: {block_type}")
