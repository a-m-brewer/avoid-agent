"""Session learnings capture for headless runs."""

from datetime import datetime
from pathlib import Path


def capture_session(session_id: str, tool_calls: list[dict], errors: list[str]) -> Path | None:
    """Write a learnings entry. Returns the path written, or None if nothing to log."""
    clean_errors = [error.strip() for error in errors if isinstance(error, str) and error.strip()]
    failed_tool_calls = [call for call in tool_calls if call.get("is_error") is True]

    if not clean_errors and not failed_tool_calls:
        return None

    now = datetime.now()
    timestamp = now.isoformat()
    filename_ts = now.strftime("%Y%m%d-%H%M%S")

    repo_root = Path(__file__).resolve().parent.parent
    learnings_dir = repo_root / ".learnings" / "sessions"
    learnings_dir.mkdir(parents=True, exist_ok=True)

    path = learnings_dir / f"{filename_ts}-{session_id[:8]}.md"
    error_count = len(clean_errors) + len(failed_tool_calls)

    lines = [
        "---",
        f"session_id: {session_id}",
        f"timestamp: {timestamp}",
        f"error_count: {error_count}",
        "---",
        "",
        "## Errors",
    ]

    if clean_errors:
        lines.extend(f"- {error}" for error in clean_errors)
    else:
        lines.append("- None")

    lines.extend(["", "## Failed Tool Calls"])

    if failed_tool_calls:
        for call in failed_tool_calls:
            name = call.get("name") or call.get("tool_name") or "unknown"
            call_id = call.get("id") or call.get("tool_call_id")
            arguments = call.get("arguments")

            parts = [name]
            if call_id:
                parts.append(f"id={call_id}")
            if arguments not in (None, "", {}, []):
                parts.append(f"arguments={arguments}")

            lines.append(f"- {', '.join(parts)}")
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
