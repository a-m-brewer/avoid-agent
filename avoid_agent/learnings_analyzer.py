"""Analyze session learnings for recurring error patterns."""

from pathlib import Path
import re


_KEYWORD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("OSError", re.compile(r"oserror", re.IGNORECASE)),
    ("PermissionError", re.compile(r"permissionerror", re.IGNORECASE)),
    ("JSON", re.compile(r"json", re.IGNORECASE)),
    ("timeout", re.compile(r"timeout", re.IGNORECASE)),
    ("tool not found", re.compile(r"tool not found", re.IGNORECASE)),
    ("import", re.compile(r"\bimport\b", re.IGNORECASE)),
)

_SUGGESTION_TEMPLATES = {
    "OSError": "Recurring OSError failures (seen in {count} sessions) — consider adding stronger filesystem and OS-level error handling.",
    "PermissionError": "Recurring permission errors (seen in {count} sessions) — consider validating file and directory permissions before operations.",
    "JSON": "Recurring JSON parse errors (seen in {count} sessions) — consider improving error handling in JSON parsing paths.",
    "timeout": "Recurring timeout issues (seen in {count} sessions) — consider adding retries and tuning timeout thresholds.",
    "tool not found": "Recurring tool-not-found failures (seen in {count} sessions) — consider validating tool availability before execution.",
    "import": "Recurring import errors (seen in {count} sessions) — consider checking dependency installation and module paths.",
}


def analyze(learnings_dir: Path) -> list[str]:
    """Return recurring error suggestions derived from markdown session learnings."""
    if not learnings_dir.exists() or not learnings_dir.is_dir():
        return []

    session_files = sorted(path for path in learnings_dir.glob("*.md") if path.is_file())
    if not session_files:
        return []

    counts = {keyword: 0 for keyword, _ in _KEYWORD_PATTERNS}

    for session_file in session_files:
        errors_text = _extract_errors_section(session_file.read_text(encoding="utf-8"))
        for keyword, pattern in _KEYWORD_PATTERNS:
            if pattern.search(errors_text):
                counts[keyword] += 1

    suggestions: list[str] = []
    for keyword, _ in _KEYWORD_PATTERNS:
        count = counts[keyword]
        if count >= 3:
            suggestions.append(_SUGGESTION_TEMPLATES[keyword].format(count=count))

    return suggestions


def _extract_errors_section(content: str) -> str:
    in_errors_section = False
    errors_lines: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()

        if not in_errors_section:
            if stripped.lower() == "## errors":
                in_errors_section = True
            continue

        if stripped.startswith("## "):
            break

        errors_lines.append(line)

    return "\n".join(errors_lines)
