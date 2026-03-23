"""Development utilities for exporting the assembled system prompt to Markdown."""

from __future__ import annotations

from pathlib import Path

from .system_prompt import SystemPromptOptions, build_system_prompt


def export_system_prompt_markdown(
    output_path: str | Path,
    *,
    options: SystemPromptOptions | None = None,
) -> Path:
    """Render the system prompt and write it to a Markdown file.

    Args:
        output_path: Destination file path.
        options: Optional system prompt options used during rendering.

    Returns:
        The resolved output path that was written.
    """
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    prompt = build_system_prompt(options=options)

    markdown = (
        "# Avoid Agent System Prompt\n\n"
        "_Generated for development review._\n\n"
        "```markdown\n"
        f"{prompt}\n"
        "```\n"
    )

    destination.write_text(markdown, encoding="utf-8")
    return destination
