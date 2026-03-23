"""Tests for system prompt markdown export utility."""

from datetime import date

from avoid_agent.prompts import SystemPromptOptions, export_system_prompt_markdown


def test_export_system_prompt_markdown_writes_file(tmp_path):
    output_path = tmp_path / "prompt.md"

    written_path = export_system_prompt_markdown(
        output_path,
        options=SystemPromptOptions(
            include_date=True,
            current_date=date(2026, 1, 2),
            working_directory="/tmp/project",
        ),
    )

    assert written_path == output_path.resolve()

    content = output_path.read_text(encoding="utf-8")
    assert "# Avoid Agent System Prompt" in content
    assert "```markdown" in content
    assert "## Identity & Mission" in content
    assert "Current date: 2026-01-02" in content
    assert "Current working directory: /tmp/project" in content
