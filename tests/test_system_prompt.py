"""Tests for system prompt assembly."""

from datetime import date

from avoid_agent.prompts.system_prompt import (
    ContextFile,
    SkillSummary,
    SystemPromptOptions,
    build_system_prompt,
    discover_skills,
)


def test_includes_runtime_context_when_provided():
    prompt = build_system_prompt(
        working_directory="/tmp/project",
        git_status="M avoid_agent/__main__.py",
        top_level_file_structure=".\n./avoid_agent\n./tests",
    )

    assert "## Runtime Context" in prompt
    assert "Working directory:\n/tmp/project" in prompt
    assert "Git status:\nM avoid_agent/__main__.py" in prompt
    assert "Top-level file structure:\n.\n./avoid_agent\n./tests" in prompt


def test_omits_runtime_context_when_not_provided():
    prompt = build_system_prompt()
    assert "## Runtime Context" not in prompt


def test_runtime_context_is_truncated_when_too_large():
    prompt = build_system_prompt(
        git_status="x" * 5000,
        top_level_file_structure="y" * 9000,
    )

    assert "... [truncated]" in prompt


def test_section_order_places_runtime_before_response_style():
    prompt = build_system_prompt(working_directory="/tmp/project")

    runtime_index = prompt.index("## Runtime Context")
    style_index = prompt.index("## Response Style")

    assert runtime_index < style_index


def test_default_prompt_has_split_policy_sections():
    prompt = build_system_prompt(options=SystemPromptOptions(include_date=False))

    assert "## Operational Policy" in prompt
    assert "## Planning & Reporting" in prompt
    assert "## Reliability Constraints" in prompt
    assert '"action":{"tool":"blocker"' in prompt
    assert '"action":{"tool":"complete"' in prompt
    assert "Use read_file ranges for large files" in prompt
    assert "Prefer edit_file's exact-string or line-range modes" in prompt


def test_identity_mentions_avoid_agent_and_its_codebase():
    prompt = build_system_prompt(options=SystemPromptOptions(include_date=False))

    assert "You are Avoid Agent (avoid_agent)" in prompt
    assert "operating on the avoid_agent codebase" in prompt
    assert "[INTERNAL SUMMARIZER]" in prompt


def test_custom_prompt_and_append_and_project_context():
    prompt = build_system_prompt(
        options=SystemPromptOptions(
            custom_prompt="CUSTOM BASE",
            append_system_prompt="APPENDED",
            context_files=[ContextFile(path="AGENTS.md", content="Follow team rules")],
            working_directory="/tmp/project",
            current_date=date(2026, 1, 2),
        )
    )

    assert "CUSTOM BASE" in prompt
    assert "APPENDED" in prompt
    assert "# Project Context" in prompt
    assert "## AGENTS.md" in prompt
    assert "Follow team rules" in prompt
    assert "Current date: 2026-01-02" in prompt
    assert "Current working directory: /tmp/project" in prompt


def test_selected_tools_and_snippets_limit_visible_tools():
    prompt = build_system_prompt(
        options=SystemPromptOptions(
            selected_tools=["read_file", "run_bash"],
            tool_snippets={"read_file": "Read file contents"},
            include_date=False,
        )
    )

    assert "- read_file: Read file contents" in prompt
    assert "- run_bash:" not in prompt


def test_prompt_guidelines_are_deduplicated():
    prompt = build_system_prompt(
        options=SystemPromptOptions(
            prompt_guidelines=[
                "Do not fabricate files, command outputs, or test results.",
                "Do not fabricate files, command outputs, or test results.",
            ],
            include_date=False,
        )
    )

    assert prompt.count("Do not fabricate files, command outputs, or test results.") == 1


def test_discover_skills_loads_name_and_description_and_deduplicates(tmp_path):
    repo_skills = tmp_path / "skills"
    user_skills = tmp_path / "user-skills"
    (repo_skills / "alpha").mkdir(parents=True)
    (repo_skills / "invalid").mkdir(parents=True)
    (user_skills / "beta").mkdir(parents=True)

    (repo_skills / "alpha" / "SKILL.md").write_text(
        """---
name: Repo Skill
description: First skill summary.
---
# Body should not be loaded into prompt summaries
""",
        encoding="utf-8",
    )
    (repo_skills / "invalid" / "SKILL.md").write_text(
        """---
name: Missing description
---
""",
        encoding="utf-8",
    )
    (user_skills / "beta" / "SKILL.md").write_text(
        """---
name: repo skill
description: Duplicate by name, should be ignored.
---
""",
        encoding="utf-8",
    )

    discovered = discover_skills(
        skills_search_paths=[str(repo_skills), str(user_skills)],
    )

    assert discovered == [
        SkillSummary(name="Repo Skill", description="First skill summary."),
    ]


def test_build_system_prompt_includes_discovered_skill_summaries(tmp_path):
    (tmp_path / "skills" / "self-review").mkdir(parents=True)
    (tmp_path / "skills" / "self-review" / "SKILL.md").write_text(
        """---
name: Self Review
description: Review recent changes before finalizing.
---
Use git diff and inspect changed files.
""",
        encoding="utf-8",
    )

    prompt = build_system_prompt(
        options=SystemPromptOptions(
            include_date=False,
            working_directory=str(tmp_path),
        )
    )

    assert "## Skills" in prompt
    assert "- Self Review: Review recent changes before finalizing." in prompt
    assert "Use git diff and inspect changed files." not in prompt
