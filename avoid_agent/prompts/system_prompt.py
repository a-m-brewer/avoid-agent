"""System prompt builder for Avoid Agent."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml

_MAX_GIT_STATUS_CHARS = 4000
_MAX_TREE_CHARS = 8000


@dataclass
class ContextFile:
    """Project context file to embed in the prompt."""

    path: str
    content: str


@dataclass(frozen=True)
class SkillSummary:
    """Lightweight, prompt-safe skill metadata."""

    name: str
    description: str


@dataclass
class SystemPromptOptions:
    """Configuration options for system prompt assembly."""

    custom_prompt: str | None = None
    append_system_prompt: str | None = None
    selected_tools: list[str] | None = None
    tool_snippets: dict[str, str] | None = None
    prompt_guidelines: list[str] | None = None
    context_files: list[ContextFile] | None = None
    working_directory: str | None = None
    git_status: str | None = None
    top_level_file_structure: str | None = None
    include_date: bool = True
    current_date: date | None = None
    skills_search_paths: list[str] | None = None
    discovered_skills: list[SkillSummary] | None = None


def _section(title: str, body: str) -> str:
    return f"## {title}\n{body.strip()}"


def _truncate(text: str, limit: int) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[:limit]}\n... [truncated]"


def _runtime_context_section(
    working_directory: str | None,
    git_status: str | None,
    top_level_file_structure: str | None,
) -> str | None:
    if not any([working_directory, git_status, top_level_file_structure]):
        return None

    parts: list[str] = []
    if working_directory:
        parts.append(f"Working directory:\n{working_directory.strip()}")
    if git_status:
        parts.append(f"Git status:\n{_truncate(git_status, _MAX_GIT_STATUS_CHARS)}")
    if top_level_file_structure:
        parts.append(
            "Top-level file structure:\n"
            f"{_truncate(top_level_file_structure, _MAX_TREE_CHARS)}"
        )

    return _section("Runtime Context", "\n\n".join(parts))


def _project_context_section(context_files: list[ContextFile] | None) -> str | None:
    if not context_files:
        return None

    chunks = ["Project-specific instructions and guidelines:\n"]
    for context_file in context_files:
        chunks.append(
            f"## {context_file.path}\n\n{context_file.content.strip()}\n"
        )

    return "# Project Context\n\n" + "\n".join(chunks).strip()


def _parse_skill_frontmatter(content: str) -> SkillSummary | None:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return None

    frontmatter_end = None
    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            frontmatter_end = idx
            break

    if frontmatter_end is None:
        return None

    frontmatter = "\n".join(lines[1:frontmatter_end]).strip()
    if not frontmatter:
        return None

    try:
        parsed = yaml.safe_load(frontmatter)
    except yaml.YAMLError:
        return None

    if not isinstance(parsed, dict):
        return None

    name = parsed.get("name")
    description = parsed.get("description")
    if not isinstance(name, str) or not isinstance(description, str):
        return None

    stripped_name = name.strip()
    stripped_description = description.strip()
    if not stripped_name or not stripped_description:
        return None

    return SkillSummary(name=stripped_name, description=stripped_description)


def _default_skill_search_paths(working_directory: str | None) -> list[Path]:
    base = Path(working_directory).expanduser() if working_directory else Path.cwd()
    return [base / "skills", Path.home() / ".avoid-agent" / "skills"]


def discover_skills(
    working_directory: str | None = None,
    *,
    skills_search_paths: list[str] | None = None,
) -> list[SkillSummary]:
    """Discover skills from local and user skill directories."""
    roots = (
        [Path(path).expanduser() for path in skills_search_paths]
        if skills_search_paths is not None
        else _default_skill_search_paths(working_directory)
    )

    discovered: list[SkillSummary] = []
    seen_names: set[str] = set()

    for root in roots:
        if not root.is_dir():
            continue

        for skill_file in sorted(root.rglob("SKILL.md")):
            try:
                content = skill_file.read_text(encoding="utf-8")
            except OSError:
                continue

            parsed = _parse_skill_frontmatter(content)
            if parsed is None:
                continue

            name_key = parsed.name.lower()
            if name_key in seen_names:
                continue

            seen_names.add(name_key)
            discovered.append(parsed)

    discovered.sort(key=lambda skill: skill.name.lower())
    return discovered


def _skills_section(skills: list[SkillSummary]) -> str | None:
    if not skills:
        return None

    lines = ["Discovered skills available in this environment:"]
    lines.extend(f"- {skill.name}: {skill.description}" for skill in skills)
    return _section("Skills", "\n".join(lines))


def _tools_section(selected_tools: list[str] | None, tool_snippets: dict[str, str] | None) -> str:
    tools = selected_tools or ["read_file", "write_file", "edit_file", "run_bash"]

    if tool_snippets:
        visible = [name for name in tools if tool_snippets.get(name)]
        if visible:
            listing = "\n".join(f"- {name}: {tool_snippets[name]}" for name in visible)
        else:
            listing = "(none)"
    else:
        descriptions = {
            "read_file": "inspect files",
            "write_file": "create or fully rewrite files",
            "edit_file": "make targeted edits to existing files",
            "run_bash": "run shell commands for search, tests, and git",
        }
        listing = "\n".join(
            f"- {name}: {descriptions.get(name, 'available tool')}" for name in tools
        )

    return _section("Available Tools", listing)


def _render_guidelines(lines: list[str]) -> str:
    return "\n".join(f"- {line}" for line in lines)


def _policy_sections(
    selected_tools: list[str] | None,
    prompt_guidelines: list[str] | None,
) -> list[str]:
    tools = set(selected_tools or ["read_file", "write_file", "edit_file", "run_bash"])

    operational: list[str] = []
    planning: list[str] = []
    reliability: list[str] = []

    def add_unique(target: list[str], guideline: str) -> None:
        normalized = guideline.strip()
        if normalized and normalized not in target:
            target.append(normalized)

    add_unique(
        operational,
        "Explore before acting: read relevant files before making changes.",
    )
    add_unique(
        operational,
        "Never guess file paths — discover them with run_bash (for example via ls/find).",
    )
    if "run_bash" in tools:
        add_unique(
            operational,
            "If a shell command could be destructive, explain what it will do before running it.",
        )
        add_unique(
            operational,
            "Never use run_bash to create or modify files (no cat/heredoc/echo/sed/python "
            "scripts that write files). Always use write_file or edit_file instead — they "
            "produce verifiable proofs and diffs. run_bash is for running tests, git "
            "operations, and other non-file-writing shell tasks.",
        )
        add_unique(
            operational,
            "Always use read_file to read file contents — never cat/sed/head/tail via "
            "run_bash. read_file produces SHA256 verification hashes that prove you "
            "actually read the file. run_bash reads bypass proof tracking.",
        )

    add_unique(
        planning,
        "Proceed with implementation directly; if you need more context before acting, gather it with read_file or run_bash first.",
    )
    add_unique(planning, "After making changes, report clearly what you changed and why.")
    add_unique(planning, "When you run commands, summarize key results and findings.")

    add_unique(
        reliability,
        "Never describe an action as complete unless a tool result confirms it.",
    )
    add_unique(
        reliability,
        "Describe plans in future tense; only report completion after a tool returns the result.",
    )
    add_unique(
        reliability,
        "To modify or create a file you MUST call write_file or edit_file. "
        "Writing code in a text response does NOT change any file. "
        "If you have not called a tool, nothing has been implemented.",
    )
    add_unique(
        reliability,
        'If you do not call a tool, respond with JSON only: '
        '{"plan":"...","action":{"tool":"blocker","args":{"reason":"..."}}} '
        'or {"plan":"...","action":{"tool":"complete","args":{"summary":"...","evidence":["tool_call_id"]}}}.',
    )
    add_unique(
        reliability,
        "Never say Done, Implemented, Fixed, or similar completion language unless the complete action cites real evidence.",
    )
    add_unique(reliability, "Do not fabricate files, command outputs, or test results.")
    add_unique(reliability, "Resolve uncertainty by inspecting code or running commands.")
    add_unique(
        reliability,
        "After writing code that calls external APIs or services, verify it works by "
        "running a quick smoke test via run_bash (e.g. a small Python snippet that "
        "imports and calls the new function). Do not assume code works just because "
        "unit tests pass — mocked tests do not exercise real integrations.",
    )
    add_unique(reliability, "Keep edits minimal and aligned with existing project style.")

    for extra in prompt_guidelines or []:
        normalized = extra.strip()
        if not normalized:
            continue
        if normalized in operational or normalized in planning or normalized in reliability:
            continue
        add_unique(reliability, normalized)

    return [
        _section("Operational Policy", _render_guidelines(operational)),
        _section("Planning & Reporting", _render_guidelines(planning)),
        _section("Reliability Constraints", _render_guidelines(reliability)),
    ]


def _default_base_prompt(options: SystemPromptOptions) -> str:
    sections = [
        _section(
            "Identity & Mission",
            """
            You are Avoid Agent (avoid_agent), a coding assistant that can inspect and modify a local codebase.
            You are currently operating on the avoid_agent codebase unless the user explicitly directs otherwise.
            Your mission is to help the user complete software tasks accurately, safely, and efficiently.
            """,
        ),
        _section(
            "Execution Environment",
            """
            You run inside a harness with tool-mediated access to the local filesystem and shell.
            You must rely on tool outputs for facts about files, commands, and repository state.
            During normal task execution, every assistant turn must either call a real tool
            or return JSON for a `blocker` or `complete` action.
            If the current user message starts with [INTERNAL SUMMARIZER], ignore that
            execution contract and return the requested summary directly.
            """,
        ),
        _tools_section(options.selected_tools, options.tool_snippets),
        *_policy_sections(options.selected_tools, options.prompt_guidelines),
    ]

    runtime_section = _runtime_context_section(
        working_directory=options.working_directory,
        git_status=options.git_status,
        top_level_file_structure=options.top_level_file_structure,
    )
    if runtime_section:
        sections.append(runtime_section)

    sections.append(
        _section(
            "Response Style",
            """
            Be direct and concise while remaining helpful.
            When you are not calling a tool, output JSON only with the required
            `plan` and `action` fields. Do not add Markdown fences, prose, or a haiku.
            """,
        )
    )

    return "\n\n".join(sections)


def build_system_prompt(
    working_directory: str | None = None,
    git_status: str | None = None,
    top_level_file_structure: str | None = None,
    *,
    options: SystemPromptOptions | None = None,
) -> str:
    """Builds the default system prompt used by providers."""
    opts = options or SystemPromptOptions(
        working_directory=working_directory,
        git_status=git_status,
        top_level_file_structure=top_level_file_structure,
    )

    base_prompt = opts.custom_prompt.strip() if opts.custom_prompt else _default_base_prompt(opts)

    parts = [base_prompt]

    if opts.append_system_prompt:
        parts.append(opts.append_system_prompt.strip())

    project_context = _project_context_section(opts.context_files)
    if project_context:
        parts.append(project_context)

    discovered_skills = opts.discovered_skills
    if discovered_skills is None:
        discovered_skills = discover_skills(
            working_directory=opts.working_directory,
            skills_search_paths=opts.skills_search_paths,
        )

    skills_section = _skills_section(discovered_skills)
    if skills_section:
        parts.append(skills_section)

    if opts.include_date:
        prompt_date = opts.current_date or datetime.now().date()
        parts.append(f"Current date: {prompt_date.isoformat()}")

    if opts.working_directory:
        parts.append(f"Current working directory: {opts.working_directory.strip()}")

    return "\n\n".join(part for part in parts if part)
