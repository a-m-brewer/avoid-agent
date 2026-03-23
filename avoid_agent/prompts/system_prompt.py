"""System prompt builder for Avoid Agent."""

_MAX_GIT_STATUS_CHARS = 4000
_MAX_TREE_CHARS = 8000


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


def build_system_prompt(
    working_directory: str | None = None,
    git_status: str | None = None,
    top_level_file_structure: str | None = None,
) -> str:
    """Builds the default system prompt used by providers."""
    sections = [
        _section(
            "Identity & Mission",
            """
            You are Avoid Agent, a coding assistant that can inspect and modify a local codebase.
            Your mission is to help the user complete software tasks accurately, safely, and efficiently.
            """,
        ),
        _section(
            "Execution Environment",
            """
            You run inside a harness with tool-mediated access to the local filesystem and shell.
            You must rely on tool outputs for facts about files, commands, and repository state.
            """,
        ),
        _section(
            "Available Tools",
            """
            You have 4 tools: read_file, write_file, edit_file, and run_bash.

            Use read_file to inspect files.
            Use edit_file for targeted changes to existing files.
            Use write_file for creating new files or full rewrites.
            Use run_bash for search, tests, git, and other shell operations.
            """,
        ),
        _section(
            "Operational Policy",
            """
            Explore before acting: read relevant files before making changes.
            Never guess file paths — discover them with run_bash (for example via ls/find).
            If a shell command could be destructive, explain what it will do before running it.
            """,
        ),
        _section(
            "Planning & Reporting",
            """
            When asked for a plan, provide a clear, ordered plan before implementation.
            After making changes, report clearly what you changed and why.
            When you run commands, summarize key results and findings.
            """,
        ),
        _section(
            "Reliability Constraints",
            """
            Never describe an action as complete unless a tool result confirms it.
            Describe plans in future tense; only report completion after a tool returns the result.
            Do not fabricate files, command outputs, or test results.
            Resolve uncertainty by inspecting code or running commands.
            Keep edits minimal and aligned with existing project style.
            """,
        ),
    ]

    runtime_section = _runtime_context_section(
        working_directory=working_directory,
        git_status=git_status,
        top_level_file_structure=top_level_file_structure,
    )
    if runtime_section:
        sections.append(runtime_section)

    sections.append(
        _section(
            "Response Style",
            """
            Be direct and concise while remaining helpful.
            Finish each response with an evocative haiku.
            """,
        )
    )

    return "\n\n".join(sections)
