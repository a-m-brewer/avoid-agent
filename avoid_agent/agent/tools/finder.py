"""This module provides functionality to discover and load available tools for the agent."""

import importlib
import logging
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

from avoid_agent.agent.tools import ToolDefinition, tool_registry
from avoid_agent.agent.tools.inspector import generate_tool_schema

logger = logging.getLogger(__name__)


def _load_tools_from_directory(directory: Path) -> None:
    """Import all Python files in a directory so their @tool decorators run."""
    for file in directory.glob("*.py"):
        if file.name == "__init__.py":
            continue

        module_name = f"avoid_agent_dynamic_tools_{file.stem}_{abs(hash(file.resolve()))}"
        spec = importlib.util.spec_from_file_location(module_name, file)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)  # @tool decorators fire


def _load_tools_from_pyproject(pyproject_path: Path) -> None:
    """Load tool directories configured in pyproject.toml."""
    config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    tool_config = config.get("tool", {}).get("avoid_agent", {})

    for directory in tool_config.get("tool_directories", []):
        _load_tools_from_directory(pyproject_path.parent / directory)


def find_available_tools(
    tool_directories: list[str | Path] | None = None,
    pyproject_paths: list[str | Path] | None = None,
) -> list[ToolDefinition]:
    """Find all available tools by inspecting built-ins, directories, and config."""
    # Load built-in tools
    # pylint: disable=import-outside-toplevel,unused-import
    import avoid_agent.agent.tools.core

    if tool_directories is not None:
        for directory in tool_directories:
            _load_tools_from_directory(Path(directory))

    if pyproject_paths is not None:
        for configured_pyproject in pyproject_paths:
            _load_tools_from_pyproject(Path(configured_pyproject))

    # Load extension tools via entry points
    for ep in entry_points(group="avoid_agent.tools"):
        ep.load()  # importing the module triggers @tool decorators

    tools: list[ToolDefinition] = []
    seen_tools: set[str] = set()

    for func in tool_registry.values():
        tool_definition = generate_tool_schema(func)
        if tool_definition.name in seen_tools:
            logger.warning(
                "Skipping duplicate tool '%s' from %s",
                tool_definition.name,
                func.__module__,
            )
            continue

        seen_tools.add(tool_definition.name)
        tools.append(tool_definition)

    return tools


def get_tool_descriptions() -> dict[str, str]:
    """Get a dict of tool_name -> description for all registered tools.
    
    This enables dynamic discovery of extension tools for system prompt generation.
    """
    tools = find_available_tools()
    return {t.name: t.description.split('\n')[0].strip() for t in tools}
