"""Tests for the finder module."""

# pylint: disable=unused-argument,missing-class-docstring

import importlib
import sys
import textwrap

from avoid_agent.agent.tools import ParameterType
from avoid_agent.agent.tools import finder, tool_registry
from avoid_agent.agent.tools.finder import find_available_tools


class TestFindAvailableTools:

    def test_returns_list_of_tool_definitions(self):
        tools = find_available_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0

    def test_includes_read_file(self):
        tools = find_available_tools()
        names = [t.name for t in tools]
        assert "read_file" in names

    def test_read_file_schema_is_correct(self):
        tools = find_available_tools()
        read_file_tool = next(t for t in tools if t.name == "read_file")
        assert read_file_tool.description == "Read the contents of a file at the given path."
        assert len(read_file_tool.parameters) == 1
        assert read_file_tool.parameters[0].name == "path"
        assert read_file_tool.parameters[0].type == ParameterType.STR
        assert read_file_tool.parameters[0].required is True

    def test_loads_tools_from_directory(self, tmp_path):
        baseline_tools = find_available_tools()
        baseline_registry = tool_registry.copy()
        assert len(baseline_tools) > 0

        tool_file = tmp_path / "extra_tool.py"
        tool_file.write_text(
            textwrap.dedent(
                """
                from typing_extensions import Annotated

                from avoid_agent.agent.tools import tool


                @tool
                def temp_tool(path: Annotated[str, "A path"]) -> str:
                    \"\"\"A temp tool loaded from a directory.\"\"\"
                    return path
                """
            ),
            encoding="utf-8",
        )

        try:
            tools = find_available_tools(tool_directories=[tmp_path])
        finally:
            tool_registry.clear()
            tool_registry.update(baseline_registry)

        temp_tool = next(tool for tool in tools if tool.name == "temp_tool")
        assert temp_tool.description == "A temp tool loaded from a directory."
        assert len(temp_tool.parameters) == 1
        assert temp_tool.parameters[0].name == "path"
        assert temp_tool.parameters[0].type == ParameterType.STR
        assert temp_tool.parameters[0].required is True

    def test_loads_tools_from_pyproject_config(self, tmp_path):
        find_available_tools()
        baseline_registry = tool_registry.copy()

        tool_dir = tmp_path / "tools"
        tool_dir.mkdir()

        tool_file = tool_dir / "pyproject_tool.py"
        tool_file.write_text(
            textwrap.dedent(
                """
                from typing_extensions import Annotated

                from avoid_agent.agent.tools import tool


                @tool
                def pyproject_tool(name: Annotated[str, "A name"]) -> str:
                    \"\"\"Loaded via pyproject config.\"\"\"
                    return name
                """
            ),
            encoding="utf-8",
        )

        pyproject_file = tmp_path / "pyproject.toml"
        pyproject_file.write_text(
            textwrap.dedent(
                """
                [tool.avoid_agent]
                tool_directories = ["tools"]
                """
            ),
            encoding="utf-8",
        )

        try:
            tools = find_available_tools(pyproject_paths=[pyproject_file])
        finally:
            tool_registry.clear()
            tool_registry.update(baseline_registry)

        pyproject_tool = next(tool for tool in tools if tool.name == "pyproject_tool")
        assert pyproject_tool.description == "Loaded via pyproject config."
        assert len(pyproject_tool.parameters) == 1
        assert pyproject_tool.parameters[0].name == "name"
        assert pyproject_tool.parameters[0].type == ParameterType.STR
        assert pyproject_tool.parameters[0].required is True

    def test_loads_tools_from_multiple_pyproject_configs(self, tmp_path):
        find_available_tools()
        baseline_registry = tool_registry.copy()

        first_root = tmp_path / "project_one"
        second_root = tmp_path / "project_two"
        first_root.mkdir()
        second_root.mkdir()

        first_tools_dir = first_root / "tools"
        second_tools_dir = second_root / "tools"
        first_tools_dir.mkdir()
        second_tools_dir.mkdir()

        (first_tools_dir / "first_tool.py").write_text(
            textwrap.dedent(
                """
                from typing_extensions import Annotated

                from avoid_agent.agent.tools import tool


                @tool
                def first_project_tool(name: Annotated[str, "A name"]) -> str:
                    \"\"\"Loaded from the first pyproject file.\"\"\"
                    return name
                """
            ),
            encoding="utf-8",
        )

        (second_tools_dir / "second_tool.py").write_text(
            textwrap.dedent(
                """
                from typing_extensions import Annotated

                from avoid_agent.agent.tools import tool


                @tool
                def second_project_tool(path: Annotated[str, "A path"]) -> str:
                    \"\"\"Loaded from the second pyproject file.\"\"\"
                    return path
                """
            ),
            encoding="utf-8",
        )

        first_pyproject = first_root / "pyproject.toml"
        second_pyproject = second_root / "pyproject.toml"

        first_pyproject.write_text(
            textwrap.dedent(
                """
                [tool.avoid_agent]
                tool_directories = ["tools"]
                """
            ),
            encoding="utf-8",
        )
        second_pyproject.write_text(
            textwrap.dedent(
                """
                [tool.avoid_agent]
                tool_directories = ["tools"]
                """
            ),
            encoding="utf-8",
        )

        try:
            tools = find_available_tools(pyproject_paths=[first_pyproject, second_pyproject])
        finally:
            tool_registry.clear()
            tool_registry.update(baseline_registry)

        names = [tool.name for tool in tools]
        assert "first_project_tool" in names
        assert "second_project_tool" in names

    def test_skips_duplicate_tools_from_multiple_pyprojects(self, tmp_path, caplog):
        find_available_tools()
        baseline_registry = tool_registry.copy()

        first_root = tmp_path / "project_one"
        second_root = tmp_path / "project_two"
        first_root.mkdir()
        second_root.mkdir()

        first_tools_dir = first_root / "tools"
        second_tools_dir = second_root / "tools"
        first_tools_dir.mkdir()
        second_tools_dir.mkdir()

        (first_tools_dir / "dup_tool.py").write_text(
            textwrap.dedent(
                """
                from typing_extensions import Annotated

                from avoid_agent.agent.tools import tool


                @tool
                def shared_tool(value: Annotated[str, "First value"]) -> str:
                    \"\"\"First duplicate tool definition.\"\"\"
                    return value
                """
            ),
            encoding="utf-8",
        )

        (second_tools_dir / "dup_tool.py").write_text(
            textwrap.dedent(
                """
                from typing_extensions import Annotated

                from avoid_agent.agent.tools import tool


                @tool
                def shared_tool(value: Annotated[str, "Second value"]) -> str:
                    \"\"\"Second duplicate tool definition.\"\"\"
                    return value
                """
            ),
            encoding="utf-8",
        )

        first_pyproject = first_root / "pyproject.toml"
        second_pyproject = second_root / "pyproject.toml"

        first_pyproject.write_text(
            textwrap.dedent(
                """
                [tool.avoid_agent]
                tool_directories = ["tools"]
                """
            ),
            encoding="utf-8",
        )
        second_pyproject.write_text(
            textwrap.dedent(
                """
                [tool.avoid_agent]
                tool_directories = ["tools"]
                """
            ),
            encoding="utf-8",
        )

        try:
            with caplog.at_level("WARNING"):
                tools = find_available_tools(pyproject_paths=[first_pyproject, second_pyproject])
        finally:
            tool_registry.clear()
            tool_registry.update(baseline_registry)

        shared_tools = [tool for tool in tools if tool.name == "shared_tool"]
        assert len(shared_tools) == 1
        assert shared_tools[0].description == "Second duplicate tool definition."

    def test_loads_tools_from_installed_entry_points(self, tmp_path, monkeypatch):
        find_available_tools()
        baseline_registry = tool_registry.copy()

        package_dir = tmp_path / "installed_plugin"
        package_dir.mkdir()

        init_file = package_dir / "__init__.py"
        init_file.write_text(
            textwrap.dedent(
                """
                from typing_extensions import Annotated

                from avoid_agent.agent.tools import tool


                @tool
                def installed_tool(value: Annotated[str, "A value"]) -> str:
                    \"\"\"Loaded from an installed package entry point.\"\"\"
                    return value
                """
            ),
            encoding="utf-8",
        )

        monkeypatch.syspath_prepend(str(tmp_path))

        class FakeEntryPoint:
            def load(self):
                return importlib.import_module("installed_plugin")

        monkeypatch.setattr(finder, "entry_points", lambda group: [FakeEntryPoint()])
        sys.modules.pop("installed_plugin", None)

        try:
            tools = find_available_tools()
        finally:
            tool_registry.clear()
            tool_registry.update(baseline_registry)
            sys.modules.pop("installed_plugin", None)

        installed_tool = next(tool for tool in tools if tool.name == "installed_tool")
        assert installed_tool.description == "Loaded from an installed package entry point."
        assert len(installed_tool.parameters) == 1
        assert installed_tool.parameters[0].name == "value"
        assert installed_tool.parameters[0].type == ParameterType.STR
        assert installed_tool.parameters[0].required is True
