"""Tests for interactive selfdev wiring and TUI status extensions."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from avoid_agent.__main__ import _run_agent, _run_selfdev
from avoid_agent.tui.components.status_bar import StatusBarComponent


def test_status_bar_renders_phase_and_progress() -> None:
    bar = StatusBarComponent(model="anthropic/claude-sonnet-4-6")
    bar.phase = "running"
    bar.progress_current = 2
    bar.progress_total = 5

    line = bar.render(200)[0]

    assert "phase: running" in line
    assert "2/5" in line


def test_run_selfdev_uses_legacy_loop_when_flag_enabled(monkeypatch) -> None:
    args = SimpleNamespace(model="anthropic/test", max_turns=12, single=True, legacy=True, interactive=False, operator=False)
    captured: dict[str, object] = {}

    def fake_run_loop(*, repo_root, model, max_turns, single):
        captured["repo_root"] = repo_root
        captured["model"] = model
        captured["max_turns"] = max_turns
        captured["single"] = single
        return 7

    with patch("avoid_agent.__main__.load_dotenv"), \
         patch("avoid_agent.selfdev.loop.run_loop", side_effect=fake_run_loop), \
         patch("avoid_agent.__main__._run_selfdev_observe") as observe, \
         patch("avoid_agent.__main__.sys.exit") as sys_exit:
        _run_selfdev(args)

    observe.assert_not_called()
    assert captured["model"] == "anthropic/test"
    assert captured["max_turns"] == 12
    assert captured["single"] is True
    sys_exit.assert_called_once_with(7)


def test_run_selfdev_uses_observe_by_default() -> None:
    args = SimpleNamespace(model="openai/gpt-5", max_turns=33, single=False, legacy=False, interactive=False, operator=False)

    with patch("avoid_agent.__main__.load_dotenv"), \
         patch("avoid_agent.selfdev.loop.run_loop") as run_loop, \
         patch("avoid_agent.__main__._run_selfdev_observe", return_value=42) as observe, \
         patch("avoid_agent.__main__.sys.exit") as sys_exit:
        _run_selfdev(args)

    run_loop.assert_not_called()
    observe.assert_called_once()
    kwargs = observe.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-5"
    assert kwargs["max_turns"] == 33
    sys_exit.assert_called_once_with(42)


def test_run_selfdev_uses_interactive_when_flag_set() -> None:
    args = SimpleNamespace(model="openai/gpt-5", max_turns=33, single=False, legacy=False, interactive=True, operator=False)

    with patch("avoid_agent.__main__.load_dotenv"), \
         patch("avoid_agent.selfdev.loop.run_loop") as run_loop, \
         patch("avoid_agent.__main__._run_selfdev_interactive", return_value=42) as interactive, \
         patch("avoid_agent.__main__.sys.exit") as sys_exit:
        _run_selfdev(args)

    run_loop.assert_not_called()
    interactive.assert_called_once()
    kwargs = interactive.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-5"
    assert kwargs["max_turns"] == 33
    sys_exit.assert_called_once_with(42)


def test_run_selfdev_uses_operator_when_flag_set() -> None:
    args = SimpleNamespace(model="anthropic/claude", max_turns=40, single=False, legacy=False, interactive=False, operator=True)

    with patch("avoid_agent.__main__.load_dotenv"), \
         patch("avoid_agent.__main__._run_selfdev_operator", return_value=42) as operator, \
         patch("avoid_agent.__main__.sys.exit") as sys_exit:
        _run_selfdev(args)

    operator.assert_called_once()
    kwargs = operator.call_args.kwargs
    assert kwargs["model"] == "anthropic/claude"
    assert kwargs["max_turns"] == 40
    sys_exit.assert_called_once_with(42)


def test_self_improve_no_items_reports_info(monkeypatch) -> None:
    """When backlog is empty, report that and return without spawning a thread."""
    submitted_tuis = []

    class FakeTUI:
        def __init__(self, **kwargs):
            self.on_submit = kwargs.get("on_submit")
            submitted_tuis.append(self)
        def set_thinking_enabled(self, *a): pass
        def set_effort(self, *a): pass
        def set_warning(self, *a): pass
        def push_item(self, *a): pass
        def run(self):
            self.on_submit("/self-improve")
        def stop(self): pass
        def set_model(self, *a): pass
        def clear_conversation(self): pass
        def ask_permission(self, cmd): return "allow"
        def append_chunk(self, text): pass
        def replace_last_assistant(self, text): pass
        def update_tokens(self, tokens): pass
        _info_calls = []
        def report_info(self, msg): self._info_calls.append(msg)
        _error_calls = []
        def report_error(self, msg): self._error_calls.append(msg)

    with patch("avoid_agent.__main__.TUI", FakeTUI), \
         patch("avoid_agent.selfdev.loop.parse_backlog", return_value=[]), \
         patch("avoid_agent.__main__.find_available_tools", return_value=[]), \
         patch("avoid_agent.__main__.providers.get_provider", return_value=MagicMock()), \
         patch("avoid_agent.__main__.build_system_prompt", return_value="sys"), \
         patch("avoid_agent.__main__.gather_initial_context", return_value=(".", "", "")), \
         patch("avoid_agent.__main__.get_saved_model", return_value="anthropic/test"), \
         patch("avoid_agent.__main__.load_user_config", return_value={}), \
         patch("avoid_agent.__main__.load_allowed", return_value=[]), \
         patch("avoid_agent.__main__.load_session", return_value=None), \
         patch("avoid_agent.__main__.load_dotenv"):
        _run_agent()

    assert len(submitted_tuis) == 1
    tui = submitted_tuis[0]
    assert any("No pending backlog items" in msg for msg in tui._info_calls)


def test_self_improve_reports_next_task(monkeypatch) -> None:
    """When there is a backlog item, report its text."""
    class FakeItem:
        def __init__(self, text):
            self.text = text

    submitted_tuis = []

    class FakeTUI:
        def __init__(self, **kwargs):
            self.on_submit = kwargs.get("on_submit")
            submitted_tuis.append(self)
        def set_thinking_enabled(self, *a): pass
        def set_effort(self, *a): pass
        def set_warning(self, *a): pass
        def push_item(self, *a): pass
        def run(self):
            self.on_submit("/self-improve")
        def stop(self): pass
        def set_model(self, *a): pass
        def clear_conversation(self): pass
        def ask_permission(self, cmd): return "allow"
        def append_chunk(self, text): pass
        def replace_last_assistant(self, text): pass
        def update_tokens(self, tokens): pass
        _info_calls = []
        def report_info(self, msg): self._info_calls.append(msg)
        _error_calls = []
        def report_error(self, msg): self._error_calls.append(msg)

    with patch("avoid_agent.__main__.TUI", FakeTUI), \
         patch("avoid_agent.selfdev.loop.parse_backlog", return_value=[FakeItem("my important task")]), \
         patch("avoid_agent.selfdev.loop.run_one_cycle", return_value="restart"), \
         patch("avoid_agent.__main__.find_available_tools", return_value=[]), \
         patch("avoid_agent.__main__.providers.get_provider", return_value=MagicMock()), \
         patch("avoid_agent.__main__.build_system_prompt", return_value="sys"), \
         patch("avoid_agent.__main__.gather_initial_context", return_value=(".", "", "")), \
         patch("avoid_agent.__main__.get_saved_model", return_value="anthropic/test"), \
         patch("avoid_agent.__main__.load_user_config", return_value={}), \
         patch("avoid_agent.__main__.load_allowed", return_value=[]), \
         patch("avoid_agent.__main__.load_session", return_value=None), \
         patch("avoid_agent.__main__.load_dotenv"):
        _run_agent()

    assert len(submitted_tuis) == 1
    tui = submitted_tuis[0]
    assert any("Next task: my important task" in msg for msg in tui._info_calls)


def test_self_improve_restart_reports_success(monkeypatch) -> None:
    """When run_one_cycle returns restart, report success."""
    class FakeItem:
        def __init__(self, text):
            self.text = text

    submitted_tuis = []

    class FakeTUI:
        def __init__(self, **kwargs):
            self.on_submit = kwargs.get("on_submit")
            submitted_tuis.append(self)
        def set_thinking_enabled(self, *a): pass
        def set_effort(self, *a): pass
        def set_warning(self, *a): pass
        def push_item(self, *a): pass
        def run(self):
            self.on_submit("/self-improve")
        def stop(self): pass
        def set_model(self, *a): pass
        def clear_conversation(self): pass
        def ask_permission(self, cmd): return "allow"
        def append_chunk(self, text): pass
        def replace_last_assistant(self, text): pass
        def update_tokens(self, tokens): pass
        _info_calls = []
        def report_info(self, msg): self._info_calls.append(msg)
        _error_calls = []
        def report_error(self, msg): self._error_calls.append(msg)

    with patch("avoid_agent.__main__.TUI", FakeTUI), \
         patch("avoid_agent.selfdev.loop.parse_backlog", return_value=[FakeItem("some task")]), \
         patch("avoid_agent.selfdev.loop.run_one_cycle", return_value="restart"), \
         patch("avoid_agent.__main__.find_available_tools", return_value=[]), \
         patch("avoid_agent.__main__.providers.get_provider", return_value=MagicMock()), \
         patch("avoid_agent.__main__.build_system_prompt", return_value="sys"), \
         patch("avoid_agent.__main__.gather_initial_context", return_value=(".", "", "")), \
         patch("avoid_agent.__main__.get_saved_model", return_value="anthropic/test"), \
         patch("avoid_agent.__main__.load_user_config", return_value={}), \
         patch("avoid_agent.__main__.load_allowed", return_value=[]), \
         patch("avoid_agent.__main__.load_session", return_value=None), \
         patch("avoid_agent.__main__.load_dotenv"):
        _run_agent()

    assert len(submitted_tuis) == 1
    tui = submitted_tuis[0]
    assert any("restart is recommended" in msg for msg in tui._info_calls)


def test_self_improve_failed_reports_error(monkeypatch) -> None:
    """When run_one_cycle returns failed, report that the cycle failed."""
    class FakeItem:
        def __init__(self, text):
            self.text = text

    submitted_tuis = []

    class FakeTUI:
        def __init__(self, **kwargs):
            self.on_submit = kwargs.get("on_submit")
            submitted_tuis.append(self)
        def set_thinking_enabled(self, *a): pass
        def set_effort(self, *a): pass
        def set_warning(self, *a): pass
        def push_item(self, *a): pass
        def run(self):
            self.on_submit("/self-improve")
        def stop(self): pass
        def set_model(self, *a): pass
        def clear_conversation(self): pass
        def ask_permission(self, cmd): return "allow"
        def append_chunk(self, text): pass
        def replace_last_assistant(self, text): pass
        def update_tokens(self, tokens): pass
        _info_calls = []
        def report_info(self, msg): self._info_calls.append(msg)
        _error_calls = []
        def report_error(self, msg): self._error_calls.append(msg)

    with patch("avoid_agent.__main__.TUI", FakeTUI), \
         patch("avoid_agent.selfdev.loop.parse_backlog", return_value=[FakeItem("some task")]), \
         patch("avoid_agent.selfdev.loop.run_one_cycle", return_value="failed"), \
         patch("avoid_agent.__main__.find_available_tools", return_value=[]), \
         patch("avoid_agent.__main__.providers.get_provider", return_value=MagicMock()), \
         patch("avoid_agent.__main__.build_system_prompt", return_value="sys"), \
         patch("avoid_agent.__main__.gather_initial_context", return_value=(".", "", "")), \
         patch("avoid_agent.__main__.get_saved_model", return_value="anthropic/test"), \
         patch("avoid_agent.__main__.load_user_config", return_value={}), \
         patch("avoid_agent.__main__.load_allowed", return_value=[]), \
         patch("avoid_agent.__main__.load_session", return_value=None), \
         patch("avoid_agent.__main__.load_dotenv"):
        _run_agent()

    assert len(submitted_tuis) == 1
    tui = submitted_tuis[0]
    assert any("Cycle failed. Branch preserved for review" in msg for msg in tui._info_calls)
