"""Tests for interactive selfdev wiring and TUI status extensions."""

from types import SimpleNamespace
from unittest.mock import patch

from avoid_agent.__main__ import _run_selfdev
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
