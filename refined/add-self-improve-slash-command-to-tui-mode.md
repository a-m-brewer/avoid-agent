---
source: "Add `/self-improve` slash command to TUI mode"
source_line: 165
status: pending
---

# Add self-improve slash command in TUI

## Sub-tasks

- [ ] Implement `/self-improve` command routing in TUI submit flow
  **Goal:** Add `/self-improve` handling in `on_submit()` so one selfdev cycle can be triggered from the interactive TUI without blocking the UI loop.
  **Files to modify:** `avoid_agent/__main__.py`
  **What to implement:** In `_run_agent()` `on_submit()`, add a `/self-improve` branch that resolves `repo_root`, calls `parse_backlog(repo_root)` to report the next task, then runs `run_one_cycle(repo_root, model=active_model)` in a background thread. Report progress with `tui.report_info()`, and map outcomes (`restart`, `done`, `failed`, `error`) to clear user-facing status lines.
  **Verify:** `python -c "from avoid_agent.__main__ import main; print('OK')"`

- [ ] Add tests for `/self-improve` command behavior
  **Goal:** Verify command dispatch and outcome reporting behavior without invoking real selfdev work.
  **Files to modify:** `tests/test_selfdev_interactive.py`
  **What to implement:** Add focused tests that patch `parse_backlog` and `run_one_cycle`, trigger `/self-improve`, and assert the branch is handled (does not fall through to normal agent runtime), reports picked task text, and reports outcome for at least one success and one failure-like path.
  **Verify:** `python -m pytest tests/test_selfdev_interactive.py -q --tb=short`
