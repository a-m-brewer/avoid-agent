---
source: "Add `/learnings` slash command to TUI mode"
source_line: 145
status: done
---

# Add learnings slash commands in TUI

## Sub-tasks

- [x] Implement `/learnings` command handling in TUI submit flow
  **Goal:** Add `/learnings`, `/learnings clear`, and `/learnings suggest` handling in `on_submit()` without crashing when `.learnings/` is absent.
  **Files to modify:** `avoid_agent/__main__.py`
  **What to implement:** Add command routing in `on_submit()` using `text.strip().startswith("/learnings")`. For `/learnings`, count `.learnings/sessions/*.md` and display up to 3 recent error lines from `## Errors`. For `/learnings clear`, delete session markdown files and report removed count. For `/learnings suggest`, call `analyze()` from `avoid_agent/learnings_analyzer.py` and print suggestions or `No suggestions yet`.
  **Verify:** `python -c "from avoid_agent.__main__ import main; print('OK')"`

- [x] Add tests for learnings slash command helpers
  **Goal:** Verify recent error extraction, clear behavior, and analyzer suggestion display paths.
  **Files to modify:** `tests/test_main_display.py`, `tests/test_tui_clear.py`
  **What to implement:** Add focused unit tests for any new helper functions in `avoid_agent/__main__.py` and command behavior branches with temporary `.learnings/sessions` fixtures and mocked TUI reporter methods.
  **Verify:** `python -m pytest tests/test_main_display.py tests/test_tui_clear.py -q --tb=short`
