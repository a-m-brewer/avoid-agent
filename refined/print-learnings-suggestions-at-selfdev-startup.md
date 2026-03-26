---
source: "Print learnings suggestions at selfdev startup"
source_line: 209
status: done
---

# Print learnings suggestions at selfdev startup

## Sub-tasks

- [x] Add learnings suggestions logging to `_run_selfdev()`
  **Goal:** Before the selfdev loop picks a task, log any analyzer suggestions so the operator can see recurring patterns in the terminal output without having to open the TUI.
  **Files to modify:** `avoid_agent/__main__.py`
  **What to implement:** In `_run_selfdev()`, before calling `run_loop()`, resolve the `.learnings/sessions/` path relative to `repo_root`, call `analyze()` from `avoid_agent/learnings_analyzer.py`, and if suggestions exist print each using the same log format as other selfdev messages with a `[suggestion]` prefix. If there are no suggestions, print nothing (don't add noise). Use a lazy local import to avoid circular dependencies.
  **Verify:** `python -m avoid_agent selfdev --help` and `python -c "from avoid_agent.__main__ import main; print('OK')"`
