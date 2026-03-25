---
source: "Wire learnings capture into headless mode"
source_line: 97
status: done
---

# Wire learnings capture into headless mode

## Sub-tasks

- [x] Capture headless single-turn failures in .learnings
  **Goal:** Ensure single-turn headless runs write learnings entries when errors or failed tool calls occur.
  **Files to modify:** `avoid_agent/__main__.py`
  **What to implement:** In `_run_headless()` single-turn mode, lazily import `capture_session`, collect failed tool calls from `result["tool_calls"]` where `is_error` is true, collect `result["error"]` if non-empty, then call `capture_session(session_id, failed_tool_calls, errors)` before printing/exiting.
  **Verify:** `python -c "from avoid_agent.__main__ import main; print('OK')"`
