---
source: "Create session learnings capture module `avoid_agent/learnings.py`"
source_line: 69
status: done
---

# Add session learnings capture writer

## Sub-tasks

- [x] Implement learnings capture module
  **Goal:** Add `capture_session()` to write timestamped markdown learnings entries for errored headless sessions.
  **Files to modify:** `avoid_agent/learnings.py`
  **What to implement:** Create `capture_session(session_id, tool_calls, errors)` that returns `None` if there are no explicit errors and no failed tool calls (`is_error=True`), otherwise writes `.learnings/sessions/<YYYYMMDD-HHMMSS>-<session_id[:8]>.md` with frontmatter and `## Errors` and `## Failed Tool Calls` sections using stdlib only.
  **Verify:** `python -c "from avoid_agent.learnings import capture_session; print('OK')"`
