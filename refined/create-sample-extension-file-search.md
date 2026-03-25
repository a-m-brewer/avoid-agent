---
source: "Create sample extension `extensions/file_search/__init__.py`"
source_line: 42
status: done
---

# Add sample file_search extension

## Sub-tasks

- [x] Implement file_search extension tool module
  **Goal:** Add a sample extension that exposes a `file_search` tool using grep and returns structured `ToolRunResult` output.
  **Files to modify:** `extensions/file_search/__init__.py`
  **What to implement:** Create a `@tool`-decorated `file_search(pattern, directory=".")` function that runs `grep -rn --include="*.py"` with `subprocess.run`, returns up to 50 matching lines, treats exit code 1 as "no matches", and reports `OSError` failures cleanly.
  **Verify:** `python -c "from avoid_agent.agent.tools.finder import find_available_tools; names=[t.name for t in find_available_tools()]; assert 'file_search' in names; print('OK')"`
