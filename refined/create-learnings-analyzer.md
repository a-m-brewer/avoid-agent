---
source: "Create learnings analyzer `avoid_agent/learnings_analyzer.py`"
source_line: 118
status: pending
---

# Add learnings analyzer

## Sub-tasks

- [ ] Implement learnings analyzer module
  **Goal:** Parse session learnings markdown files and emit recurring-error backlog suggestions.
  **Files to modify:** `avoid_agent/learnings_analyzer.py`
  **What to implement:** Add `analyze(learnings_dir: Path) -> list[str]` that safely returns `[]` when directory/files are absent, reads each `.md`, extracts the `## Errors` section, groups session-level occurrences by keywords (`OSError`, `PermissionError`, `JSON`, `timeout`, `tool not found`, `import`), and returns plain-English suggestions for keywords seen in 3+ files.
  **Verify:** `python -c "from avoid_agent.learnings_analyzer import analyze; print(analyze.__doc__)"`
