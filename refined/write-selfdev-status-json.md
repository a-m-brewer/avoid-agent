---
source: "Write `selfdev-status.json` after each selfdev cycle"
source_line: 234
status: pending
---

# Write selfdev-status.json after each selfdev cycle

## Sub-tasks

- [x] Add status JSON writing to `_run_selfdev()`
  **Goal:** Enable external monitoring by writing a machine-readable status file after each selfdev run.
  **Files to modify:** `avoid_agent/__main__.py`
  **What to implement:** In `_run_selfdev()`, after `run_loop()` returns `exit_code` but before `sys.exit(exit_code)`, write `<repo_root>/selfdev-status.json` using `json.dump()` with `indent=2`. Fields: `last_run` (ISO 8601 timestamp from `datetime.now().isoformat()`), `exit_code` (integer), `completed_count` (count of `[x]` lines in `backlog.md`), `pending_count` (count of `[ ]` lines), `failed_count` (count of `[!]` lines). Read `avoid_agent/selfdev/loop.py` to understand how to parse `backlog.md` for counting.
  **Verify:** Run `python -m avoid_agent selfdev --single` and confirm `selfdev-status.json` is written with valid JSON.

- [ ] Add `selfdev-status.json` to `.gitignore`
  **Goal:** Prevent the status file from being committed to the repository.
  **Files to modify:** `.gitignore`
  **What to implement:** Check if `.gitignore` exists. If it does, append `selfdev-status.json` to a new line if not already present. If `.gitignore` doesn't exist, create it with `selfdev-status.json` as the first line.
  **Verify:** `grep -q "selfdev-status.json" .gitignore` returns exit code 0.
