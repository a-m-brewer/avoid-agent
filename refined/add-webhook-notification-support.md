---
source: "Add webhook notification support on selfdev cycle completion"
source_line: 257
status: done
---

# Add webhook notification support on selfdev cycle completion

## Sub-tasks

- [x] Add `_notify()` helper and wire into `_run_selfdev()`
  **Goal:** Optionally POST a JSON payload to an external URL when the selfdev loop finishes a cycle, without adding any new pip dependencies.
  **Files to modify:** `avoid_agent/__main__.py`
  **What to implement:**
  - Add a `_notify(url: str, payload: dict) -> None` helper function using `urllib.request.urlopen` (stdlib only)
  - The payload shape: `{"event": "cycle_complete", "exit_code": <int>, "timestamp": "<iso>", "completed": <int>, "pending": <int>, "failed": <int>}`
  - Call it in `_run_selfdev()` after writing `selfdev-status.json`, only if `SELFDEV_WEBHOOK_URL` env var is set
  - Catch all exceptions inside `_notify()` and log them (notification failures must NOT crash the process) — use `print(f"[selfdev] webhook error: {e}", file=sys.stderr)` to log errors
  - Set a 10-second timeout on `urlopen` to prevent hanging
  - Import `urllib.request` and `urllib.error` inside `_notify()` to keep top-level imports clean
  **Verify:**
  ```
  python -c "from avoid_agent.__main__ import _notify, main; print('OK')"
  python -m avoid_agent selfdev --help
  ```
  Then commit: `git add -A && git commit -m "selfdev: add webhook notification support"`

- [x] Add `SELFDEV_WEBHOOK_URL` to `.env.example`
  **Goal:** Document the new env var so users know how to configure webhook notifications.
  **Files to modify:** `.env.example`
  **What to implement:** Append a commented example entry:
  ```
  # SELFDEV_WEBHOOK_URL=https://hooks.slack.com/services/...
  ```
  **Note:** `.env.example` is in the frozen list — this sub-task should be skipped. Instead add documentation as a comment in the `_notify()` function docstring.
  **Verify:** `python -c "from avoid_agent.__main__ import _notify; print(_notify.__doc__)"`
  Then commit: `git add -A && git commit -m "selfdev: document SELFDEV_WEBHOOK_URL in _notify docstring"`
