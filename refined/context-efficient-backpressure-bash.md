---
source: "Implement context-efficient backpressure for bash output"
source_line: 319
status: pending
---

# Context-efficient backpressure for bash output

## Sub-tasks

- [x] Add output suppression for verbose successful bash commands
  **Goal:** When `run_bash` succeeds and returns more than 20 lines of stdout, replace the full
  output with a compact summary to save context tokens. Failures always return full output.
  **Files to modify:** `avoid_agent/agent/tools/core.py`
  **What to implement:**
  - Add `_BASH_QUIET_THRESHOLD = 20` constant near `_PREVIEW_LIMIT`
  - In `run_bash()`, after `subprocess.run()`, when `result.returncode == 0`:
    - Count lines: `stdout_lines = result.stdout.count('\n')`
    - If `stdout_lines > _BASH_QUIET_THRESHOLD`, replace stdout in content with:
      `f"Exit code: 0 (success, {stdout_lines} lines of output suppressed — re-run with verbose flag if needed)"`
    - Keep stderr in output regardless
  - Do NOT suppress on non-zero exit codes (failures need full output for debugging)
  - The `proof` details (`stdout_preview`, `stderr_preview`) remain unchanged
  **Verify:**
  ```
  python -c "
  from avoid_agent.agent.tools.core import run_bash
  result = run_bash(command='seq 1 100')
  print(repr(result))
  assert 'suppressed' in str(result), 'Expected suppressed output for long successful command'
  print('OK')
  "
  ```

- [ ] Add tests for bash output backpressure
  **Goal:** Verify that long successful outputs are suppressed, short outputs are kept, and
  failed commands always return full output.
  **Files to modify:** `tests/test_agent_runtime.py` (or create `tests/test_bash_backpressure.py`)
  **What to implement:**
  - Test: `run_bash('seq 1 100')` → content contains "suppressed", not all 100 lines
  - Test: `run_bash('echo hello')` → content contains "hello" (short output preserved)
  - Test: `run_bash('seq 1 100; exit 1')` → content contains full output (failure not suppressed)
  - Test: `run_bash('seq 1 100')` with stderr → stderr still appears in content
  **Verify:** `python -m pytest tests/ -k "backpressure or bash" -q --tb=short`
