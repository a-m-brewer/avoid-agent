# Backlog

Items are worked top-to-bottom. The self-dev loop picks the first unchecked item.
Mark items `[x]` when done, `[!]` if failed (with a note).

## Phase 1: Skills system (safe, no restart needed)

- [x] Create `skills/` directory with auto-discovery in the system prompt builder
  - Scan `skills/` and `~/.avoid-agent/skills/` for `SKILL.md` files
  - Load only `name` and `description` from frontmatter into system prompt
  - Load full skill body on-demand when the agent references a skill by name
  - Update `avoid_agent/prompts/system_prompt.py` to include discovered skill summaries

- [x] Create a sample built-in skill `skills/self-review/SKILL.md`
  - A skill that instructs the agent how to review its own recent changes
  - Should reference `git log`, `git diff`, and reading modified files

## Phase 2: Extensions system (tools via worktree + validation)

- [x] Wire `extensions/` auto-discovery into `find_available_tools()`
  **Goal:** Any `.py` file placed in `extensions/` at the project root (or `~/.avoid-agent/extensions/`)
  is auto-imported when the agent starts, registering its `@tool` functions without any manual config.

  **Files to read first:**
  - `avoid_agent/agent/tools/finder.py` — `find_available_tools()` already has `_load_tools_from_directory(path)`;
    you need to call it automatically for the two discovery paths
  - `avoid_agent/prompts/system_prompt.py` — see `_default_skill_search_paths()` for the exact pattern
    used to resolve `<project_root>/skills` and `~/.avoid-agent/skills`; replicate this for extensions

  **File to modify:** `avoid_agent/agent/tools/finder.py`

  **What to change:** Before the entry-points loop in `find_available_tools()`, add calls to
  `_load_tools_from_directory()` for the two paths. Use `Path(__file__).resolve().parent.parent.parent`
  to locate the project root. Skip directories that don't exist (`if not directory.is_dir(): continue`).
  Wrap in a try/except so a broken extension file doesn't crash the agent.

  **Verify:**
  ```
  python -c "from avoid_agent.agent.tools.finder import find_available_tools; find_available_tools()"
  ```

- [x] Create sample extension `extensions/file_search/__init__.py`
  **Goal:** Demonstrate the extension pattern and give the agent a `file_search` tool it can use
  to locate code across the codebase.

  **Files to read first:**
  - `avoid_agent/agent/tools/core.py` — follow the exact `@tool` decorator and `ToolRunResult` pattern
  - `avoid_agent/agent/tools/__init__.py` — understand what `tool` and `ToolRunResult` are imported from

  **File to create:** `extensions/file_search/__init__.py`

  **What to implement:** A single `file_search(pattern, directory=".")` function decorated with `@tool`.
  Run `grep -rn --include="*.py" <pattern> <directory>` via `subprocess.run`. Return up to 50 matching
  lines. Handle non-zero exit codes (no matches) and `OSError` gracefully. Use `ToolRunResult` for the
  return value, as in `core.py`.

  **Verify:** With extensions/ wired up, `file_search` appears in the tool list:
  ```
  python -c "
  from avoid_agent.agent.tools.finder import find_available_tools
  names = [t.name for t in find_available_tools()]
  assert 'file_search' in names, f'not found in {names}'
  print('OK:', names)
  "
  ```

## Phase 3: Learnings system (error capture + backlog feeding)

- [x] Create session learnings capture module `avoid_agent/learnings.py`
  **Goal:** A small module that writes timestamped error summaries to `.learnings/sessions/` after
  each headless run. The selfdev loop already reads from this directory (`_gather_learnings()` in
  `avoid_agent/selfdev/loop.py`) — this task creates the write side.

  **Files to read first:**
  - `avoid_agent/selfdev/loop.py` — read `_gather_learnings()` to understand the file format and naming
    convention that the loop expects when reading files back
  - `avoid_agent/session.py` — follow the same file I/O style (pathlib, `mkdir(parents=True, exist_ok=True)`)

  **File to create:** `avoid_agent/learnings.py`

  **What to implement:**
  ```python
  def capture_session(session_id: str, tool_calls: list[dict], errors: list[str]) -> Path | None:
      """Write a learnings entry. Returns the path written, or None if nothing to log."""
  ```
  - Only write if there is at least one error string or a tool call with `is_error=True`
  - Output path: `.learnings/sessions/<YYYYMMDD-HHMMSS>-<session_id[:8]>.md`
  - Format: manual YAML-style frontmatter (`---\nsession_id: ...\ntimestamp: ...\nerror_count: ...\n---\n`)
    followed by a `## Errors` section and a `## Failed Tool Calls` section
  - No external dependencies — use only stdlib (`pathlib`, `datetime`)

  **Verify:**
  ```
  python -c "from avoid_agent.learnings import capture_session; print('OK')"
  ```

- [x] Wire learnings capture into headless mode
  **Goal:** After each single-turn headless run, log any errors and failed tool calls to `.learnings/`.

  **Files to read first:**
  - `avoid_agent/__main__.py` — focus on `run_one_turn()` inside `_run_headless()`; the `tool_calls_log`
    list already tracks `is_error` per tool call; errors are emitted via `emit_event({"type": "error", ...})`
  - `avoid_agent/learnings.py` — the `capture_session()` function you'll be calling

  **File to modify:** `avoid_agent/__main__.py`

  **What to change:** In the single-turn path (after `run_one_turn()` returns and before `sys.exit()`),
  extract failed tool calls from `result["tool_calls"]` (those with `is_error=True`) and any error
  string from `result["error"]`. Call `capture_session()` with these. Use a lazy local import inside
  `_run_headless()` to avoid any import ordering issues.

  **Verify:** Run a clean headless call and confirm no `.learnings/` file is written for a successful run:
  ```
  python -m avoid_agent headless --prompt "say hello" --auto-approve --no-session
  ls .learnings/sessions/ 2>/dev/null || echo "directory empty or missing — correct"
  ```

- [ ] Create learnings analyzer `avoid_agent/learnings_analyzer.py`
  **Goal:** Scan `.learnings/sessions/` and surface recurring error patterns as plain-English
  backlog item suggestions. The selfdev startup log can print these so the operator sees them.

  **Files to read first:**
  - `avoid_agent/selfdev/loop.py` — see `_gather_learnings()` for how to iterate the sessions directory
    and read the markdown files
  - `avoid_agent/learnings.py` — understand the exact file format (sections, frontmatter) you'll parse

  **File to create:** `avoid_agent/learnings_analyzer.py`

  **What to implement:**
  ```python
  def analyze(learnings_dir: Path) -> list[str]:
      """Return a list of plain-English backlog item suggestions based on recurring errors."""
  ```
  - Iterate `.md` files in `learnings_dir`, read the `## Errors` section from each
  - Group by keyword: `OSError`, `PermissionError`, `JSON`, `timeout`, `tool not found`, `import`
  - If a keyword appears in 3+ session files, emit a suggestion string like
    `"Recurring JSON parse errors (seen in 4 sessions) — consider improving error handling in ..."`
  - Return an empty list if `learnings_dir` doesn't exist or has no files

  **Verify:**
  ```
  python -c "from avoid_agent.learnings_analyzer import analyze; print(analyze.__doc__)"
  ```

- [ ] Add `/learnings` slash command to TUI mode
  **Goal:** Let the user inspect recent session errors and get backlog suggestions without leaving the TUI.

  **Files to read first:**
  - `avoid_agent/__main__.py` — study the pattern of `/strategy`, `/thinking`, `/effort` commands
    in `on_submit()`; follow the exact same structure (check `text.strip().startswith("/learnings")`)
  - `avoid_agent/learnings_analyzer.py` — use `analyze()` to generate suggestions

  **File to modify:** `avoid_agent/__main__.py` in `on_submit()`

  **What to add:**
  - `/learnings` — count `.learnings/sessions/*.md` files and show the 3 most recent error lines
  - `/learnings clear` — delete all files in `.learnings/sessions/` and report how many were removed
  - `/learnings suggest` — call `analyze()` and display proposed backlog items, or "No suggestions yet"

  **Verify:** Start the TUI and type `/learnings`. It must not crash even when `.learnings/` is absent.
  Expected output: `"No learnings yet"` or similar.

## Phase 4: Autonomous improvements

- [ ] Add `/self-improve` slash command to TUI mode
  **Goal:** Let the user trigger one self-improvement cycle interactively from within the TUI,
  see which task was picked, and be informed of the outcome without an auto-restart.

  **Files to read first:**
  - `avoid_agent/__main__.py` — study `on_submit()`, especially how `/model` uses a background call
    and `tui.report_info()` to stream progress; note how `active_model` is accessed from the closure
  - `avoid_agent/selfdev/loop.py` — read `run_one_cycle(repo_root, model, max_turns)` and its four
    return values: `"restart"`, `"done"`, `"failed"`, `"error"`; also read `parse_backlog()` to show
    the user which task will be picked before starting

  **File to modify:** `avoid_agent/__main__.py`

  **What to add in `on_submit()`:**
  - `/self-improve` — call `parse_backlog(repo_root)` to find and display the next task, then call
    `run_one_cycle(repo_root, model=active_model)` in a background thread to avoid blocking the TUI
  - Report progress via `tui.report_info()` for each stage (picked, running, outcome)
  - On `"restart"`: tell the user changes were merged and a restart is recommended (do NOT auto-restart)
  - On `"done"`: tell the user the backlog is empty
  - On `"failed"` or `"error"`: show the branch name preserved for review

  **Verify:** `/self-improve` is recognized (doesn't fall through to the agent). Import check:
  ```
  python -c "from avoid_agent.__main__ import main; print('OK')"
  ```

- [ ] Add size labels to all unchecked backlog items
  **Goal:** Annotate each unchecked `backlog.md` item with a rough size estimate so future
  Claude Code reviews can triage quickly. This is a `backlog.md`-only change — no code needed.

  **File to modify:** `backlog.md`

  **Convention:** Add `<!-- size:S -->`, `<!-- size:M -->`, or `<!-- size:L -->` at the end of the
  first line of each unchecked (`[ ]`) item:
  - `S` = single file, <20 lines changed
  - `M` = 1–3 files, moderate changes
  - `L` = multiple files or a significant new module

  **What to do:** Go through every unchecked item and add the appropriate label. Read the item's
  sub-bullets to estimate scope. Do not modify checked (`[x]`) or failed (`[!]`) items.

  **Verify:** Every unchecked item's first line ends with a `<!-- size:_ -->` comment.
  No `[ ]` lines should be missing the label after this change.

- [ ] Print learnings suggestions at selfdev startup
  **Goal:** Before the selfdev loop picks a task, log any analyzer suggestions so the operator
  can see recurring patterns in the terminal output without having to open the TUI.

  **Files to read first:**
  - `avoid_agent/__main__.py` — `_run_selfdev()` is where `run_loop()` is called; this is the right
    integration point for pre-loop logging
  - `avoid_agent/selfdev/loop.py` — use the `log()` function format for consistent timestamped output
  - `avoid_agent/learnings_analyzer.py` — call `analyze()` with the `.learnings/sessions/` path

  **File to modify:** `avoid_agent/__main__.py` in `_run_selfdev()`

  **What to add:** Before `run_loop()`, resolve the `.learnings/sessions/` path relative to `repo_root`,
  call `analyze()`, and if suggestions exist print each using the `log()` format with a `[suggestion]`
  prefix. If there are no suggestions, print nothing (don't add noise).
  Use a lazy local import to avoid circular dependencies.

  **Verify:**
  ```
  python -m avoid_agent selfdev --help
  python -c "from avoid_agent.__main__ import main; print('OK')"
  ```

## Phase 5: Remote and long-running operation

- [ ] Write `selfdev-status.json` after each selfdev cycle
  **Goal:** Enable external monitoring by writing a machine-readable status file after each run.

  **Files to read first:**
  - `avoid_agent/__main__.py` — `_run_selfdev()` calls `run_loop()` and uses `sys.exit(exit_code)`;
    write the status file just before the exit
  - `avoid_agent/selfdev/loop.py` — read `parse_backlog()` and `_gather_completed_tasks()` to
    understand how to count pending/completed/failed items from `backlog.md`

  **File to modify:** `avoid_agent/__main__.py` in `_run_selfdev()`

  **What to add:** After `run_loop()` returns `exit_code`, write `<repo_root>/selfdev-status.json`
  using `json.dump()` with `indent=2`. Fields:
  - `last_run`: ISO 8601 timestamp (`datetime.now().isoformat()`)
  - `exit_code`: the integer exit code
  - `completed_count`: count of `[x]` lines in `backlog.md`
  - `pending_count`: count of `[ ]` lines
  - `failed_count`: count of `[!]` lines

  Also add `selfdev-status.json` to `.gitignore` if that file exists and the entry isn't already there.

  **Verify:** Run `python -m avoid_agent selfdev --single` and confirm `selfdev-status.json` is written.

- [ ] Add webhook notification support on selfdev cycle completion
  **Goal:** Optionally POST a JSON payload to an external URL (Slack, custom endpoint, etc.)
  when the selfdev loop finishes a cycle, without adding any new pip dependencies.

  **Files to read first:**
  - `avoid_agent/__main__.py` — `_run_selfdev()` is the integration point; status writing from the
    previous task is already there to show the pattern
  - `.env.example` — understand how env vars are documented; add a commented example entry

  **Files to modify:**
  - `avoid_agent/__main__.py` — add a `_notify(url, payload)` helper using `urllib.request.urlopen`;
    call it in `_run_selfdev()` after writing status if `SELFDEV_WEBHOOK_URL` env var is set;
    catch all exceptions and log them (notification failures must not crash the process)
  - `.env.example` — add `# SELFDEV_WEBHOOK_URL=https://hooks.slack.com/...` as a commented example

  **Payload shape:** `{"event": "cycle_complete", "exit_code": <int>, "timestamp": "<iso>",
  "completed": <int>, "pending": <int>, "failed": <int>}`

  **Verify:** `python -m avoid_agent selfdev --help` works. No new entries in `requirements.txt`
  or `pyproject.toml` dependencies.

- [ ] Create a systemd unit file template for long-running selfdev
  **Goal:** Make it easy for users to run the selfdev loop as a background service that
  restarts automatically via `supervisor.sh`.

  **Files to read first:**
  - `supervisor.sh` — understand the start command, restart logic, and how the virtualenv is activated;
    the unit file's `ExecStart` should wrap this script
  - `.env.example` — note which env vars are required (`ANTHROPIC_API_KEY`, `DEFAULT_MODEL`)

  **File to create:** `avoid-agent-selfdev.service` at the repo root

  **What to include:**
  ```ini
  [Unit]
  Description=avoid-agent self-improvement loop
  After=network.target

  [Service]
  Type=simple
  WorkingDirectory=/path/to/avoid-agent   # user must set this
  ExecStart=/path/to/avoid-agent/supervisor.sh
  Restart=on-failure
  RestartSec=10
  StandardOutput=journal
  StandardError=journal
  EnvironmentFile=/path/to/avoid-agent/.env  # user must set this

  [Install]
  WantedBy=multi-user.target
  ```
  Add a header comment block explaining: this is a template, the user must set `WorkingDirectory`,
  `User`, and `EnvironmentFile` before installing; installation command:
  `sudo systemctl enable --now avoid-agent-selfdev`.

  **Verify:** The file parses as valid INI (no Python needed — just review it reads correctly).
