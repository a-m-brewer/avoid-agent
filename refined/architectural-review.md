# Architectural Review: avoid-agent

**Date**: 2026-03-27
**Reviewer**: Senior Python Architect
**Scope**: Full codebase analysis

---

## High‑Level Overview

**avoid-agent** is a terminal-based AI coding assistant that drives a language model (Claude, GPT-5, Codex, etc.) through a structured tool‑call loop. The agent reads, writes, and edits files and runs shell commands, but it must **prove** every action via tool output before claiming success. The project has three distinct modes:

1. **Interactive TUI** — a terminal UI with a chat‑style conversation, inline permission prompts, a searchable model picker, and a spinning indicator.
2. **Headless** — a non‑interactive mode that accepts a prompt and returns a structured JSON result (designed to be scripted or called by the selfdev operator).
3. **Self‑improvement** — a supervisor loop that reads a `backlog.md`, creates git worktrees, runs the agent headless in each worktree to implement a backlog item, validates the result, and merges back to `main`.

The core "job" is: **give the model a task, let it use tools to accomplish it, and enforce verifiable evidence before the turn closes.** This is a sophisticated execution‑controller pattern layered on top of multiple LLM providers.

---

## User Workflows

1. **Interactive coding session** — User launches `avoid-agent`, picks a model, types a task, watches the agent loop through tool calls in the TUI, approves/denies shell commands, and reviews the final result.
2. **Headless automation** — A script or supervisor calls `avoid-agent headless --prompt "..."`, gets back a JSON object with `success`, `tool_calls`, and `tokens`.
3. **Self‑improvement loop** — The developer runs `avoid-agent selfdev`, the system picks the next unchecked backlog item, creates a worktree, runs the agent headless to implement it, validates against frozen‑file and command policies, and either merges or preserves the branch for review.

---

## Architecture Assessment

### God Files

| File | Lines | Problem |
|------|-------|---------|
| `avoid_agent/__main__.py` | 915 | A single monolithic `main()` function that handles all three modes (TUI, headless, selfdev) and all argument parsing. It wires together the provider, runtime, tools, permissions, session, TUI, selfdev loop, and output — everything mixed together. |
| `avoid_agent/selfdev/loop.py` | ~480 | `run_one_cycle()` is long and handles multiple responsibilities: backlog parsing, worktree creation, agent invocation, result parsing, diff checking, validation dispatch, merging, and error handling. `build_prompt_for_task()` is also a large pure‑function that assembles a complex prompt from multiple sources. |
| `avoid_agent/tui/__init__.py` | ~420 | The `TUI` class is doing too much: event handling, rendering, input state, spinner, permission prompts, model picker, conversation management, and key‑debug logging. The `run()` method and `_handle_key()` are each handling many concerns. |
| `avoid_agent/agent/runtime.py` | ~500 | `AgentRuntime.run_user_turn()` is the longest loop (while True with ~10 branches). `ExecutionController` is actually well‑structured; the bloat is in how the loop orchestrates retries and context preparation rather than in the controller itself. |
| `avoid_agent/providers/__init__.py` | ~450 | `get_provider()` is a long factory function with duplicated key‑loading logic for each provider. `normalize_messages()` has a complex state machine inside a loop. The file also contains all shared dataclasses (`Message`, `AssistantMessage`, `ToolResultMessage`, etc.) making it a catch‑all. |

### Coupling Issues

- `__main__.py` imports and directly instantiates nearly every module. If you rename a class or change a constructor signature, you must update `__main__.py` — and only `__main__.py`.
- `AgentRuntime` takes `request_permission: Callable` as a constructor argument and stores it. In `__main__.py`, the TUI's `ask_permission` method is passed as this callable, creating a circular dependency: `TUI → AgentRuntime → back to TUI` via the callback.
- `avoid_agent/selfdev/loop.py` directly calls `subprocess.run`, `subprocess.Popen`, `git worktree` commands, `yaml.safe_load`, and `Path` I/O — it is a single module that simultaneously acts as a CLI orchestrator, a git integration layer, and a YAML config reader.
- `avoid_agent/selfdev/operator.py` has functions that return a tuple `(cmd, env, repo_root)` and also have `stream_operator_to_tui` — the file mixes the "build command" concern with the "run and stream" concern.

### Global / Module‑Level State

- `_MODEL_CACHE` dict in `providers/__init__.py` is module‑level mutable state with a TTL. It survives across calls but is not explicit about its lifetime.
- `_SESSIONS_DIR`, `_cwd_key`, `_repo_dir` in `session.py` are module‑level constants that hard‑code `~/.avoid-agent/sessions`. If you ever want sessions per‑project or in a different location, you must change every call site.
- `_debug_keys` and `_debug_keys_path` in `TUI.__init__` are set once and capture process environment at construction time.

### Cross‑Cutting Concerns Scattered Everywhere

- **Logging**: `log()` is defined as a module‑level function in `selfdev/loop.py`. `stderr_thread` in `loop.py` and `TUI.__init__` each do their own key‑debug logging. No centralized `logging` module.
- **Config**: API keys are read via `os.getenv(...)` in dozens of places inside `providers/__init__.py`, `openai.py`, `anthropic.py`, `openai_codex.py`. There is no `config.py` or `settings.py` — every module invents its own pattern.
- **Error handling**: `except Exception` (bare) appears in `providers/__init__.py` (`_list_dynamic_models`), `tui/__init__.py` (`_log_key_debug`), `session.py` (`load_session`), and `loop.py` (`_load_frozen_patterns`). These silently swallow errors and return empty/fallback results, making debugging very difficult.

### Missing Layer Boundaries

- **CLI vs. domain logic**: `__main__.py` contains both the argument parser (`argparse` configuration) and the business logic for every mode. The "main" function should be a thin dispatcher; the mode implementations belong elsewhere.
- **UI vs. agent logic**: The `TUI` class creates the `AgentRuntime` and owns the permission callback. These responsibilities should be separated: the TUI should be a pure rendering/input layer, and an orchestration layer should sit between TUI and runtime.
- **Selfdev vs. core**: `selfdev/` is a top‑level package with `loop.py`, `operator.py`, `validate.py`, and `__init__.py`. These are deeply entangled with core concepts (provider, tools, session), but the selfdev layer should be "higher up" in an architectural sense — it *uses* the core, it shouldn't be co‑located with it.

---

## Proposed Architecture

### Core Principle
The project has two distinct "products" running on the same code base:
1. **Agent runtime** — the tool‑call loop, providers, tools, permissions, session management.
2. **Selfdev supervisor** — a workflow that uses (1) to improve itself.

These should be separated so that (1) is a clean, testable library and (2) is a thin orchestration layer on top.

### Recommended Directory Tree

```
avoid_agent/
├── __main__.py              # Thin CLI entry point (parse args → dispatch)
├── cli/
│   ├── __init__.py
│   ├── tui.py               # TUI mode: builds and runs the TUI loop
│   ├── headless.py          # Headless mode: builds and runs headless session
│   └── selfdev.py           # Selfdev mode: runs the supervisor loop
│
├── agent/
│   ├── __init__.py           # Exports AgentRuntime, ExecutionController
│   ├── runtime.py           # AgentRuntime, ExecutionController (as-is, already clean)
│   ├── context.py           # ContextStrategy, prepare_context (already separate)
│   └── tools.py             # ToolDefinition, run_tool (already separate)
│
├── providers/
│   ├── __init__.py           # Provider ABC, shared dataclasses, get_provider factory
│   ├── base.py               # Provider, ProviderStream, ProviderEvent, Message types
│   ├── anthropic.py
│   ├── openai.py
│   ├── openai_codex.py
│   ├── openai_codex_oauth.py
│   └── anthropic_oauth.py
│
├── infra/                    # Cross-cutting infrastructure
│   ├── __init__.py
│   ├── config.py            # Centralized config loading (API keys, model prefs, paths)
│   ├── logging.py           # Centralized logging setup
│   ├── session.py           # (already exists, move here)
│   └── permissions.py        # command_prefix, PermissionsStore (already exists)
│
├── selfdev/
│   ├── __init__.py           # RESTART_EXIT_CODE, entry point (selfdev command)
│   ├── loop.py              # Refactored: split into workflow/steps/
│   ├── operator.py          # Refactored: split command-building from execution
│   ├── validate.py          # (already clean, keep)
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── backlog.py       # BacklogItem, parse_backlog, mark_backlog_item
│   │   ├── worktree.py      # create/cleanup/detach/merge worktree
│   │   ├── prompt_builder.py # build_prompt_for_task, _gather_* helpers
│   │   └── runner.py        # run_agent_headless, _stream_stderr
│   └── prompts/
│       └── operator_prompt.py  # build_operator_prompt
│
├── tui/
│   ├── __init__.py          # TUI class (refactor: extract input/rendering/perms)
│   ├── components/          # (already has these)
│   ├── rendering.py         # Renderer, physical_rows calculation (extract from __init__)
│   ├── input.py             # Input state, history, key handling (extract from __init__)
│   └── events.py            # PermissionItem, model picker (extract from __init__)
│
└── learnings/
    ├── __init__.py
    ├── capture.py           # LearningsCapture, capture_learnings
    ├── analyzer.py          # LearningsAnalyzer (already exists)
    └── storage.py           # .learnings/ directory I/O
```

### Module Responsibilities

- **`cli/`** — Thin, stateless dispatchers. Each `*.py` file owns one CLI command (`tui`, `headless`, `selfdev`). No business logic lives here.
- **`agent/`** — Pure domain logic: given a provider + tools + messages, produce a result. No I/O, no CLI, no config.
- **`providers/`** — Each provider file is an adapter for one external API. `__init__.py` holds the factory and shared types.
- **`infra/`** — Application‑wide concerns: where config lives, how sessions are stored, how to log. Everything here is importable but not business‑logic‑heavy.
- **`selfdev/workflow/`** — Each module in `workflow/` is a single‑responsibility step: parse the backlog, manage a worktree, build a prompt, run the headless agent. `loop.py` becomes a simple orchestrator that calls these steps.
- **`tui/`** — Split into `rendering.py`, `input.py`, and `events.py` so the `TUI` class itself is small and delegates to collaborators.

---

## Refactor Plan

### Phase 1: Extract Infrastructure (High Impact, Low Risk)

1. **Create `avoid_agent/infra/config.py`** — Pull all `os.getenv(...)` calls for API keys, base URLs, token budgets, and session paths into a single `AppConfig` dataclass with typed fields. Every module that needs a key imports from here. This is a pure refactor — no behavior change — but it eliminates the "where do I put my key?" guesswork.

2. **Create `avoid_agent/infra/logging.py`** — Replace all `print(...)` and inline `log()` functions with a centralized `get_logger(name)` that wraps `logging.getLogger`. Add structured fields (timestamp, component) so logs are greppable. The `log()` in `selfdev/loop.py` becomes `get_logger("selfdev").info(...)`.

3. **Move `session.py` to `infra/session.py`** — The module is already well‑structured. Move it; update imports in `__main__.py` and anywhere else that imports it.

**Why first**: These changes touch every module but are mechanically simple (add an import, replace a function call). They immediately make the codebase easier to test and debug.

### Phase 2: Extract CLI Dispatch (Medium Impact, Low Risk)

4. **Create `avoid_agent/cli/tui.py`**, **`cli/headless.py`**, **`cli/selfdev.py`** — Extract the body of each `if/elif` branch from `main()` into a separate module. Each module exposes a `run(args: argparse.Namespace) -> int` function. `__main__.py` becomes ~40 lines: parse args, import the right `cli.*`, call `run()`.

5. **Extract TUI's collaborators** — In `tui/__init__.py`, extract `_render()`, `_spin_loop()`, `ask_permission()`, and `pick_from_list()` into `tui/rendering.py`, `tui/spinner.py`, `tui/events.py` respectively. The `TUI` class becomes a thin composition of these.

**Why here**: The CLI dispatch refactor is mechanically safe because you can run `python -m avoid_agent tui` before and after and compare output. No logic changes, only file moves.

### Phase 3: Break Up `selfdev/loop.py` (Medium Impact, Low-Medium Risk)

6. **Create `selfdev/workflow/backlog.py`** — Move `parse_backlog`, `mark_backlog_item`, and the `BacklogItem` dataclass. Add type hints and a docstring.

7. **Create `selfdev/workflow/worktree.py`** — Move `create_worktree`, `cleanup_worktree`, `detach_worktree`, `merge_worktree`, `_branch_exists`. Add a `WorktreeManager` class that encapsulates the git subprocess calls.

8. **Create `selfdev/workflow/prompt_builder.py`** — Move `build_prompt_for_task` and all `_gather_*` helper functions. Break `_gather_*` into smaller pure functions with clear return types.

9. **Create `selfdev/workflow/runner.py`** — Move `run_agent_headless` and `_stream_stderr`. This becomes the interface between selfdev and the headless CLI.

10. **Simplify `selfdev/loop.py`** — After the above extractions, `run_one_cycle()` should be ~60 lines: parse backlog → create worktree → build prompt → run agent → check result → validate → merge/detach. Each step calls a collaborator.

11. **Refactor `selfdev/operator.py`** — Split `run_operator()` (returns `cmd, env, repo_root`) from `run_operator_headless()` and `stream_operator_to_tui()`. The first belongs in a "command builder" module; the latter two belong in `selfdev/workflow/runner.py` alongside the worker runner.

**Why here**: These functions are already well‑named and loosely coupled. The risk is low because each function has a clear input/output contract. The biggest risk is the `git worktree` calls — guard them with integration tests that call `git status`.

### Phase 4: Clean Up `providers/__init__.py` (Medium Impact, Medium Risk)

12. **Create `providers/base.py`** — Move all dataclass definitions (`Message`, `UserMessage`, `AssistantMessage`, `ToolResultMessage`, `ProviderEvent`, `ProviderToolCall`, `Usage`, etc.) into `base.py`. Export them from `__init__.py` for backward compatibility during the transition.

13. **Refactor `get_provider()`** — Extract each provider's instantiation into a helper function (`_make_openai_provider`, `_make_anthropic_provider`, etc.) inside `__init__.py`. The main factory becomes a simple dispatch table. API key loading moves to `infra/config.py`.

14. **Refactor `normalize_messages()`** — Extract the inner `flush_orphaned_tool_calls()` helper to module level and add type annotations. The logic is correct but the nested function makes it harder to test.

**Why here**: Changing shared dataclasses is high‑risk because every module that imports from `providers` could break. Do this after Phase 1 (centralized config makes import paths stable) and use `grep -r "from avoid_agent.providers import"` before and after to catch any missed updates.

### Phase 5: Long-term Cleanup

15. **Remove bare `except Exception`** — Replace each one with specific exception types. At minimum, log the error in `infra/logging.py` so failures are visible. Silent swallowing makes production debugging nearly impossible.
16. **Add a smoke test suite** — A small script that imports `agent.runtime`, `providers`, `session`, and `selfdev` and calls the main public functions with mock data. This gives you a safety net for all the above refactors.
17. **Type the TUI and selfdev modules** — These are the least‑typed modules and would benefit most from `mypy` checking.

### `main()` Function Breakdown (before refactoring → after)

**Current `main()` (~915 lines)** handles TUI mode, headless mode, selfdev mode, and argument parsing all in one function. The breakdown after Phase 2 looks like:

```
main(args)                          # ~40 lines
├── _parse_args()                   # argparse config
├── if args.command == "tui":
│   └── cli.tui.run(args)           # delegates to tui.py
├── elif args.command == "headless":
│   └── cli.headless.run(args)      # delegates to headless.py
├── elif args.command == "selfdev":
│   └── cli.selfdev.run(args)       # delegates to selfdev.py
└── return exit_code
```

Each `cli/*.py` module owns its mode:

```
# cli/tui.py
def run(args) -> int:
    config = infra.config.load()
    provider = providers.get_provider(...)
    runtime = agent.AgentRuntime(...)
    tui = TUI(
        on_submit=lambda text: runtime.run_user_turn(...),
        ...
    )
    tui.run()
    return 0
```

```
# cli/selfdev.py
def run(args) -> int:
    infra.logging.configure("selfdev")
    repo_root = Path.cwd()
    from avoid_agent.selfdev.loop import run_loop
    return run_loop(repo_root, model=args.model, ...)
```

---

## Examples / Snippets

### Centralized Config (`infra/config.py`)

```python
"""Centralized application configuration — all API keys and paths in one place."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Lazy imports for optional dependencies (yaml, etc.) are fine here.


@dataclass(frozen=True)
class AppConfig:
    """Global application configuration. All fields are immutable at runtime."""
    anthropic_api_key: str | None
    openai_api_key: str | None
    openrouter_api_key: str | None
    ollama_host: str
    default_model: str | None
    max_tokens: int
    token_budget: int
    session_dir: Path

    @classmethod
    def from_env(cls) -> AppConfig:
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
            default_model=os.getenv("DEFAULT_MODEL"),
            max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
            token_budget=int(os.getenv("TOKEN_BUDGET", "40000")),
            session_dir=Path.home() / ".avoid-agent" / "sessions",
        )
```

### Well‑structured CLI Dispatch (`__main__.py`)

```python
"""avoid_agent/__main__.py — thin CLI entry point."""
import argparse
import sys
from pathlib import Path

from avoid_agent import infra


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avoid-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    # TUI subcommand
    tui = sub.add_parser("tui", help="Interactive terminal UI")
    tui.add_argument("--model", help="Model in provider/name format")
    tui.add_argument("--session", help="Session name to load/restore")

    # Headless subcommand
    headless = sub.add_parser("headless", help="Non-interactive headless mode")
    headless.add_argument("--prompt", required=True)
    headless.add_argument("--model")
    headless.add_argument("--auto-approve", action="store_true")
    headless.add_argument("--no-session", action="store_true")
    headless.add_argument("--max-turns", type=int, default=40)

    # Selfdev subcommand
    sd = sub.add_parser("selfdev", help="Self-improvement loop")
    sd.add_argument("--model")
    sd.add_argument("--max-turns", type=int, default=40)
    sd.add_argument("--single", action="store_true")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "tui":
        from avoid_agent.cli import tui as tui_module
        return tui_module.run(args)
    if args.command == "headless":
        from avoid_agent.cli import headless as headless_module
        return headless_module.run(args)
    if args.command == "selfdev":
        from avoid_agent.cli import selfdev as selfdev_module
        return selfdev_module.run(args)
    return 1  # unreachable


if __name__ == "__main__":
    sys.exit(main())
```

### Refactored `selfdev/workflow/backlog.py`

```python
"""Backlog parsing and mutation for the selfdev workflow."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacklogItem:
    line_number: int
    text: str
    raw_line: str


@dataclass(frozen=True)
class ParseResult:
    items: list[BacklogItem]
    path: Path


def parse_backlog(repo_root: Path) -> list[BacklogItem]:
    """Return unchecked backlog items in order, or an empty list if no backlog."""
    backlog_path = repo_root / "backlog.md"
    if not backlog_path.exists():
        return []

    items: list[BacklogItem] = []
    for line_number, line in enumerate(backlog_path.read_text().splitlines(), start=1):
        match = re.match(r"^- \[ \] (.+)$", line.strip())
        if match:
            items.append(BacklogItem(
                line_number=line_number,
                text=match.group(1).strip(),
                raw_line=line,
            ))
    return items


def mark_item(repo_root: Path, item: BacklogItem, status: str, note: str = "") -> None:
    """Mark a backlog item. status must be 'done' or 'failed'."""
    backlog_path = repo_root / "backlog.md"
    lines = backlog_path.read_text().splitlines(keepends=True)
    marker = "[x]" if status == "done" else "[!]"
    idx = item.line_number - 1
    if 0 <= idx < len(lines):
        lines[idx] = lines[idx].replace("[ ]", marker, 1)
        if note and status == "failed":
            lines[idx] = lines[idx].rstrip() + f" <!-- {note} -->\n"
    backlog_path.write_text("".join(lines))
```

### Refactored `TUI._handle_key()` (illustrating extraction)

```python
# tui/input.py — extracted from TUI.__init__
class InputHandler:
    def __init__(self, line: EditableLine, history: History):
        self._line = line
        self._history = history
        self._in_paste = False

    def handle_key(self, key: str, data: bytes) -> KeyResult:
        """Returns a KeyResult indicating what happened."""
        if key == "enter" and not self._in_paste:
            text = self._line.clear()
            self._history.push(text)
            return KeyResult.submit(text)
        # ... other key handling ...
        return KeyResult.consumed()


@dataclass(frozen=True)
class KeyResult:
    kind: Literal["submit", "consumed", "exit"]
    text: str | None = None
```

---

## Risks and Checks

### Risks

1. **Import graph breakage** — Moving modules (`session.py` → `infra/session.py`) will break every import. **Guard**: Before the move, run `grep -rn "from avoid_agent.session import\|import avoid_agent.session"` and update every path in one commit.

2. **Dataclass field changes** — If `AssistantMessage` or `ToolResultMessage` fields change, session load/restore will silently produce wrong objects. **Guard**: Add a schema version to the session JSON and validate it on load. The existing `_SESSION_VERSION = 2` is good; make sure it increments when the schema changes.

3. **Git worktree side effects** — The `subprocess.run(["git", "worktree", ...])` calls in `selfdev/workflow/worktree.py` modify the host repo's `.git/worktrees` database. A test that creates worktrees will pollute the real repo. **Guard**: Run worktree tests against a bare clone or a temp directory, not against the live repo.

4. **`os.getenv()` scattered across providers** — Until Phase 1 is complete, changing a key name requires updating multiple provider files. **Guard**: Use the centralized config from day one; don't add new `os.getenv` calls anywhere.

5. **Bare `except Exception`** swallowing real errors — In production, `_list_dynamic_models()` silently fails when an API is unreachable, making model discovery appear to work but return stale results. **Guard**: Replace with specific exceptions and log at `WARNING` level.

### Regression Checks

- **Smoke test**: `python -m avoid_agent tui --help` and `python -m avoid_agent headless --help` should print help without importing any provider or agent code (defer those imports until after arg parsing).
- **Session round‑trip**: Load a session, append a message, save it, reload it, verify the deserialized message is structurally equal to the original.
- **Selfdev validation**: Run `python -m avoid_agent selfdev --single --max-turns 1` against a clean backlog item and verify it either creates a worktree or exits cleanly — don't validate the agent output, just validate that the workflow runs without crashing.
- **Headless output**: `python -m avoid_agent headless --prompt "echo hello"` should output exactly one JSON object to stdout (no extra log lines).
