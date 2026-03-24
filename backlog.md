# Backlog

Items are worked top-to-bottom. The self-dev loop picks the first unchecked item.
Mark items `[x]` when done, `[!]` if failed (with a note).

## Phase 1: Skills system (safe, no restart needed)

- [ ] Create `skills/` directory with auto-discovery in the system prompt builder
  - Scan `skills/` and `~/.avoid-agent/skills/` for `SKILL.md` files
  - Load only `name` and `description` from frontmatter into system prompt
  - Load full skill body on-demand when the agent references a skill by name
  - Update `avoid_agent/prompts/system_prompt.py` to include discovered skill summaries

- [ ] Create a sample built-in skill `skills/self-review/SKILL.md`
  - A skill that instructs the agent how to review its own recent changes
  - Should reference `git log`, `git diff`, and reading modified files

## Phase 2: Extensions system (tools via worktree + validation)

- [ ] Create `extensions/` directory with auto-discovery in `find_available_tools()`
  - Scan `extensions/` for Python modules containing `@tool`-decorated functions
  - Merge discovered extension tools with built-in tools
  - Ensure extensions can be added without modifying core tool code

- [ ] Create a sample extension `extensions/file_search/__init__.py`
  - A `file_search` tool that uses grep/find to search across the codebase
  - Demonstrates the extension pattern for future self-created tools

## Phase 3: Learnings system (error capture + backlog feeding)

- [ ] Create `.learnings/` directory and capture module
  - After each session (TUI or headless), log errors and failed tool calls
  - Format: timestamped markdown files in `.learnings/sessions/`
  - Include: error messages, user corrections, failed validation attempts

- [ ] Create a learnings analyzer that can propose backlog items
  - Read `.learnings/sessions/*.md` and identify recurring patterns
  - Group by: tool errors, provider errors, user friction, missing capabilities
  - Output proposed backlog items that can be appended to `backlog.md`

## Phase 4: Autonomous improvements

- [ ] Add `/self-improve` slash command to TUI mode
  - Runs one self-improvement cycle interactively
  - Shows the user what task was picked, what changes were made, validation results
  - User can approve or reject before merge

- [ ] Improve selfdev loop with better task classification
  - Classify backlog items as Tier 1 (skill), Tier 2 (extension), or Tier 3 (core)
  - Tier 1 items skip worktree and apply directly
  - Tier 2/3 use the full worktree + validate + merge flow

- [ ] Add learnings-to-backlog automation in selfdev loop
  - Before picking a task, check `.learnings/` for new entries
  - Auto-propose and append new backlog items from learnings

## Phase 5: Remote and long-running operation

- [ ] Add status reporting to selfdev loop
  - Write `selfdev-status.json` with: current task, last completed, failures, uptime
  - Enables monitoring the agent remotely

- [ ] Add webhook/notification support
  - Notify on task completion or failure (Slack, email, or simple HTTP POST)
  - Configurable via `.env` or config file

- [ ] Harden supervisor for remote operation
  - Log rotation for selfdev output
  - systemd unit file for running as a service
  - Health check endpoint
