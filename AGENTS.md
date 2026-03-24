# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project intent

This is now a more traditional software project. Agents should actively help implement features, fix bugs, refactor code, and improve the developer experience when asked.

When asked to help with this codebase:
- Implement requested changes directly unless the user asks for guidance-only help
- Explain what you are doing and why as you work, especially when making non-obvious design decisions
- Prefer small, comprehensible changes over clever abstractions
- Review implementations when asked, pointing out bugs, design issues, and testing gaps
- Do not rewrite stable working code without a clear reason
- Keep the codebase minimal. This project follows the pi-mono 4-tool philosophy: resist adding tools or abstractions unless clearly necessary

## Running the agent

```bash
source .venv/bin/activate
python -m avoid_agent
```

## Environment

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. Optional env vars: `DEFAULT_MODEL`, `MAX_TOKENS`.

## Testing the agent with Claude Code / Codex (headless supervision)

Claude Code / Codex acts as a supervisor that drives the avoid-agent via headless mode to implement features, while observing its behavior for harness-level issues.

### Workflow

1. **Pick a small feature** for the agent to implement (e.g. a new slash command, a UI improvement).
2. **Craft a prompt** and run the agent via headless mode:
   ```bash
   python -m avoid_agent headless --prompt '...' --auto-approve --no-session --model openai-codex/gpt-5.3-codex 2>/tmp/agent_events.jsonl
   ```
3. **Analyze the output** — review tool calls, reasoning, and the result JSON for behavioral issues.
4. **Fix only harness bugs** — if the agent misbehaves (wrong tool choice, hallucinated output, ID replay failures, bad structured actions, etc.), fix the agent runtime, system prompt, or execution controller.
5. **Re-run the agent** on the same task to verify the fix helped.

### What Claude Code / Codex fixes vs. what the agent fixes

| Claude Code / Codex fixes (harness/runtime) | Agent fixes (feature code) |
|--------------------------------------|----------------------------|
| System prompt guidelines | Feature implementation bugs |
| Tool execution / validation | Wrong API headers or params |
| Provider bugs (e.g. store=false ID replay) | Missing imports or wiring |
| Structured action parsing | Test failures in feature code |
| Context management / session issues | Business logic errors |

**Claude Code / Codex must NOT fix feature-level bugs directly.** If the agent produces buggy feature code, that's signal about what the harness or prompt needs to improve — send a follow-up prompt to the agent instead.

### Event monitoring

- Stderr (`2>/tmp/agent_events.jsonl`) contains JSONL events for every tool call, text delta, permission decision, and turn boundary.
- Stdout contains the final result JSON with success status, tool call log, and assistant text.
- Check for: `run_bash` used for file reads/writes (should use `read_file`/`write_file`/`edit_file`), hallucinated tool results, excessive turns, validation errors.

## Design philosophy

- 4 tools only: `read_file`, `write_file`, `edit_file`, `run_bash`. Bash is the escape hatch for everything else.
- The agent is still being built toward self-improvement of the agent — keep that goal in mind when suggesting changes.
- Preserve a teaching mindset in communication: explain the intent of changes, important tradeoffs, and how the pieces fit together, even when implementing directly.
