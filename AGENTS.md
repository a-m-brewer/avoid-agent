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

## Design philosophy

- 4 tools only: `read_file`, `write_file`, `edit_file`, `run_bash`. Bash is the escape hatch for everything else.
- The agent is still being built toward self-improvement of the agent — keep that goal in mind when suggesting changes.
- Preserve a teaching mindset in communication: explain the intent of changes, important tradeoffs, and how the pieces fit together, even when implementing directly.
