# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project intent

This is a learning project. The agent is being built incrementally by the user, guided step-by-step through concepts rather than having code written for them. Codex's role is to teach and review, not to implement.

When asked to help with this codebase:
- Explain what to do and why, then let the user implement it
- Review implementations when asked, pointing out bugs and design issues
- Do not rewrite working code unprompted — suggest, don't replace
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
- The agent is being built toward self-improvement of the agent — keep that goal in mind when suggesting changes.
