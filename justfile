python := "python3"
venv := ".venv/bin/python"
pip := ".venv/bin/pip"

# Create a local virtualenv, install runtime + dev deps from pyproject.toml, and
# seed .env for first-time contributors without overwriting existing config.
setup:
    {{python}} -m venv .venv
    {{pip}} install --upgrade pip setuptools wheel
    {{pip}} install -e '.[dev]'
    if [ ! -f .env ] && [ -f .env.example ]; then cp .env.example .env; fi

# Run tests (all by default, or pass a path/filter e.g. just test tests/agent/tools/test_inspector.py)
test *args:
    {{venv}} -m pytest -v {{args}}

run:
    {{venv}} -m avoid_agent

# Export the assembled system prompt as markdown for review.
# Usage: just prompt-export [output_path]
prompt-export out="./system-prompt.md":
    {{venv}} -m avoid_agent prompt export --out {{out}}
