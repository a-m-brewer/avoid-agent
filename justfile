venv := ".venv/bin/python"

# Run tests (all by default, or pass a path/filter e.g. just test tests/agent/tools/test_inspector.py)
test *args:
    {{venv}} -m pytest -v {{args}}
