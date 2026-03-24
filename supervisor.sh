#!/bin/bash
# supervisor.sh - Restart loop for avoid-agent self-improvement.
#
# The agent exits with code 42 when it has merged improvements to main
# and wants to restart with the new code. Any other exit stops the loop.
#
# This file is FROZEN - the agent must never modify it.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

RESTART_CODE=42
MAX_CONSECUTIVE_FAILURES=3
failure_count=0

echo "[supervisor] Starting avoid-agent self-improvement loop"
echo "[supervisor] Repo: $REPO_DIR"
echo "[supervisor] Restart exit code: $RESTART_CODE"
echo "[supervisor] Max consecutive failures: $MAX_CONSECUTIVE_FAILURES"
echo ""

while true; do
    git checkout main 2>/dev/null || true
    git pull --ff-only origin main 2>/dev/null || true

    echo "[supervisor] Launching: python -m avoid_agent selfdev"
    echo "---"

    source .venv/bin/activate 2>/dev/null || true
    python -m avoid_agent selfdev
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq $RESTART_CODE ]; then
        failure_count=0
        echo ""
        echo "[supervisor] Agent requested restart (exit $RESTART_CODE). Restarting..."
        echo ""
        continue
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "[supervisor] Agent exited cleanly (exit 0). Stopping."
        break
    fi

    failure_count=$((failure_count + 1))
    echo ""
    echo "[supervisor] Agent crashed (exit $EXIT_CODE). Failure $failure_count/$MAX_CONSECUTIVE_FAILURES"

    if [ $failure_count -ge $MAX_CONSECUTIVE_FAILURES ]; then
        echo "[supervisor] Too many consecutive failures. Stopping."
        exit 1
    fi

    echo "[supervisor] Retrying in 5 seconds..."
    sleep 5
done
