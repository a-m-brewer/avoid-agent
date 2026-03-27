"""Centralized, lazily-evaluated access to environment variables and derived config.

All modules that need env vars should import *from* this module rather than
calling ``os.getenv`` directly.  This makes it easy to:
- See every runtime knob in one place
- Override values in tests without patching ``os.environ``
- Add validation / type coercion in one spot
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Callable


# ---------------------------------------------------------------------------
# Provider keys
# ---------------------------------------------------------------------------
def _key(name: str) -> str | None:
    return os.getenv(name)


ANTHROPIC_API_KEY: str | None = _key("ANTHROPIC_API_KEY")
OPENAI_API_KEY: str | None = _key("OPENAI_API_KEY")
OPENROUTER_API_KEY: str | None = _key("OPENROUTER_API_KEY")
ZAI_API_KEY: str | None = _key("ZAI_API_KEY")
LMSTUDIO_API_KEY: str | None = _key("LMSTUDIO_API_KEY")

# ---------------------------------------------------------------------------
# Provider base URLs
# ---------------------------------------------------------------------------
OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
LMSTUDIO_BASE_URL: str = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1").rstrip("/")
ZAI_BASE_URL: str = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4").rstrip("/")

# ---------------------------------------------------------------------------
# Agent defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4-6")
MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "8192"))
CONTEXT_STRATEGY: str = os.getenv("CONTEXT_STRATEGY", "compact+window")

# ---------------------------------------------------------------------------
# Session / persistence
# ---------------------------------------------------------------------------
SESSIONS_DIR: Path = Path.home() / ".avoid-agent" / "sessions"

# ---------------------------------------------------------------------------
# Selfdev
# ---------------------------------------------------------------------------
SELFDEV_INCLUDE_LEARNINGS: bool = os.getenv(
    "SELFDEV_INCLUDE_LEARNINGS", ""
).lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# TUI / debugging
# ---------------------------------------------------------------------------
AVOID_AGENT_DEBUG_KEYS: bool = (
    os.getenv("AVOID_AGENT_DEBUG_KEYS", "").strip().lower() in ("1", "true", "yes", "on")
)
AVOID_AGENT_DEBUG_KEYS_PATH: str = os.getenv(
    "AVOID_AGENT_DEBUG_KEYS_PATH", "/tmp/avoid-agent-keys.log"
)

# ---------------------------------------------------------------------------
# Provider: Codex debug
# ---------------------------------------------------------------------------
DEBUG_CODEX_EVENTS: str | None = os.getenv("DEBUG_CODEX_EVENTS")

# ---------------------------------------------------------------------------
# Public helpers used across the codebase
# ---------------------------------------------------------------------------


def env_flag(name: str, default: bool = False) -> bool:
    """Return True if ``name`` is set to a truthy value, otherwise ``default``."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    """Parse ``name`` as an int, returning ``default`` on failure."""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


# Simple dependency-injection container so callers can replace implementations
# at test time without patching globals.
class _Config:
    """Mutable config store for swappable implementations (mainly for tests)."""

    def __init__(self) -> None:
        self._getenv: Callable[[str], str | None] = os.getenv
        self._sessions_dir: Path = SESSIONS_DIR
        self._time: Callable[[], float] = time.time

    def getenv(self, name: str, default: str | None = None) -> str | None:
        return self._getenv(name, default)

    @property
    def sessions_dir(self) -> Path:
        return self._sessions_dir

    @sessions_dir.setter
    def sessions_dir(self, value: Path) -> None:
        self._sessions_dir = value

    def time(self) -> float:
        return self._time()


config = _Config()
