"""CLI module for avoid-agent.

This module provides thin, stateless entry points for each CLI command.
Each subcommand owns its mode: tui, headless, selfdev.
"""

from avoid_agent.cli import tui, headless, selfdev

__all__ = ["tui", "headless", "selfdev"]
