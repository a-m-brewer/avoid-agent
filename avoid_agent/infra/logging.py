"""Minimal logging setup.

This module provides a single ``setup_logging()`` call that configures the
standard library ``logging`` module with a sensible default format and level.
Import the logger for a module with::

    import logging
    log = logging.getLogger(__name__)

No-op until ``setup_logging()`` is called once at application startup.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: int = logging.WARNING) -> None:
    """Configure the root logger with a human-readable format.

    Call this exactly once, early in your entry point (``__main__.py``),
    before any other module configures logging.

    Args:
        level: Minimum log level to emit.  Defaults to WARNING so the
            terminal stays clean during normal operation.
            Use ``logging.INFO`` or ``logging.DEBUG`` for development.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
