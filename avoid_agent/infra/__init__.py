"""Infrastructure layer — cross-cutting concerns that don't belong in domain logic."""

from avoid_agent.infra.config import config
from avoid_agent.infra.logging import setup_logging

__all__ = ["config", "setup_logging"]
