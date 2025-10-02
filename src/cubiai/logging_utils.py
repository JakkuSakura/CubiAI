"""Logging helpers for CubiAI."""
from __future__ import annotations

import logging
from typing import Literal

from rich.logging import RichHandler


def configure_logging(level: str | Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO") -> None:
    """Configure root logging with Rich formatting."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
        force=True,
    )
