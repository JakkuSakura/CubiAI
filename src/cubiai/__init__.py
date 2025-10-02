"""CubiAI package entry point and metadata."""
from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("cubiai")
except metadata.PackageNotFoundError:  # pragma: no cover - during local development
    __version__ = "0.1.0"

__all__ = ["__version__"]
