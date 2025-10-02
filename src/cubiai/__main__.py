"""Command-line entry point for CubiAI."""
from __future__ import annotations

from .cli import app


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
