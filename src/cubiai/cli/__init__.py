"""Command-line interface bootstrap for CubiAI."""
from __future__ import annotations

import typer

from .commands.inspect import inspect
from .commands.model import app as model
from .commands.process import process


app = typer.Typer(help="AI-assisted Live2D asset generator")

app.command()(process)
app.command()(inspect)
app.add_typer(model, name="model")

__all__ = ["app"]
