"""Command-line interface bootstrap for CubiAI."""
from __future__ import annotations

import typer

from .commands.inspect import inspect
from .commands.models import models
from .commands.process import process
from .commands.profiles import profiles
from .commands.workspace import workspace


app = typer.Typer(help="AI-assisted Live2D asset generator")

app.command()(process)
app.command()(inspect)
app.add_typer(models, name="models")
app.add_typer(profiles, name="profiles")
app.add_typer(workspace, name="workspace")

__all__ = ["app"]
