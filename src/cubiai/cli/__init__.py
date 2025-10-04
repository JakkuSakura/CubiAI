"""Command-line interface bootstrap for CubiAI."""
from __future__ import annotations

import typer

from .commands.train import app as train
from .commands.annotate import annotate
from .commands.inspect import inspect
from .commands.preview import preview
from .commands.split import split
from .commands.models import models
from .commands.process import process
from .commands.workspace import workspace


app = typer.Typer(help="AI-assisted Live2D asset generator")

app.command()(process)
app.command()(annotate)
app.command()(preview)
app.command()(split)
app.command()(inspect)
app.add_typer(models, name="models")
app.add_typer(workspace, name="workspace")
app.add_typer(train, name="train")

__all__ = ["app"]
