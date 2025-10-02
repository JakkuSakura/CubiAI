"""`cubiai workspace` command group."""
from __future__ import annotations

import shutil
import time
from datetime import timedelta
from pathlib import Path

import typer
from rich.console import Console

console = Console()
workspace = typer.Typer(help="Workspace housekeeping utilities.")


def _parse_age(age: str) -> timedelta:
    units = {"h": 3600, "d": 86400, "w": 604800}
    try:
        scalar = float(age[:-1])
        suffix = age[-1]
        seconds = scalar * units[suffix]
        return timedelta(seconds=seconds)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter("Age must match <number><h|d|w>, e.g. 48h or 14d") from exc


@workspace.command("clean")
def clean(
    root: Path = typer.Option(Path("build"), "--root", help="Root directory containing workspaces."),
    older_than: str = typer.Option(
        "14d", "--older-than", help="Delete workspaces older than the given horizon (e.g. 48h, 14d)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show deletions without removing anything."),
) -> None:
    """Delete workspace directories older than the given threshold."""
    horizon = _parse_age(older_than)
    cutoff = time.time() - horizon.total_seconds()

    if not root.exists():
        console.print(f"[yellow]Workspace root {root} does not exist.[/yellow]")
        return

    removed = 0
    for path in sorted(root.glob("**/metadata.json")):
        workspace_dir = path.parent
        if workspace_dir.stat().st_mtime > cutoff:
            continue
        console.print(f"[yellow]Removing {workspace_dir}[/yellow]")
        removed += 1
        if not dry_run:
            shutil.rmtree(workspace_dir, ignore_errors=True)

    if removed == 0:
        console.print("[green]No workspaces matched the criteria.[/green]")
