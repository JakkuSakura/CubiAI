"""`cubiai inspect` command implementation."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()


def inspect(workspace_path: Path = typer.Argument(..., help="Workspace directory to inspect.")) -> None:
    """Display metadata and diagnostics from a previous run."""
    metadata_path = workspace_path / "metadata.json"
    if not metadata_path.exists():
        raise typer.BadParameter(f"metadata.json not found under {workspace_path}")

    metadata = json.loads(metadata_path.read_text())

    console.print(f"[bold]Workspace:[/bold] {metadata.get('workspace')}")
    console.print(f"[bold]Profile:[/bold] {metadata.get('profile')}")
    console.print(f"[bold]Duration:[/bold] {metadata.get('duration_seconds', 0):.2f}s")

    stages = metadata.get("stages", [])
    stage_table = Table(title="Stages", show_header=True, header_style="bold green")
    stage_table.add_column("Stage")
    stage_table.add_column("Status")
    stage_table.add_column("Duration (s)")
    for stage in stages:
        stage_table.add_row(stage.get("name", "?"), stage.get("status", "?"), f"{stage.get('duration_seconds', 0):.2f}")
    console.print(stage_table)

    outputs = metadata.get("outputs", {})
    output_table = Table(title="Outputs", show_header=True, header_style="bold blue")
    output_table.add_column("Name")
    output_table.add_column("Path")
    for key, value in outputs.items():
        output_table.add_row(key, value)
    console.print(output_table)

    diagnostics_path = workspace_path / "diagnostics.json"
    if diagnostics_path.exists():
        diagnostics = json.loads(diagnostics_path.read_text())
        console.print("[bold]Diagnostics:[/bold]")
        console.print_json(data=diagnostics)
    else:
        console.print("[yellow]No diagnostics.json found.[/yellow]")
