"""`cubiai profiles` command group."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from ...config.loader import discover_profiles, load_profile

console = Console()
profiles = typer.Typer(help="Profile discovery and management.")


@profiles.command("list")
def list_profiles() -> None:
    """List available processing profiles."""
    table = Table(title="Available Profiles", show_header=True, header_style="bold magenta")
    table.add_column("Name")
    table.add_column("Description")
    for profile_path in discover_profiles():
        profile_cfg = load_profile(profile_path.stem)
        table.add_row(profile_cfg.name, profile_cfg.description)
    console.print(table)


@profiles.command("show")
def show_profile(
    profile: str = typer.Argument(..., help="Profile name to display."),
) -> None:
    """Dump the resolved profile configuration as JSON."""
    profile_cfg = load_profile(profile_name=profile)
    console.print_json(data=profile_cfg.model_dump(mode="json"))
