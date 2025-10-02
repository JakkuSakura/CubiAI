"""`cubiai models` command group."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ...config.loader import load_profile

console = Console()
models = typer.Typer(help="Model asset management utilities.")


@models.command("sync")
def sync(
    profile: str = typer.Option(
        "anime-default", "--profile", "-p", help="Profile used to determine required models."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report missing models without downloading them."
    ),
) -> None:
    """Verify local model assets for a given profile."""
    profile_cfg = load_profile(profile_name=profile)

    required_assets = profile_cfg.models.assets
    if not required_assets:
        console.print("[green]Profile does not declare external model assets.[/green]")
        return

    missing: list[str] = []
    for asset in required_assets:
        path = Path(asset.path).expanduser()
        if not path.exists():
            missing.append(str(path))

    if not missing:
        console.print("[bold green]All model assets are available locally.[/bold green]")
        return

    console.print("[yellow]Missing model assets:[/yellow]")
    for item in missing:
        console.print(f"  - {item}")

    if dry_run:
        console.print("Dry run enabled; no downloads attempted.")
        return

    console.print(
        "[red]Automatic downloads are not implemented yet. Download the assets manually and rerun this command.[/red]"
    )
