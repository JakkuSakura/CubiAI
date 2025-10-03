"""`cubiai models` command group."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ...config.loader import DEFAULT_CONFIG_PATH, load_config

console = Console()
models = typer.Typer(help="Model asset management utilities.")


@models.command("sync")
def sync(
    config: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Configuration file used to determine required model assets.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report missing models without downloading them."
    ),
) -> None:
    """Verify local model assets declared in the configuration file."""
    cfg = load_config(config)

    required_assets = cfg.models.assets
    if not required_assets:
        console.print("[green]Configuration does not declare external model assets.[/green]")
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
