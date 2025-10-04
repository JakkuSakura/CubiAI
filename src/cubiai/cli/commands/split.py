"""`cubiai split` command to export layers from LabelMe annotations."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ...logging_utils import configure_logging
from ...services.labelme_splitter import LayerSlice, split_annotation_layers

console = Console()


def split(
    image_path: Path = typer.Argument(..., help="Path to the source image."),
    annotation_path: Path | None = typer.Option(
        None,
        "--annotation",
        "-a",
        help="Path to the LabelMe JSON file (defaults to <image>.labelme.json).",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory where the extracted layers will be saved (defaults to <image>_layers).",
    ),
    include_mask: bool = typer.Option(
        True,
        "--include-mask/--no-mask",
        help="Also export alpha masks for each layer (default: enabled).",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity (default: INFO)."),
) -> None:
    """Extract polygon-defined layers from a LabelMe annotation."""

    configure_logging(level=log_level)

    if not image_path.exists():
        raise typer.BadParameter(f"Image not found: {image_path}")

    annotation_file = annotation_path or image_path.with_suffix(".labelme.json")
    if not annotation_file.exists():
        raise typer.BadParameter(f"Annotation file not found: {annotation_file}")

    target_dir = output_dir or image_path.with_name(f"{image_path.stem}_layers")

    try:
        slices = split_annotation_layers(
            image_path=image_path,
            annotation_path=annotation_file,
            output_dir=target_dir,
            include_mask=include_mask,
        )
    except Exception as exc:  # pragma: no cover - runtime
        console.print(f"[bold red]Layer extraction failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    if not slices:
        console.print("[bold yellow]No polygon layers were exported from the annotation.[/bold yellow]")
        raise typer.Exit(code=0)

    _print_summary(slices, target_dir, include_mask)


def _print_summary(slices: list[LayerSlice], target_dir: Path, include_mask: bool) -> None:
    console.print(f"[bold green]Exported {len(slices)} layer(s) to {target_dir}[/bold green]")
    for item in slices:
        mask_info = f", mask={item.mask_path.name}" if include_mask and item.mask_path else ""
        console.print(
            f" - #{item.index:02d} {item.label} → {item.image_path.name}{mask_info} (area={item.area:.0f}px²)"
        )
