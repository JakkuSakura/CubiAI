"""`cubiai preview` command to visualise LabelMe annotations using Skia."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ...logging_utils import configure_logging
from ...services.annotation_preview import PreviewStyle, render_annotation_preview

console = Console()


def preview(
    image_path: Path = typer.Argument(..., help="Path to the source image."),
    annotation_path: Path | None = typer.Option(
        None,
        "--annotation",
        "-a",
        help="Path to the LabelMe annotation JSON (defaults to <image>.labelme.json).",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Destination path for the rendered preview PNG (defaults to <image>_preview.png).",
    ),
    stroke_width: float | None = typer.Option(
        None,
        "--stroke-width",
        min=0.5,
        help="Outline thickness for polygons (default 2.5).",
    ),
    fill_opacity: float | None = typer.Option(
        None,
        "--fill-opacity",
        min=0.0,
        max=1.0,
        help="Opacity applied to polygon fills (default 0.35).",
    ),
    label_background_opacity: float | None = typer.Option(
        None,
        "--label-bg-opacity",
        min=0.0,
        max=1.0,
        help="Opacity of the label text background chip (default 0.85).",
    ),
    label_text_size: float | None = typer.Option(
        None,
        "--label-text-size",
        min=6.0,
        help="Font size used for labels in pixels (default 16).",
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity (default: INFO)."),
) -> None:
    """Render an overlay of LabelMe polygons on top of the source image."""

    configure_logging(level=log_level)

    if not image_path.exists():
        raise typer.BadParameter(f"Image not found: {image_path}")

    annotation_file = annotation_path or image_path.with_suffix(".labelme.json")
    if not annotation_file.exists():
        raise typer.BadParameter(f"Annotation file not found: {annotation_file}")

    target_path = output_path or image_path.with_name(f"{image_path.stem}.preview.png")

    base_style = PreviewStyle()
    style = PreviewStyle(
        stroke_width=stroke_width if stroke_width is not None else base_style.stroke_width,
        fill_opacity=fill_opacity if fill_opacity is not None else base_style.fill_opacity,
        label_background_opacity=(
            label_background_opacity
            if label_background_opacity is not None
            else base_style.label_background_opacity
        ),
        label_text_size=label_text_size if label_text_size is not None else base_style.label_text_size,
    )

    try:
        render_annotation_preview(
            image_path=image_path,
            annotation_path=annotation_file,
            output_path=target_path,
            style=style,
        )
    except Exception as exc:  # pragma: no cover - runtime specific
        console.print(f"[bold red]Preview failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[bold green]Preview saved:[/bold green] {target_path}")
