#!/usr/bin/env python3
"""Decode a video into the dataset layout expected by PortraitVideoDataset.

This Typer-powered CLI extracts video frames with ``ffmpeg`` into ``{name}_video``
and saves the first frame as ``{name}.png`` under the chosen dataset root.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import typer


app = typer.Typer(help="Decode a video into portrait + frame directory pairs.", add_completion=False)


def abort(message: str) -> None:
    """Exit the CLI early with an error message."""

    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def slugify(stem: str) -> str:
    """Convert a filename stem into a simple dataset-friendly identifier."""

    cleaned = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in stem)
    cleaned = cleaned.strip("_-")
    return cleaned or "sample"


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        abort("ffmpeg not found in PATH")


def run_ffmpeg(args: list[str]) -> None:
    completed = subprocess.run(args, check=False)
    if completed.returncode != 0:
        abort("ffmpeg command failed")


def decode_video(video_path: Path, dataset_root: Path, *, name: str, overwrite: bool) -> Path:
    ensure_ffmpeg()

    dataset_root.mkdir(parents=True, exist_ok=True)

    portrait_path = dataset_root / f"{name}.png"
    frames_dir = dataset_root / f"{name}_video"

    if frames_dir.exists():
        if not overwrite:
            abort(f"{frames_dir} already exists (use --overwrite to replace)")
        if not frames_dir.is_dir():
            abort(f"{frames_dir} exists but is not a directory")
        shutil.rmtree(frames_dir)

    if portrait_path.exists():
        if not overwrite:
            abort(f"{portrait_path} already exists (use --overwrite to replace)")
        portrait_path.unlink()

    frames_dir.mkdir(parents=True, exist_ok=True)

    run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vsync",
            "0",
            "-start_number",
            "1",
            str(frames_dir / "%08d.png"),
        ]
    )

    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        abort("ffmpeg did not produce any PNG frames")

    first_frame = frames[0]
    shutil.copy(first_frame, portrait_path)

    return frames_dir


@app.command()
def decode(
    video: Path = typer.Argument(..., help="Input video file"),
    dataset_root: Path = typer.Argument(..., help="Destination dataset directory"),
    overwrite: bool = typer.Option(False, "--overwrite", "-f", help="Replace existing portrait/frames if present"),
) -> None:
    """Extract frames and portrait image for dataset ingestion."""

    video = video.expanduser()
    dataset_root = dataset_root.expanduser()

    if not video.is_file():
        abort(f"video file not found: {video}")

    slug = slugify(video.stem)
    frames_dir = decode_video(video, dataset_root, name=slug, overwrite=overwrite)

    typer.echo(f"Decoded frames to {frames_dir} and portrait to {dataset_root / (slug + '.png')}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
