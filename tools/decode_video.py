#!/usr/bin/env python3
"""Decode a video into the dataset layout expected by PortraitVideoDataset.

This script extracts all frames of a video using ``ffmpeg`` into a directory
named ``{name}_video`` and saves the first frame as ``{name}.png``. It ensures
the directory structure matches what ``PortraitVideoDataset`` expects.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def slugify(stem: str) -> str:
    """Convert a filename stem into a simple dataset-friendly identifier."""

    cleaned = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in stem)
    cleaned = cleaned.strip("_-")
    return cleaned or "sample"


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found in PATH", file=sys.stderr)
        sys.exit(1)


def run_ffmpeg(args: list[str]) -> None:
    completed = subprocess.run(args, check=False)
    if completed.returncode != 0:
        print("error: ffmpeg command failed", file=sys.stderr)
        sys.exit(completed.returncode)


def decode_video(video_path: Path, dataset_root: Path, *, name: str, overwrite: bool) -> None:
    ensure_ffmpeg()

    dataset_root.mkdir(parents=True, exist_ok=True)

    portrait_path = dataset_root / f"{name}.png"
    frames_dir = dataset_root / f"{name}_video"

    if frames_dir.exists():
        if not overwrite:
            print(f"error: {frames_dir} already exists (use --overwrite to replace)", file=sys.stderr)
            sys.exit(1)
        if not frames_dir.is_dir():
            print(f"error: {frames_dir} exists but is not a directory", file=sys.stderr)
            sys.exit(1)
        shutil.rmtree(frames_dir)

    if portrait_path.exists():
        if not overwrite:
            print(f"error: {portrait_path} already exists (use --overwrite to replace)", file=sys.stderr)
            sys.exit(1)
        portrait_path.unlink()

    frames_dir.mkdir(parents=True, exist_ok=True)

    # Extract all frames to the frames directory.
    run_ffmpeg([
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
    ])

    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        print("error: ffmpeg did not produce any PNG frames", file=sys.stderr)
        sys.exit(1)

    first_frame = frames[0]
    shutil.copy(first_frame, portrait_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Destination dataset directory (e.g. data/dataset_1)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Base name for the dataset entry (defaults to video stem)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing frames directory and portrait image",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    opts = parse_args(argv)

    video_path = opts.video.expanduser()
    dataset_root = opts.dataset_root.expanduser()

    if not video_path.is_file():
        print(f"error: video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    name = slugify(opts.name) if opts.name else slugify(video_path.stem)

    decode_video(video_path, dataset_root, name=name, overwrite=opts.overwrite)


if __name__ == "__main__":
    main()

