"""`cubiai process` command implementation."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Sequence

import typer
from rich.console import Console
from rich.table import Table

from ...config.loader import InlineOverride, load_profile
from ...logging_utils import configure_logging
from ...pipeline.runner import PipelineRunner
from ...workspace import Workspace

console = Console()


def process(
    input_path: Path = typer.Argument(..., help="Path to the source image."),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory that will contain generated artifacts. Defaults to ./build/<stem>.",
    ),
    profile: str = typer.Option(
        "anime-default",
        "--profile",
        "-p",
        help="Processing profile name located under the profiles/ directory.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Optional YAML/TOML file providing overrides that extend the profile.",
    ),
    set: Optional[Sequence[str]] = typer.Option(  # noqa: A002 - aligns with CLI vernacular
        None,
        "--set",
        help="Inline overrides using dotted keys, e.g. segmentation.num_segments=16.",
    ),
    keep_intermediate: bool = typer.Option(
        False, "--keep-intermediate", help="Retain intermediate files such as raw masks."
    ),
    resume: bool = typer.Option(False, "--resume", help="Resume from an existing workspace."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
) -> None:
    """Run the CubiAI processing pipeline."""
    configure_logging(level=log_level)

    if not input_path.exists():
        raise typer.BadParameter(f"Input path {input_path} does not exist")

    workspace_root = (
        output_dir
        if output_dir is not None
        else Path("build") / input_path.stem / time.strftime("%Y%m%d-%H%M%S")
    )
    workspace = Workspace.create(
        input_path=input_path,
        root=workspace_root,
        keep_intermediate=keep_intermediate,
        resume=resume,
    )

    profile_config = load_profile(
        profile_name=profile,
        overrides_path=config,
        inline_overrides=[InlineOverride.parse(o) for o in (set or [])],
    )

    runner = PipelineRunner(profile=profile_config, workspace=workspace)

    start = time.perf_counter()
    console.log("Starting pipeline", profile_config.name, "→", workspace.root)
    result = runner.run()
    duration = time.perf_counter() - start

    metadata_path = workspace.root / "metadata.json"
    metadata = {
        "input": str(input_path),
        "profile": profile_config.name,
        "workspace": str(workspace.root),
        "stages": [stage.model_dump() for stage in result.stage_results],
        "outputs": {key: str(value) for key, value in result.outputs.items()},
        "duration_seconds": duration,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    console.print(f"[bold green]Pipeline finished[/bold green] in {duration:.2f}s")
    table = Table(title="Outputs", show_header=True, header_style="bold magenta")
    table.add_column("Artifact")
    table.add_column("Path")
    for name, path in result.outputs.items():
        table.add_row(name, str(path))
    console.print(table)
