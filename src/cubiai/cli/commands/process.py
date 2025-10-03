"""`cubiai process` command implementation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ...config.loader import DEFAULT_CONFIG_PATH, load_config
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
    config: Path = typer.Option(
        DEFAULT_CONFIG_PATH,
        "--config",
        "-c",
        help="Path to the CubiAI configuration file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
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

    cfg = load_config(config)

    runner = PipelineRunner(config=cfg, workspace=workspace)

    start = time.perf_counter()
    console.log("Starting pipeline", cfg.name, "→", workspace.root)
    result = runner.run()
    duration = time.perf_counter() - start

    metadata_path = workspace.root / "metadata.json"
    metadata = {
        "input": str(input_path),
        "config_path": str(config),
        "config_name": cfg.name,
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
