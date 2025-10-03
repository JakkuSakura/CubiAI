"""`cubiai annotate` command for generating LabelMe JSON via the Codex CLI."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ...config.loader import DEFAULT_CONFIG_PATH, load_config
from ...logging_utils import configure_logging
from ...services.labelme_annotation import AnnotationLLMError, CodexAnnotationTool

console = Console()


def annotate(
    image_path: Path = typer.Argument(..., help="Path to the image requiring annotation."),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Target path for the LabelMe JSON. Defaults to <image>.labelme.json",
    ),
    instructions: str | None = typer.Option(
        None,
        "--instructions",
        "-i",
        help="Additional prompt guidance passed to the LLM.",
    ),
    instructions_file: Path | None = typer.Option(
        None,
        "--instructions-file",
        help="Load extra instructions from a text file.",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    label: list[str] | None = typer.Option(
        None,
        "--label",
        help="Append a semantic label that the annotator should consider (repeatable).",
    ),
    model: str | None = typer.Option(None, "--model", help="Override the configured model name."),
    codex_binary: str | None = typer.Option(
        None,
        "--codex-binary",
        help="Path to the Codex CLI executable (default: value from configuration).",
    ),
    extra_cli_arg: list[str] | None = typer.Option(
        None,
        "--extra-cli-arg",
        help="Additional argument forwarded to the Codex CLI (repeatable).",
    ),
    timeout_seconds: int | None = typer.Option(
        None,
        "--timeout-seconds",
        min=30,
        help="Override the Codex CLI subprocess timeout in seconds.",
    ),
    include_image_data: bool = typer.Option(
        False,
        "--embed-image",
        help="Embed base64 image data into the LabelMe JSON output.",
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
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity (default: INFO)."),
) -> None:
    """Generate LabelMe annotations using the Codex CLI backend."""

    configure_logging(level=log_level)

    if not image_path.exists():
        raise typer.BadParameter(f"Image not found: {image_path}")

    cfg = load_config(config)
    llm_settings = cfg.annotation.llm.model_copy(deep=True)
    llm_settings.enabled = True
    if model:
        llm_settings.model = model
    if codex_binary:
        llm_settings.codex_binary = codex_binary
    if timeout_seconds is not None:
        llm_settings.timeout_seconds = timeout_seconds
    if extra_cli_arg is not None:
        llm_settings.extra_cli_args = list(extra_cli_arg)

    if instructions_file:
        extra_instructions = instructions_file.read_text().strip()
        instructions = f"{instructions or ''}\n{extra_instructions}".strip()

    labels = label or []

    tool_kwargs: dict[str, object] = {
        "settings": llm_settings,
        "include_image_data": include_image_data,
    }
    if labels:
        tool_kwargs["labels"] = labels

    try:
        tool = CodexAnnotationTool(**tool_kwargs)
    except AnnotationLLMError as exc:
        raise typer.BadParameter(str(exc)) from exc

    target_path = output_path or image_path.with_suffix(".labelme.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = tool.annotate(
            image_path=image_path,
            output_path=target_path,
            instructions=instructions,
            extra_labels=labels,
        )
    except AnnotationLLMError as exc:
        console.print(f"[bold red]Annotation failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    target_path.write_text(result.annotation.model_dump_json(indent=2))

    console.print(f"[bold green]Annotation saved:[/bold green] {target_path}")
    console.print(f"[dim]Summary:[/dim] {result.summary}")
    console.print(f"[dim]Classes used:[/dim] {', '.join(label for label in labels) if labels else 'default set'}")
