"""`cubiai annotate` command for generating LabelMe JSON via LangChain."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from ...config.loader import DEFAULT_CONFIG_PATH, load_config
from ...logging_utils import configure_logging
from ...services.labelme_annotation import (
    AnnotationLLMError,
    LangChainAnnotationTool,
)

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
    label: list[str] = typer.Option(
        None,
        "--label",
        help="Append a semantic label that the annotator should consider (repeatable).",
    ),
    model: str | None = typer.Option(None, "--model", help="Override the configured chat model name."),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Override the OpenAI-compatible base URL (e.g., for Azure or local gateways).",
    ),
    api_key: str | None = typer.Option(None, "--api-key", help="API key used for the chat model."),
    temperature: float | None = typer.Option(None, "--temperature", help="Sampling temperature override."),
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
    """Generate LabelMe annotations using a LangChain LLM backend."""

    configure_logging(level=log_level)

    if not image_path.exists():
        raise typer.BadParameter(f"Image not found: {image_path}")

    cfg = load_config(config)
    llm_settings = cfg.annotation.llm.model_copy(deep=True)
    llm_settings.enabled = True
    if model:
        llm_settings.model = model
    if base_url is not None:
        llm_settings.base_url = base_url or None
    if temperature is not None:
        llm_settings.temperature = temperature

    if instructions_file:
        extra_instructions = instructions_file.read_text().strip()
        instructions = f"{instructions or ''}\n{extra_instructions}".strip()

    labels = label or []

    tool_kwargs: dict[str, object] = {
        "settings": llm_settings,
        "api_key": api_key,
        "include_image_data": include_image_data,
    }
    if labels:
        tool_kwargs["labels"] = labels

    try:
        tool = LangChainAnnotationTool(**tool_kwargs)
    except AnnotationLLMError as exc:
        raise typer.BadParameter(str(exc)) from exc

    try:
        annotation = tool.annotate(image_path=image_path, instructions=instructions, extra_labels=labels)
    except AnnotationLLMError as exc:
        console.print(f"[bold red]Annotation failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    target_path = output_path or image_path.with_suffix(".labelme.json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(annotation.model_dump_json(indent=2))

    console.print(f"[bold green]Annotation saved:[/bold green] {target_path}")
    console.print(f"[dim]Classes used:[/dim] {', '.join(label for label in labels) if labels else 'default set'}")
