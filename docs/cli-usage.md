# CLI Usage Guide

The Typer-based CLI (`cubiai`) streamlines processing and configuration management. This guide outlines expected commands and options for phase one.

## Installation
1. Install [uv](https://github.com/astral-sh/uv).
2. Clone the repository and sync dependencies:
   ```bash
   uv sync
   ```
3. Run the CLI:
   ```bash
   uv run cubiai --help
   ```

Set the required credentials in your shell session before processing:

```bash
export HF_API_TOKEN="hf_..."
export OPENAI_API_KEY="sk-..."
export CUBISM_CLI="/path/to/cubism-cli"
```

## Key Commands

### `cubiai process`
Process an input illustration end-to-end.

```bash
uv run cubiai process ./inputs/character.png \
    --profile anime-default \
    --output-dir ./build/character
```

**Options**
- `--profile`: Selects the configuration profile (YAML) describing model backends and rig template.
- `--output-dir`: Destination directory for pipeline artifacts.
- `--keep-intermediate`: Retain raw segmentations and temporary files.
- `--resume`: Skip stages that have already succeeded for the given workspace.
- `--no-psd`, `--no-live2d`: Disable specific export stages.

### `cubiai inspect`
Summarize outputs from a previous run.

```bash
uv run cubiai inspect ./build/character
```

Displays run metadata, stage durations, quality metrics, and validation results. The workspace root contains layered PNGs, `layers.psd`, and a `Live2D/` project folder.

### `cubiai models sync`
Ensure required model weights are present.

```bash
uv run cubiai models sync --profile anime-default
```

### `cubiai profiles list`
Discover available processing profiles and their descriptions.

```bash
uv run cubiai profiles list
```

### `cubiai workspace clean`
Remove cached models or stale workspaces to reclaim disk space.

```bash
uv run cubiai workspace clean --older-than 14d
```

## Configuration
Profiles live in `profiles/*.yaml`. Override individual settings via CLI flags:

```bash
uv run cubiai process input.png --profile anime-default --config overrides.yaml
```

Inline overrides are also supported:

```bash
uv run cubiai process input.png --set segmentation.backend=sam-l-small
```

### Rigging Builder

Define `rigging.builder.command` in your profile to point at a Live2D-capable exporter. The command is a list; placeholders are replaced automatically:

| Placeholder   | Description |
|---------------|-------------|
| `{PSD_PATH}`  | Absolute path to the exported `layers.psd`. |
| `{RIG_JSON}`  | Absolute path to the generated rig description. |
| `{WORKSPACE}` | Workspace root directory. |
| `{OUTPUT_DIR}`| Destination for Live2D assets (defaults to `Live2D/`). |

The pipeline stops with a `PipelineStageError` if the resulting `model.moc3` file is not present in `Live2D/` after the command finishes.

## Logging & Diagnostics
- Verbose logs: `--log-level DEBUG`
- Structured JSON logs: `--log-format json`
- LLM/builder transcripts: `diagnostics.json` contains the rigging JSON, builder stdout/stderr, and segmentation metadata.

## Exit Codes
- `0`: Success.
- `10`: Validation failure (e.g., missing Live2D assets).
- `20`: Model dependency missing.
- `30`: Unexpected exception.

## Troubleshooting
- Use `--dry-run` to validate configuration without running models.
- Check `diagnostics.json` in the workspace for detailed errors.
- Missing `HF_API_TOKEN` or `OPENAI_API_KEY` results in an immediate `PipelineStageError` with the offending variable listed.
- See `docs/ai-models.md` for resolving external service issues and builder configuration.

This CLI guide will expand as additional subcommands and options are implemented.
