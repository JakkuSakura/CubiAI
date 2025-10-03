# CLI Usage Guide

The Typer-based CLI (`cubiai`) streamlines processing and configuration management. This guide outlines expected commands and options for phase one.

## Installation
1. Install [uv](https://github.com/astral-sh/uv).
2. Clone the repository and sync dependencies (this installs `torch`, `transformers`, `psd-tools`, etc.):
   ```bash
   uv sync
   ```
3. Run the CLI:
   ```bash
   uv run cubiai --help
   ```

Set the required credentials in your shell session before processing:

```bash
export OPENAI_API_KEY="sk-..."
export CUBISM_CLI="/path/to/cubism-cli"      # optional: required only if rigging is enabled
# optional: export CUBIAI_SAM_HQ_MODEL to override the transformers model id
# optional: export CUBIAI_SAM_HQ_DEVICE to force cpu/cuda/mps
# optional: enable rigging by setting rigging.enabled: true in your config and supplying the above keys
```

## Key Commands

### `cubiai process`
Process an input illustration end-to-end.

```bash
uv run cubiai process ./inputs/character.png \
    --config config/cubiai.yaml \
    --output-dir ./build/character
```

**Options**
- `--config`: Path to the YAML configuration file (defaults to `config/cubiai.yaml`).
- `--output-dir`: Destination directory for pipeline artifacts.
- `--keep-intermediate`: Retain raw segmentations and temporary files.
- `--resume`: Skip stages that have already succeeded for the given workspace.

### `cubiai inspect`
Summarize outputs from a previous run.

```bash
uv run cubiai inspect ./build/character
```

Displays run metadata, stage durations, quality metrics, and validation results. The workspace root contains numbered PNGs under `png/`, `layers.psd`, and a `Live2D/` project folder.

### `cubiai models sync`
Ensure required model weights are present.

```bash
uv run cubiai models sync --config config/cubiai.yaml
```

### `cubiai workspace clean`
Remove cached models or stale workspaces to reclaim disk space.

```bash
uv run cubiai workspace clean --older-than 14d
```

## Configuration
`config/cubiai.yaml` holds all runtime settings (segmentation backend, rigging strategy, builder command, etc.). Duplicate this file if you need project-specific variants and point the CLI at the desired path with `--config`.

### Rigging Builder
Rigging is disabled by default. To enable it, set `rigging.enabled: true` in your configuration and define `rigging.builder.command` to point at a Live2D-capable exporter. The command is a list; placeholders are replaced automatically:

| Placeholder   | Description |
|---------------|-------------|
| `{PSD_PATH}`  | Absolute path to the exported `layers.psd`. |
| `{RIG_JSON}`  | Absolute path to the generated rig description. |
| `{WORKSPACE}` | Workspace root directory. |
| `{OUTPUT_DIR}`| Destination for Live2D assets (defaults to `Live2D/`). |

The pipeline stops with a `PipelineStageError` if the resulting `model.moc3` file is not present in `Live2D/` after the command finishes.

### SAM-HQ Settings
- `segmentation.num_segments` limits how many masks SAM-HQ keeps.
- `segmentation.sam_hq_local_score_threshold` filters low-confidence proposals (0–1 range).
- Override the model/device temporarily with `CUBIAI_SAM_HQ_MODEL` and `CUBIAI_SAM_HQ_DEVICE` environment variables.
- When rigging remains disabled, the pipeline stops after generating PNG/PSD outputs.

## Logging & Diagnostics
- Verbose logs: `--log-level DEBUG`
- Structured JSON logs: `--log-format json`
- LLM/builder transcripts: `diagnostics.json` contains the rigging JSON, builder stdout/stderr, and segmentation metadata.
- First run with `sam-hq-local` downloads weights to the Hugging Face cache; subsequent runs reuse the local files.

## Exit Codes
- `0`: Success.
- `10`: Validation failure (e.g., missing Live2D assets).
- `20`: Model dependency missing.
- `30`: Unexpected exception.

## Troubleshooting
- Use `--dry-run` to validate configuration without running models.
- Check `diagnostics.json` in the workspace for detailed errors.
- Missing `OPENAI_API_KEY` or `CUBISM_CLI` results in an immediate `PipelineStageError` with the offending variable listed.
- See `docs/ai-models.md` for resolving external service issues and builder configuration.

This CLI guide will expand as additional subcommands and options are implemented.
