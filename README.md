# CubiAI

CubiAI is a two-phase project that converts a single illustration into a fully rigged Live2D asset. Phase one delivers a Typer-based CLI that automates image layer extraction, PSD assembly, Live2D asset packaging, and lightweight rigging. Phase two adds a PySide-powered GUI that builds on the same pipeline for interactive workflows.

## Why CubiAI?
- **AI-assisted layer extraction** – Call Hugging Face Segment Anything endpoints (with local SLIC fallback) to produce semantically meaningful layers straight from a PNG.
- **Production-aware export** – Produce layered PSDs for revision workflows and assemble a Cubism-friendly project structure with textures, physics, and parameter presets.
- **Rigging automation** – Drive an OpenAI-compatible LLM to draft rig parameters, then delegate moc3 creation to a Live2D builder command so the asset can load directly in Cubism.
- **Extensible architecture** – Swap AI providers, customize rig templates, and extend the pipeline for bespoke character styles.

## Project Phases
1. **Phase One – CLI foundation**
   - Typer CLI with subcommands for processing assets, inspecting outputs, and managing profiles.
   - Local workspace management via uv, structured logging, and reproducible processing pipelines.
   - Deterministic configuration files describing model choices, layer schemas, and rig presets.
2. **Phase Two – GUI experience**
   - PySide-based desktop application that wraps the CLI core.
   - Visual previews of layer segmentation, editable rig graphs, and Live2D project validation.
   - Background task orchestration, history management, and export wizards.

## Repository Layout
```
README.md           Project overview
LICENSE             License information
docs/               Extended documentation (architecture, models, usage, roadmap)
```
Additional directories for source code, tests, and assets will be introduced as implementation progresses.

## Getting Started
CubiAI relies on the [uv](https://github.com/astral-sh/uv) Python toolchain plus external AI services. Configure credentials first:

```bash
export HF_API_TOKEN="hf_..."        # Hugging Face token with access to Segment Anything
export OPENAI_API_KEY="sk-..."       # OpenAI-compatible token for rigging LLM calls
export CUBISM_CLI="/Applications/CubismEditor.app/Contents/MacOS/cubism-cli"  # Example path

uv sync
uv run cubiai --help
```

Invoke the pipeline:

```bash
uv run cubiai process ./input/character.png \
    --profile anime-default \
    --output-dir ./build/character
```

> **Hard failure when misconfigured:** the segmentation stage aborts if `HF_API_TOKEN` is missing, and the rigging stage refuses to continue without both an LLM key and a `rigging.builder.command`. This prevents placeholder rigs from being emitted.

## Live2D Builder Integration
- Copy `profiles/anime-default.yaml` and fill in `rigging.builder.command` with the Live2D automation tool you rely on (Cubism CLI, proprietary pipeline, etc.).
- Command arguments may use `{PSD_PATH}`, `{RIG_JSON}`, `{WORKSPACE}`, and `{OUTPUT_DIR}` placeholders, which are resolved before execution.
- The rigging stage confirms `model.moc3` exists after the command finishes; absence raises a `PipelineStageError`.

Consult `docs/cli-usage.md` for end-to-end CLI examples and additional troubleshooting tips.

## Documentation
The `docs/` directory captures the design decisions behind CubiAI:
- `architecture.md` – System boundaries, data flow, and module responsibilities.
- `pipeline.md` – Step-by-step processing pipeline describing each transformation stage.
- `ai-models.md` – Supported AI/LLM providers, required assets, and configuration knobs.
- `gui-roadmap.md` – Phase two planning and UX considerations.
- `cli-usage.md` – CLI usage guide, configuration samples, and troubleshooting tips.

These documents will evolve alongside the implementation. Start with `docs/architecture.md` for a deep dive into the pipeline.

## Contributing
1. Install uv and sync dependencies.
2. Run formatters and tests before submitting changes (`uv run hatch fmt`, `uv run hatch test`, etc. – exact commands TBD).
3. Document new commands, configuration flags, or model integrations.

See `docs/pipeline.md` for development conventions, and `docs/ai-models.md` for detailed service configuration and troubleshooting expectations.

## License
CubiAI is released under the MIT License (see `LICENSE`).
