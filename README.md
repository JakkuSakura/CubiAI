# CubiAI

CubiAI is a two-phase project that converts a single illustration into a fully rigged Live2D asset. Phase one delivers a Typer-based CLI that automates image layer extraction, PSD assembly, Live2D asset packaging, and lightweight rigging. Phase two adds a PySide-powered GUI that builds on the same pipeline for interactive workflows.

## Why CubiAI?
- **AI-assisted layer extraction** – Run SAM-HQ locally via `transformers`/`torch` (with optional hosted SAM or SLIC fallbacks) to produce semantically meaningful layers straight from a PNG.
- **Production-aware export** – Produce layered PSDs for revision workflows and assemble a Cubism-friendly project structure with textures, physics, and parameter presets.
- **Layer archive** – Save every extracted layer as a transparent PNG (`png/001.png`, `png/002.png`, …) for quick reviews or manual edits.
- **Rigging automation (optional)** – When enabled, drive an OpenAI-compatible LLM to draft rig parameters, then delegate moc3 creation to a Live2D builder command so the asset can load directly in Cubism.
- **Extensible architecture** – Swap AI providers, customize rig templates, and extend the pipeline for bespoke character styles.

## Project Phases
1. **Phase One – CLI foundation**
   - Typer CLI with subcommands for processing assets, inspecting outputs, and maintaining a single YAML configuration.
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
## Data
This project expects a raw portrait dataset under `data/raw/danbooru2019/`.

Danbooru2023 is huge (~8 TB). For experimentation you can fetch a single shard,
for example `original/data-0000.tar`.
https://huggingface.co/datasets/nyanko7/danbooru2023/resolve/main/original/data-0000.tar
```shell
tar xf data-0000.tar -C data/raw/danbooru2019/
```

## Getting Started
CubiAI relies on the [uv](https://github.com/astral-sh/uv) Python toolchain plus external AI services. Configure credentials first:

```bash
export OPENAI_API_KEY="sk-..."       # OpenAI-compatible token for rigging LLM calls
export CUBISM_CLI="/Applications/CubismEditor.app/Contents/MacOS/cubism-cli"  # Optional (needed only if rigging is enabled)
# optional overrides for SAM-HQ
# export CUBIAI_SAM_HQ_MODEL=syscv-community/sam-hq-vit-base
# export CUBIAI_SAM_HQ_DEVICE=cuda

uv sync
uv run cubiai --help
```

SAM-HQ segmentation runs locally using `transformers` + `torch`. The first execution downloads the model weights into your Hugging Face cache; subsequent runs work offline.

Invoke the pipeline:

```bash
uv run cubiai process ./input/character.png \
    --config config/cubiai.yaml \
    --output-dir ./build/character
```

### Train the pass-through animator

Organise your dataset as `root/video_id/frame.png`. Each folder should contain sequential frames from a capture or paired video. Then run:

```bash
uv sync
uv run python scripts/train_pass_through.py ./dataset/live2d ./runs/pass_through \
    --size 1024 --low-res 256 --batch 1 --steps 2000
```

The script trains the flow+mask model, prints metrics, and writes `preview.png` with a sample result. Adjust `--size`, `--steps`, and other flags as needed. For inference, import `PassThroughAnimator` and call it with a source (`S`) and driver (`D`) frame:

```python
from cubiai.models.pass_through_animator import PassThroughAnimator
model = PassThroughAnimator(low_res=256).to(device)
with torch.no_grad():
    result = model(source_tensor, driver_tensor)["output"]
```

Generate LabelMe annotations (cluster backend runs by default):

```bash
uv run cubiai annotate ./input/character.png --output ./build/character/character.labelme.json
```

Need a language model instead? Pass `--strategy codex` (and optionally labels/instructions) to fall back to the Codex-driven pipeline.

> **Hard failure when misconfigured:** the rigging stage is disabled by default; enable it by setting `rigging.enabled: true` and supplying both an LLM key and a `rigging.builder.command`. The segmentation backend loads SAM-HQ locally via `transformers`+`torch`; the first run downloads weights from Hugging Face unless they are already cached.

Outputs include:
- `layers.psd` built from the PNG stack.
- Numbered transparent PNGs for each layer under `png/`.
- Optional Live2D project structure (textures, `model3.json`, moc3) when rigging is enabled and a builder command is provided.

## Live2D Builder Integration
- Edit `config/cubiai.yaml` and fill in `rigging.builder.command` with the Live2D automation tool you rely on (Cubism CLI, proprietary pipeline, etc.).
- Command arguments may use `{PSD_PATH}`, `{RIG_JSON}`, `{WORKSPACE}`, and `{OUTPUT_DIR}` placeholders, which are resolved before execution.
- The rigging stage confirms `model.moc3` exists after the command finishes; absence raises a `PipelineStageError`.

Consult `docs/cli-usage.md` for end-to-end CLI examples and additional troubleshooting tips.

## Configuration
- All runtime settings reside in `config/cubiai.yaml`. Duplicate this file for project-specific presets and point the CLI at your variant with `--config`.
- Key sections include `segmentation` (SAM-HQ parameters), `annotation.strategy` + `annotation.cluster` (semi-supervised model paths and thresholds), `annotation.llm` (Codex fallback settings), `rigging` (builder command), and `export` (PSD/Live2D options).

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
