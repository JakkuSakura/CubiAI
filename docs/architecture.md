# CubiAI Architecture

CubiAI automates the journey from a single illustration to a Live2D-ready asset. The solution is structured as a series of modular services orchestrated through a configurable pipeline. This document captures the conceptual architecture and engineering priorities for both phases of the project.

## High-Level Components

1. **CLI / GUI Front-Ends**
   - Phase one exposes functionality through a Typer CLI (`cubiai`).
   - Phase two introduces a PySide GUI that wraps the same application services and provides visualization.
2. **Application Core**
   - Pipeline coordinator that orchestrates discrete processing stages.
   - Dependency injection container for AI providers, exporters, and rig generators.
   - Configuration loader that resolves model checkpoints, thresholds, and rig templates from YAML files.
3. **AI Processing Services**
   - **Segmentation Service**: Splits the base illustration into semantic layers. The default path loads SAM-HQ locally through `transformers`/`torch`, while hosted SAM and SLIC remain as alternative backends.
   - **Detail Refinement**: Applies matting and edge refinement to remove halos and maintain alpha fidelity.
   - **Post-processing**: Normalizes colors, fills gaps, and enforces consistent canvas sizes across layers.
4. **Asset Exporters**
   - **Layered PSD Exporter**: Collects individually masked layers and renders them as a layered PSD with organized groups for manual review or manual editing.
   - **PNG Exporter**: Writes every processed layer as a transparent PNG for quick inspection or hand edits.
   - **Live2D Exporter**: Packages textures, meshes, physics, motions, and metadata into a Cubism project layout.
5. **Rigging Engine (Optional)**
   - When enabled, sends layer metadata to an OpenAI-compatible LLM that proposes parts, parameters, deformers, and motion stubs.
   - Delegates moc3 creation to an external builder command (e.g., Cubism SDK automation). Builder stdout/stderr are captured for diagnostics.
   - Produces `model3.json`, pose/physics JSON, and optional `.motion3.json` animations, skipping any artefact the builder already provided.
6. **Storage and Asset Management**
   - Workspace abstraction to manage input assets, intermediate caches, and final outputs.
   - Persistence for pipeline checkpoints, enabling resumable processing and reproducibility.
7. **Observability**
   - Structured logging, progress events, API payload tracing, and captured stdout/stderr from the Live2D builder.

## Data Flow

```text
input image → preprocessing → segmentation & matting → layer post-processing
            → PSD generation → Live2D asset compilation → rig validation
```

Each stage produces artifacts under a run-specific workspace directory (e.g., `build/<timestamp>/`). The coordinator enforces a deterministic order, handles retries, and records metadata about model versions and configuration overrides.

## Configuration Model

- **Configuration files (`config/*.yaml`)** describe model backends, hyper-parameters, and rig templates. Example keys:
  - `segmentation.backend`: `sam-hq-local` (default), `huggingface-sam`, or `slic`.
  - `segmentation.sam_hq_local_*`: local SAM-HQ model identifier, device, and score thresholds.
  - `rigging.strategy`: `llm` or `heuristic` along with LLM model ID and API key environment variable.
  - `rigging.builder.command`: command template used to invoke a Live2D moc3 builder.
  - `export.live2d.texture_size`: output atlas size.
- The application core loads the configuration file, hydrates services, and validates dependencies (model weights, API keys).

## Extensibility Strategy

- **Plugin Interfaces**: Abstract base classes define the contracts for segmentation providers, PSD exporters, and rig pipelines.
- **Entry Points**: Additional providers can be registered via `pyproject.toml` entry points, enabling third-party contributions.
- **Configuration-driven Behavior**: Most runtime decisions derive from YAML configuration files, minimizing code changes for experimentation.
- **Sandboxed Execution**: Potentially expensive stages (e.g., AI inference) can run in isolated subprocesses or via gRPC workers.

## Error Handling & Validation

- Validation at the start of each stage ensures required inputs are present.
- Stages emit human-readable diagnostics stored alongside outputs (`diagnostics.json`).
- The rigging step performs compatibility checks with Live2D Cubism (texture sizes, polygon counts).

## Phase Two Considerations

- The GUI will share the application core via a service layer. UI components subscribe to progress events and render previews in Qt widgets.
- Long-running tasks will execute in worker threads/processes with cancellation support.
- User adjustments (e.g., mask cleanup) feed back into the pipeline for reprocessing.

## Security & Privacy

- Configuration files can declare remote inference backends (e.g., private API endpoints). Secrets are loaded from environment variables or `.env` files and never stored in run artifacts.
- The CLI supports `--offline` mode to disable remote providers entirely.

## Next Steps

1. Implement the pipeline coordinator and default segmentation backend.
2. Define configuration schemas and validation logic.
3. Flesh out rig templates and ensure Live2D exports meet Cubism import requirements.
4. Instrument logs and diagnostics for troubleshooting.

This architecture will evolve as we validate performance, accuracy, and usability goals. Feedback and contributions are welcome.
