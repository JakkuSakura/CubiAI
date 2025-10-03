# Processing Pipeline

This document breaks down the CubiAI end-to-end pipeline. Each stage is modular and can be toggled or swapped via YAML configuration files.

## 1. Intake & Workspace Setup
- Validate file format (PNG, PSD, TIFF, WebP).
- Normalize color space to RGBA, resize or pad to power-of-two dimensions if required by downstream models.
- Create a unique workspace directory that captures intermediate artifacts and logs.

## 2. Semantic Segmentation
- When `backend = sam-hq-local`, load the SAM-HQ weights locally via `transformers` and generate masks from an automatically generated point grid.
- `huggingface-sam` can still be selected to call the hosted inference API; `slic` remains as an offline fallback without deep models.
- Masks smaller than `segmentation.min_area_px` are discarded before PSD export to avoid noise.
- All parameters (`sam_hq_local_*`, `num_segments`, etc.) are controlled in the YAML configuration file.
- Output: Layered RGBA renders preserved as PNGs plus metadata (score, area, bounding box) saved per segment.

## 3. Matting & Edge Refinement
- Apply alpha matting and edge-aware smoothing to remove halos and ensure crisp boundaries.
- Blend overlapping layers to maintain occlusion order.
- Output: Clean RGBA layers ready for export.

## 4. Layer Post-Processing
- Snap layers to a consistent canvas size and origin.
- Auto-fill gaps with inpainting when necessary (e.g., reconstructing hair behind the head when arm is separated).
- Perform color correction and optional relighting to maintain consistency across layers.

## 5. PSD Compilation
- Rebuild a layered PSD from the numbered PNG stack so the PSD matches exactly what was exported.
- Embed metadata about layer names, z-order, and segmentation confidence in PSD layer tags.
- Optional: output additional review artifacts (flattened previews, thumbnails) to accompany the PSD.

## 6. PNG Archive
- Write every processed layer to an individual transparent PNG (`png/001.png`, `png/002.png`, …) for quick inspection or manual adjustments.
- The PNG directory path is recorded in the run metadata for downstream tools.

## 7. Live2D Asset Generation (Optional)
- When `rigging.enabled` is set to `true`, invoke the configured builder command (see `rigging.builder.command`) to convert the PSD and AI-generated rig data into a real `model.moc3` file. A missing command or `model.moc3` triggers a hard failure.
- Generate texture atlases for each layer group (textures stored under `Live2D/Resources/Textures`).
- Build `model3.json` referencing textures, physics settings, motions, and the builder-produced `model.moc3`.
- Package additional assets (expression JSON, physics graphs, pose configurations). Placeholder physics/motions are written only if the builder does not provide them.

## 8. Rigging Automation (Optional)
- Feed layer metadata to the LLM rigging assistant. The assistant must return structured JSON defining parts, deformers, physics, and motion stubs.
- Validate the JSON response and persist it as `rig_config.json` for traceability.
- Hand the rig description to the external builder; builder stdout/stderr is saved in diagnostics for audit.
- Export additional diagnostics (e.g., summary of LLM parameters) to `diagnostics.json`.

## 9. Validation & Packaging
- Run sanity checks (texture size limits, parameter count, missing meshes).
- Confirm `model.moc3` exists and log the builder command that produced it.
- Produce a summary report with pipeline timings and quality metrics (IoU, edge preservation).
- Archive the workspace as a `.zip` optionally uploaded to a storage provider.

## Incremental Workflow
- Each stage is resumable. If a step fails, rerun with `--resume` to skip previously completed stages.
- Developers can enable `--keep-intermediate` to retain raw tensors for debugging.

## Extending the Pipeline
- Add custom stages by registering callables in the pipeline configuration.
- Chain remote inference services by implementing the provider interface.
- Insert manual review gates (e.g., require human approval before rigging) by toggling `stop_after` parameter.

For CLI usage examples, see `docs/cli-usage.md`. For details on AI providers, see `docs/ai-models.md`.
