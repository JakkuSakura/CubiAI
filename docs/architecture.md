# Architecture Overview

CubiAI is evolving into a few-shot motion transfer system for Live2D characters. The solution is organised into modular services so that motion descriptors, adapters, and image generators can be upgraded independently.

## Component Map

1. **Data Layer**
   - Portrait/video dataset loader (`PortraitVideoDataset`) handles static character art and driver clips.
   - Future drivers (face parser outputs, human video) will be normalised through shared preprocessing utilities.

2. **Motion Descriptor Stack (planned)**
   - **Unsupervised keypoint extractor** discovers canonical motion landmarks from Live2D clips with minimal labels.
   - **Domain adapters** map external drivers (human faces, parser maps) into the same descriptor space using lightweight MLP/CNN heads.
   - **Descriptor buffers** cache trajectories for training the animator without repeatedly re-encoding video frames.

3. **Animator**
   - Residual U-Net conditioned via FiLM/AdaIN on a portrait appearance embedding.
   - Consumes the static portrait, driver frame, and (soon) descriptor tensors to predict the next stylised frame directly at target resolution.
   - Emits only the final RGB image—motion-alignment penalties are computed post hoc for training losses.

4. **Training Orchestrator**
   - `PassThroughTrainer` (to be renamed) manages optimiser state, gradient clipping, and logging.
   - Losses combine reconstruction, driver-aligned motion regularisation, and identity preservation. Planned upgrades will swap raw RGB terms for descriptor-based comparisons.

5. **CLI Interface**
   - Typer commands expose training/inference (`cubiai model train`, `cubiai model infer`).
   - Future commands will ingest descriptor caches, run evaluation suites, and export preview videos.

6. **Evaluation & Tooling (roadmap)**
   - Metric scripts to compute descriptor alignment, colour drift, and motion smoothness.
   - Visualisation utilities for discovered keypoints and generated motion traces.

## Data Flow

```text
portrait + driver clip → dataset loader → motion descriptor extraction (planned) → animator training → checkpoints & metrics
                                                      ↓
                                        inference portraits / drivers → generated Live2D frames
```

The architecture emphasises a clean separation between motion understanding and image synthesis so each can evolve with new research.

## Extensibility
- **Descriptor plugins** – swap in alternate keypoint/feature extractors by conforming to a simple interface (tensor out, metadata on coverage/confidence).
- **Appearance encoders** – experiment with different image backbones (ConvNeXt, ViT) for portrait embeddings without touching the motion modules.
- **Training recipes** – register new loss functions or schedulers via configuration to evaluate ideas like diffusion-based refinement or perceptual metrics.

## Current Gaps
- Motion descriptors are still derived implicitly from RGB differences; the descriptor module is the next major milestone.
- No automated evaluation of motion alignment yet; manual inspection is required.
- Inference path does not yet accept external face videos—adapters need to be trained.

These notes will continue to evolve as the motion descriptor stack and evaluation tooling come online.
