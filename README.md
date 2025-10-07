# CubiAI

CubiAI explores few-shot motion transfer for Live2D-style portraits. The current focus is learning how stylised characters should move from a small collection of reference clips, then retargeting real-face (or parser-produced) motion to entirely new characters at native resolution.

## Project Goals
- **Cross-domain motion descriptors** – Extract or learn canonical facial motion cues that work for anime, real portraits, and potential face-parsing maps without dense labels.
- **Appearance-conditioned animator** – Generate high-resolution Live2D frames by conditioning on a static portrait while following motion signals, keeping colour fidelity and structural detail.
- **Tiny-data viability** – Operate with roughly a dozen Live2D clips plus optional scripted calibration sweeps, using self-supervised keypoints, augmentation, and lightweight adapters.
- **Practical tooling** – Provide CLI workflows for training, evaluation, and future face-to-Live2D inference, with documentation on data layout and configuration.

## Current Capabilities
- Portrait/video dataset loader that expects each portrait alongside a folder of driver frames.
- `Animator` model that applies motion to a still portrait via a residual U-Net with latent FiLM conditioning.
- Training loop with reconstruction, motion-alignment, and identity consistency losses designed to discourage colour drift while rewarding driver-aligned movement.
- Typer-based CLI (`cubiai model train` / `cubiai model infer`) for quick experimentation.

## Roadmap Highlights
1. **Canonical motion extractor** – Train an unsupervised keypoint or semantic map network on available Live2D clips and map human-face landmarks into the same space via a small adapter.
2. **Descriptor-conditioned animator** – Integrate descriptor tensors directly so the generator reacts to motion rather than raw RGB differences.
3. **Few-shot character adaptation** – Explore appearance encoders and fast fine-tuning so brand-new Live2D characters can borrow learned motion without full retraining.
4. **Evaluation tooling** – Add motion-alignment metrics, colour-drift scores, and qualitative preview scripts for new drivers/characters.

## Repository Layout
```
README.md           Project overview and roadmap
src/                Core library (models, CLI commands, data loader)
docs/               Architecture notes, pipeline plans, model references
config/             Example YAML configs for training and experiments
```

## Data Expectations
- Place portrait/driver pairs under a dataset root:
  ```
  dataset/
    alice.png
    alice_video/00000000.png
    alice_video/00000001.png
    ...
  ```
- Portrait filenames must match the driver directory stem (`<name>_video`).
- Additional characters can be added the same way; short scripted sweeps are encouraged to widen motion coverage.

## Quick Start
1. Install dependencies with [uv](https://github.com/astral-sh/uv):
   ```bash
   uv sync
   ```
2. Train the animator:
   ```bash
   uv run cubiai model train ./dataset ./runs/animator \
       --size 1024 --epochs 10 --batch 1 \
       --lambda-align 0.3 --lambda-motion 0.1
   ```
3. Run inference on a portrait/driver pair:
   ```bash
   uv run cubiai model infer portrait.png driver.png \
       --checkpoint ./runs/animator/animator.pt \
       --output ./outputs/preview.png
   ```

The CLI saves intermediate metrics every 50 steps and writes the trained weights to the specified `workdir`. Rerunning the train command automatically resumes from the saved checkpoint (including optimizer state) and continues until the requested budget is reached (`--epochs`, default `10`, or `--steps` when supplied).

Add `--steps <total>` if you prefer to cap optimisation by step count rather than epochs.

## Key Concepts
- **Motion Descriptor Alignment** – Design or learn a representation that captures head pose and expression changes uniformly across domains (anime, real, parser output). Future releases will replace raw RGB conditioning with these descriptors.
- **Appearance Conditioning** – Encode the static portrait to modulate generator layers (FiLM/AdaIN) so the animator respects each character’s structure and style.
- **Motion-Regularised Losses** – Compare predicted motion against driver motion in a blur-tolerant space and regularise with total variation to avoid scattered artefacts.
- **Limited Data Strategy** – Combine unsupervised keypoint discovery, heavy augmentation, and optional synthetic rig sweeps to overcome scarce labelled anime data.

## Documentation
- `docs/architecture.md` – Updated component diagram reflecting motion descriptor extraction, adapters, and the animator.
- `docs/pipeline.md` – Training/inference pipeline and planned extensions.
- `docs/ai-models.md` – Notes on unsupervised keypoint models, descriptor adapters, and candidate feature backbones.
- Additional documents cover CLI usage and GUI roadmap placeholders from earlier phases.

## Contributing
Contributions are welcome—from motion descriptor research to tooling improvements.
1. Use uv for dependency management (`uv sync`).
2. Open issues for proposed architectural changes or new training scripts.
3. Document new configuration flags and update the docs accordingly.

## License
CubiAI is released under the MIT License (see `LICENSE`).
