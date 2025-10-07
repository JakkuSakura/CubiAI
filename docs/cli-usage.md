# CLI Usage Guide

The Typer CLI exposes the training and inference utilities required to experiment with the Animator model. All commands can be invoked through `uv run cubiai ...`.

## Installation
```bash
uv sync
uv run cubiai --help
```

## `cubiai model train`
Train the Animator on a portrait/driver dataset.

```bash
uv run cubiai model train ./dataset ./runs/animator \
    --size 1024 --steps 2000 --batch 1 \
    --lambda-align 0.3 --lambda-motion 0.1 \
    --device cuda
```

**Arguments**
- `data_root`: Root directory containing portrait PNGs and `<name>_video/` frame folders.
- `workdir`: Output directory for checkpoints and previews (defaults to `runs/animator`).

**Options**
- `--size`: Crop/resize resolution for training samples (default `1024`).
- `--steps`: Optional total optimisation steps. When provided, training (and resume) continues until this budget is consumed.
- `--epochs`: Total dataset passes to target when `--steps` is omitted (defaults to `1`).
- `--batch`: Mini-batch size (defaults to `1`).
- `--lr`: Learning rate for AdamW.
- `--lambda-align`, `--lambda-motion`: Loss weights controlling driver alignment and motion smoothness trade-offs.
- `--device`: Torch device string (`cuda`, `mps`, `cpu`).
- `--num-workers`: Dataloader workers.

Training prints metrics every 50 steps and saves `animator.pt` plus a preview render under the workdir. The CLI automatically resumes training from this checkpoint—restoring the optimizer state and continuing until the requested step or epoch budget is met.

## `cubiai model infer`
Render a single portrait/driver pair using a saved checkpoint.

```bash
uv run cubiai model infer portrait.png driver.png \
    --checkpoint ./runs/animator/animator.pt \
    --output ./outputs/preview.png \
    --strength 0.8 --size 1024
```

**Options**
- `--checkpoint`: Path to the trained weights.
- `--strength`: Scales deformation intensity (1.0 = training strength).
- `--driver-domain`: Integer domain id (placeholder for future multi-domain embeddings).
- `--output`: Destination PNG.
- `--device`: Torch device string.

## Environment Variables
No external credentials are required today. Future integrations (e.g., remote descriptor services) will document additional environment variables here.

## Tips
- Use `uv run cubiai model train ... --steps <small>` for quick smoke tests.
- Keep datasets on fast storage; each frame is loaded from disk per step.
- Enable mixed precision manually by wrapping the training loop once the PyTorch version is confirmed.
- After large refactors, regenerate previews (`cubiai model infer`) to ensure checkpoints still load correctly.

This guide will expand as we add descriptor extraction commands, evaluation scripts, and GUI hooks.
