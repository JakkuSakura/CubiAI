# Training & Inference Pipeline

This document outlines the current motion-transfer pipeline and the upgrades planned to support cross-domain drivers and limited-data scenarios.

## 1. Data Preparation
- Organise portraits and driver clips under `dataset/<name>.png` with `<name>_video/####.png` frames.
- Apply optional augmentations (flip, jitter, temporal subsampling) offline to enlarge the motion repertoire.
- Future work: precompute motion descriptors and cache them to disk for reproducible experiments.

## 2. Motion Descriptor Extraction (roadmap)
- Train an unsupervised keypoint or dense descriptor model on the Live2D clips.
- Align external domains (human faces, parser output) into the same descriptor space using small adapters.
- Normalise descriptors (scale, rotation, canonical ordering) before feeding them to the animator.

## 3. Animator Training
1. Load batches via `PortraitVideoDataset`.
2. Forward pass through the `Animator` with the static portrait and driver frame (descriptor tensors will be added here).
3. Compute losses:
   - Reconstruction against the driver frame (`L1`).
   - Alignment via blurred residual comparisons to the driver (to be replaced with descriptor deltas).
   - Identity consistency by driving the portrait with itself at reduced strength.
   - Regularisation (total variation on alignment residuals, gradient clipping).
4. Optimise with AdamW; metrics are logged every 50 steps.

## 4. Checkpointing & Outputs
- Model weights are stored under the training workdir (default `pass_through.pt`, slated for renaming).
- Metrics dictionary includes reconstruction, alignment loss, motion smoothness, and residual magnitude.
- Planned additions: descriptor alignment scores, Fréchet-style metrics, preview GIF renders.

## 5. Inference
- Load the saved checkpoint, portrait, and driver frame/clip.
- (Future) Extract descriptors for the driver and feed them into the animator.
- Save generated frames to disk; upcoming utilities will stitch sequences into videos and overlay diagnostic plots.

## 6. Future Enhancements
- **Descriptor caches** – Persist motion features alongside RGB frames for faster experiments.
- **Adapter training scripts** – Align human landmark detectors to the canonical descriptor space using contrastive or cycle losses.
- **Evaluation suite** – Command-line tooling to measure temporal coherence, descriptor agreement, and colour drift.
- **GUI preview** – Revive the PySide plan as a lightweight viewer for generated sequences and keypoint overlays.

The pipeline is intentionally modular so each stage—descriptor extraction, appearance encoding, loss design—can evolve independently as new data or ideas arrive.
