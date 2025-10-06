# Motion Descriptor Upgrade Plan

Large-scale annotated anime datasets are rare, so CubiAI will rely on self-supervised motion cues and lightweight adapters. This plan tracks the steps required to replace RGB residual cues with robust motion descriptors.

## Goals
- Discover canonical facial motion landmarks from a handful of Live2D clips.
- Align human face motion and parser outputs to the same descriptor space.
- Feed descriptors directly into the Animator for sharper, motion-consistent animation.

## 1. Descriptor Discovery
1. **Model survey** – Evaluate unsupervised keypoint methods (FOMM, KeyNet) and dense contrastive features (DINO, LoFTR) on Live2D clips.
2. **Stability tricks** – Apply heavy augmentations (flip, colour jitter, temporal smoothing) to keep keypoints consistent across stylised frames.
3. **Implementation** – Wrap the chosen method in a reusable module (`cubiai.motion.descriptors`) returning tensors + confidence maps.

## 2. Domain Alignment
1. **Human driver adapter** – Map MediaPipe Face Mesh landmarks to the canonical descriptor via a small MLP trained with cycle or contrastive losses.
2. **Parser adapter** – When parser outputs arrive, add a CNN head that projects segmentation maps to the same latent space.
3. **Consistency checks** – Train with a small paired set (e.g., your face + Live2D clip) to ensure descriptors move together; regularise with temporal smoothness.

## 3. Integration with Animator
1. Concatenate or cross-attend descriptor tensors with portrait embeddings inside the Animator.
2. Replace colour-based motion loss with descriptor delta loss (`||Δ_pred - Δ_drv||`).
3. Add augmentation that jitters descriptors during training to encourage robustness.

## 4. Evaluation
- Metrics: cosine similarity between predicted vs. driver descriptors, endpoint error for keypoints, temporal jitter statistics.
- Visual tools: overlay descriptors on frames, export side-by-side comparisons.
- Human evaluation: quick survey forms or in-app tagging to label satisfying motion transfers.

## 5. Roadmap
- **Phase A:** Prototype unsupervised keypoint discovery on existing Live2D clips, log qualitative stability.
- **Phase B:** Train human-face adapter using limited paired data or synthetic alignment.
- **Phase C:** Integrate descriptors into training loop and benchmark against RGB-only baseline.
- **Phase D:** Ship evaluation scripts and GUI overlays to diagnose descriptor quality.

Progress here unlocks the ability to animate entirely new characters using motion captured from any domain.
