# Model & Descriptor Options

The project intentionally keeps model choices modular so different motion descriptors or generators can be evaluated quickly.

## Motion Descriptor Candidates
- **Unsupervised keypoints** – MONKEY-Net, First Order Motion Model, Jakab et al. equivariant landmarks. Pros: no labels; Cons: may need domain-specific augmentation to stabilise on anime.
- **Semantic face maps** – Lightweight face-parsing networks trained on stylised data (or adapters on top of human parsers) to output dense region maps.
- **Vision transformers** – DINOv2 or ViT-S features pooled over facial regions to capture motion without explicit keypoints.

## Domain Adapters (Roadmap)
- Learn small MLP/CNN heads that map human landmark detectors (e.g., MediaPipe Face Mesh) into the canonical descriptor space discovered from Live2D clips.
- Use contrastive losses or cycle consistency between paired motions (if available) to align distributions.

## Animator Backbone
- Current implementation: residual U-Net with FiLM conditioning on portrait embeddings.
- Alternatives under consideration: ConvNeXt blocks, diffusion decoders, NeRF-style volumetric heads for higher fidelity or 3D-aware motion.

## Loss Design
- Reconstruction (`L1`) remains the primary supervision.
- Motion loss will migrate from blurred RGB residuals to descriptor deltas to avoid penalising valid displacements.
- Identity loss keeps deformation bounded when motion should be minimal.
- Regularisers: total variation on residuals, optional perceptual loss using shallow VGG/DINO features for colour stability.

## Training Utilities
- Optimiser: AdamW with gradient clipping (default 1.0).
- Scheduler: none yet; plan to experiment with cosine decay and warm restarts once descriptor-conditioned training stabilises.
- Mixed precision: not enabled yet—monitor memory before introducing it.

Future revisions of this document will catalogue exact checkpoints, hyper-parameters, and adapters once the descriptor stack is implemented.
