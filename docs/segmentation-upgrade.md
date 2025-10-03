# Segmentation Upgrade Plan

This document outlines the steps for replacing the current MobileNetV2-based UNet with a stronger anime face segmentation stack and automating the label creation workflow.

## Goals
- Increase segmentation quality beyond SAM-HQ and the original MobileNetV2-UNet baseline.
- Produce high-resolution, semantically-labelled masks suitable for Live2D layer generation.
- Automate annotation as much as possible using vision-capable LLMs.

## 1. Backbone + Architecture Refresh
1. **Benchmark candidates**
   - EfficientNetV2 or ConvNeXt encoder backbones for 512×512 inputs.
   - Transformer-based options (Swin UNet, SegFormer) for long-range context.
   - Keep UNet-style skip connections for precise boundaries; consider hybrid UNet-Transformer models (e.g., UNet++ or UNetFormer).
2. **Modular design**
   - Wrap encoder/decoder choices behind a `segmentation` backend flag (`anime-unet`, `anime-transformer`, etc.).
   - Maintain ONNX / TorchScript export for integration with the pipeline.
3. **Training improvements**
   - Mixed precision training for large backbones.
   - Data augmentations beyond flips/rotations: color jitter, elastic transforms.
   - Losses: combination of Dice, Cross-Entropy, boundary-aware loss.

## 2. Automated Label Workflow
1. **Vision LLM Annotation**
   - Use GPT-4o or similar to produce initial semantic masks: provide prompts defining regions (hair, face, eyes, mouth, skin, clothes, accessories).
   - Run in batch over curated datasets (Danbooru portraits + in-house references).
2. **Post-processing**
   - Convert LLM outputs into clean label maps (palette to class IDs).
   - Apply morphological cleanup, ensure masks align with image resolution.
3. **Human-in-the-loop Verification**
   - Spot-check samples; build a quick correction UI if accuracy < desired threshold.
   - Use corrected masks to fine-tune the LLM prompt / post-processing rules.
4. **Dataset Assembly**
   - Persist in `datasets/anime_segmentation/v1` with train/val/test splits.
   - Version with metadata (source, annotation method, quality flags).

## 3. Integration with CubiAI
1. **Backend switch**
   - Add `segmentation.backend: anime-advanced` in config to load the new model weights (`models/anime_advanced_seg.pt`).
   - Update segmentation stage to route to the chosen backend and return masks + class names.
2. **PNG/PSD export alignment**
   - Map each semantic class to layer filenames (`png/hair_001.png`, `png/eyesL_001.png`, etc.).
   - Update PSD rebuild logic to stack semantic layers in predictable order.
3. **Fallbacks**
   - Keep SAM-HQ and SLIC as fallback choices for non-face images.
   - Provide warnings when backend confidence is low.

## 4. Roadmap
- **Phase 1:** Prototype LLM-assisted annotation on 200–500 samples from Danbooru2019 or Danbooru2023 shards; evaluate accuracy.
  - Use `data/prepare_danbooru2019.py` (Kaggle portraits) or `data/extract_danbooru2023_subset.py` (HF shard) to draw seed batches.
- **Phase 2:** Train upgraded segmentation model and compare against MobileNetV2 baseline.
- **Phase 3:** Integrate into CubiAI pipeline with configuration toggles and documentation.
- **Phase 4:** Optional disocclusion / inpainting module leveraging semantic masks.

## References
- siyeong0/Anime-Face-Segmentation (MobileNetV2 baseline).
- NVIDIA SPADE / GauGAN for potential future synthesis or augmentation.
- Recent UNet derivatives: UNetFormer, SegFormer, EfficientUNetV2.
