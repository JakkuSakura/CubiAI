"""Local SAM-HQ segmentation utilities."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

from ..errors import MissingDependencyError, PipelineStageError
from ..pipeline.artifacts import LayerArtifact


@dataclass
class SamHQConfig:
    model_id: str
    device: str | None = None
    max_layers: int = 32
    score_threshold: float = 0.0


class SamHQLocalSegmenter:
    """Run SAM-HQ locally via Hugging Face Transformers."""

    def __init__(self, config: SamHQConfig) -> None:
        self.config = config
        try:
            import torch  # noqa: F401
            from transformers import SamHQModel, SamHQProcessor  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingDependencyError(
                "transformers[torch] is required for the sam-hq-local backend. "
                "Install with `uv add torch transformers`."
            ) from exc

    def segment(self, image: Image.Image) -> List[LayerArtifact]:
        torch, model, processor, device = _load_sam_hq(
            model_id=self.config.model_id,
            device_override=self.config.device,
        )

        torch_image = image.convert("RGB")
        points = _generate_prompt_grid(torch_image.size)

        layers: list[LayerArtifact] = []
        seen_masks: list[np.ndarray] = []

        for point in points:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"The following named arguments are not valid for `SamImageProcessor.preprocess`",
                    category=UserWarning,
                )
                inputs = processor(
                    torch_image,
                    input_points=[[point]],
                    return_tensors="pt",
                ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            post_masks = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )
            mask_stack = post_masks[0].detach().cpu().numpy()
            if mask_stack.ndim >= 3:
                mask_stack = mask_stack.reshape(-1, mask_stack.shape[-2], mask_stack.shape[-1])
            elif mask_stack.ndim == 2:
                mask_stack = mask_stack[np.newaxis, ...]

            scores = (
                outputs.iou_scores[0]
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

            for idx, score in enumerate(scores):
                if idx >= mask_stack.shape[0]:
                    break
                mask_array = mask_stack[idx]
                if score < self.config.score_threshold:
                    continue
                if not mask_array.any():
                    continue
                if _is_duplicate(mask_array, seen_masks):
                    continue

                artifact = _build_layer(mask_array, torch_image, score)
                if artifact is None:
                    continue
                layers.append(artifact)
                seen_masks.append(mask_array > 0.5)

                if len(layers) >= self.config.max_layers:
                    return layers

        if not layers:
            raise PipelineStageError(
                stage="Segmentation",
                message="SAM-HQ did not produce any usable masks. Try adjusting prompts or thresholds.",
            )

        layers.sort(key=lambda item: item.metadata.get("score", 0.0), reverse=True)
        return layers


@lru_cache(maxsize=2)
def _load_sam_hq(model_id: str, device_override: str | None):
    import torch
    from transformers import SamHQModel, SamHQProcessor

    if device_override:
        device = torch.device(device_override)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = SamHQProcessor.from_pretrained(model_id)
    model = SamHQModel.from_pretrained(model_id)
    model = model.to(device)
    model.eval()

    return torch, model, processor, device


def _generate_prompt_grid(size: Tuple[int, int], grid: int = 4) -> List[List[int]]:
    width, height = size
    xs = np.linspace(0.1, 0.9, grid)
    ys = np.linspace(0.1, 0.9, grid)
    points: list[list[int]] = []
    for y in ys:
        for x in xs:
            points.append([int(x * width), int(y * height)])
    # ensure centre is included
    points.append([width // 2, height // 2])
    return points


def _is_duplicate(mask: np.ndarray, existing: Iterable[np.ndarray], threshold: float = 0.9) -> bool:
    mask_bool = mask > 0.5
    mask_area = mask_bool.sum()
    if mask_area == 0:
        return True
    for candidate in existing:
        intersection = np.logical_and(mask_bool, candidate).sum()
        union = np.logical_or(mask_bool, candidate).sum()
        if union == 0:
            continue
        iou = intersection / union
        if iou >= threshold:
            return True
    return False


def _build_layer(mask: np.ndarray, image: Image.Image, score: float) -> LayerArtifact | None:
    mask_bool = mask > 0.5
    if not mask_bool.any():
        return None

    bbox = _bbox_from_mask(mask_bool)
    if bbox is None:
        return None

    mask_image = Image.fromarray((mask_bool * 255).astype("uint8"), mode="L")
    transparent = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer_image = Image.composite(image.convert("RGBA"), transparent, mask_image)

    area = int(mask_bool.sum())
    score_tag = int(score * 1000)
    return LayerArtifact(
        name=f"sam_hq_{score_tag:03d}_{area:06d}",
        image=layer_image,
        mask=mask_image,
        bbox=bbox,
        metadata={
            "score": float(score),
            "area": area,
        },
    )


def _bbox_from_mask(mask_bool: np.ndarray) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_bool)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return int(x0), int(y0), int(x1 + 1), int(y1 + 1)
