"""Hugging Face segmentation client."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import httpx
from PIL import Image

from ..errors import PipelineStageError
from ..pipeline.artifacts import LayerArtifact


@dataclass(slots=True)
class HFSAMSegmenter:
    """Invoke a Hugging Face SAM endpoint to obtain segmentation masks."""

    model: str
    endpoint: str | None
    token: str
    max_layers: int
    score_threshold: float = 0.0
    timeout_seconds: int = 120

    def segment(self, image: Image.Image) -> list[LayerArtifact]:
        url = self.endpoint or f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {"Authorization": f"Bearer {self.token}"}

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        payload = buffer.getvalue()

        try:
            response = httpx.post(url, headers=headers, content=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - network dependent
            detail = exc.response.text
            raise PipelineStageError(
                stage="Segmentation",
                message=f"Hugging Face segmentation request failed: {detail}",
            ) from exc
        except httpx.HTTPError as exc:  # pragma: no cover - network dependent
            raise PipelineStageError(stage="Segmentation", message=str(exc)) from exc

        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            raise PipelineStageError(stage="Segmentation", message=str(data.get("error")))
        if not isinstance(data, list):
            raise PipelineStageError(stage="Segmentation", message="Unexpected response format from Hugging Face API")

        transparent = Image.new("RGBA", image.size, (0, 0, 0, 0))
        layers: list[LayerArtifact] = []

        for index, entry in enumerate(data):
            score = float(entry.get("score", 0.0) or 0.0)
            if score < self.score_threshold:
                continue
            mask_b64 = entry.get("mask")
            if not mask_b64:
                continue
            mask_bytes = base64.b64decode(mask_b64)
            mask_image = Image.open(io.BytesIO(mask_bytes)).convert("L")
            bbox = mask_image.getbbox()
            if bbox is None:
                continue

            layer_image = Image.composite(image, transparent, mask_image)
            layer_name = entry.get("label") or f"segment_{index:02d}"
            area = sum(1 for value in mask_image.getdata() if value > 0)

            artifact = LayerArtifact(
                name=layer_name,
                image=layer_image,
                mask=mask_image,
                bbox=bbox,
                metadata={
                    "score": score,
                    "area": area,
                },
            )
            layers.append(artifact)
            if len(layers) >= self.max_layers:
                break

        layers.sort(key=lambda item: float(item.metadata.get("score", 0.0)), reverse=True)
        return layers
