"""Dataclasses describing intermediate artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


@dataclass(slots=True)
class LayerArtifact:
    name: str
    image: Image.Image
    mask: Image.Image
    bbox: tuple[int, int, int, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bbox": self.bbox,
            "metadata": self.metadata,
        }
