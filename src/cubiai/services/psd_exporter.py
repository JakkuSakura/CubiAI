"""Facilities for writing layered PSD files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image

from ..errors import MissingDependencyError
from ..pipeline.artifacts import LayerArtifact
from ..workspace import Workspace


def _create_pixel_layer(psd_layers, image: Image.Image, name: str):
    if hasattr(psd_layers, "PixelLayer") and hasattr(psd_layers.PixelLayer, "from_pil"):
        return psd_layers.PixelLayer.from_pil(image, name=name)
    if hasattr(psd_layers, "Layer") and hasattr(psd_layers.Layer, "from_pil"):
        return psd_layers.Layer.from_pil(image, name=name)
    raise RuntimeError("psd-tools API incompatible: expected PixelLayer.from_pil")


class PSDExporter:
    """Export a collection of layers into a layered PSD document."""

    def __init__(self, workspace: Workspace, group_name: str = "CubiAI Layers") -> None:
        self.workspace = workspace
        self.group_name = group_name

    def export(self, layers: Iterable[LayerArtifact]) -> Path:
        layers = list(layers)
        if not layers:
            msg = "No layers available to export as PSD"
            raise ValueError(msg)

        try:
            from psd_tools.user_api import layers as psd_layers
            from psd_tools.user_api import psd_image
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise MissingDependencyError(
                "psd-tools is required to export PSD files. Install with `uv add psd-tools`."
            ) from exc

        canvas_size = layers[0].image.size
        psd = psd_image.PSDImage.new(mode="RGBA", size=canvas_size, color=(0, 0, 0, 0))
        group = psd_layers.Group(name=self.group_name)

        for artifact in layers:
            pil_image = artifact.image
            if pil_image.size != canvas_size:
                pil_image = pil_image.resize(canvas_size, Image.LANCZOS)
            pixel_layer = _create_pixel_layer(psd_layers, pil_image, artifact.name)
            group.append(pixel_layer)

        psd.append(group)
        output_path = self.workspace.layers_dir / "layers.psd"
        psd.save(output_path)
        return output_path
