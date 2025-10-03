"""Facilities for writing layered PSD files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image

from ..errors import MissingDependencyError
from ..pipeline.artifacts import LayerArtifact
from ..workspace import Workspace
from psd_tools import PSDImage
from psd_tools.api.layers import Group, PixelLayer


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

        canvas_size = layers[0].image.size
        psd = PSDImage.new(mode="RGBA", size=canvas_size, color=(0, 0, 0, 0))
        group = Group(name=self.group_name)

        for artifact in layers:
            pil_image = artifact.image
            if pil_image.size != canvas_size:
                pil_image = pil_image.resize(canvas_size, Image.LANCZOS)
            pixel_layer = PixelLayer.from_image(pil_image, name=artifact.name)
            group.append(pixel_layer)

        psd.append(group)
        output_path = self.workspace.layers_dir / "layers.psd"
        psd.save(output_path)
        return output_path
