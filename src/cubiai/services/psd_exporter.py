"""Facilities for writing layered PSD files."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image
from psd_tools import PSDImage
from psd_tools.api.layers import PixelLayer
from psd_tools.constants import Compression

from ..pipeline.artifacts import LayerArtifact
from ..workspace import Workspace


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

        for artifact in layers:
            pil_image = artifact.image
            if pil_image.size != canvas_size:
                pil_image = pil_image.resize(canvas_size, Image.LANCZOS)

            x0, y0, x1, y1 = artifact.bbox
            cropped = pil_image.crop((x0, y0, x1, y1))
            layer = PixelLayer.frompil(
                cropped,
                psd,
                artifact.name,
                top=y0,
                left=x0,
                compression=Compression.RLE,
            )
            layer.name = artifact.name
            psd.append(layer)

        output_path = self.workspace.layers_dir / "layers.psd"
        psd.save(output_path)
        return output_path
