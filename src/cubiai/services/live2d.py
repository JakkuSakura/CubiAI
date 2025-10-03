"""Helpers to assemble a minimal Live2D-compatible project layout."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from PIL import Image

from ..config.models import Live2DSettings, RiggingSettings
from ..errors import PipelineStageError
from ..pipeline.artifacts import LayerArtifact
from ..workspace import Workspace


@dataclass(slots=True)
class AtlasPlacement:
    name: str
    texture_index: int
    position: Tuple[int, int]
    size: Tuple[int, int]

    def to_dict(self, texture_size: int) -> dict[str, Any]:
        x, y = self.position
        w, h = self.size
        return {
            "name": self.name,
            "texture_index": self.texture_index,
            "position": [x, y],
            "size": [w, h],
            "uv": [x / texture_size, y / texture_size, (x + w) / texture_size, (y + h) / texture_size],
        }


class Live2DExporter:
    """Create a Live2D project directory with textures, rig metadata, and motions."""

    def __init__(self, workspace: Workspace, settings: Live2DSettings, rigging: RiggingSettings) -> None:
        self.workspace = workspace
        self.settings = settings
        self.rigging = rigging

    def export(self, layers: Iterable[LayerArtifact], rig_data: dict[str, Any]) -> Dict[str, Path]:
        layers = list(layers)
        if not layers:
            msg = "No layers available for Live2D export"
            raise PipelineStageError(stage="Live2DExport", message=msg)

        dest = self.workspace.live2d_dir
        textures_dir = dest / "Resources" / "Textures"
        motions_dir = dest / "Motions"
        physics_dir = dest / "Physics"
        meta_dir = dest / "Meta"
        sources_dir = dest / "Sources"

        for directory in (textures_dir, motions_dir, physics_dir, meta_dir, sources_dir):
            directory.mkdir(parents=True, exist_ok=True)

        atlas_image, placements = build_texture_atlas(
            layers=layers,
            texture_size=self.settings.texture_size,
            padding=self.settings.padding,
            background_color=self.settings.background_color,
        )
        texture_path = textures_dir / "texture_00.png"
        atlas_image.save(texture_path)

        rig_path = meta_dir / "rig.json"
        if not rig_path.exists():
            rig_path.write_text(json.dumps(rig_data, indent=2))

        physics_payload = {
            "Version": 3,
            "Meta": {"PhysicsSettingCount": 0, "TotalInputCount": 0, "TotalOutputCount": 0},
            "PhysicsSettings": [],
        }
        physics_path = physics_dir / "model.physics3.json"
        if not physics_path.exists():
            physics_path.write_text(json.dumps(physics_payload, indent=2))

        motion_payload = create_idle_motion()
        motion_path = motions_dir / "idle.motion3.json"
        if not motion_path.exists() and self.rigging.idle_motion:
            motion_path.write_text(json.dumps(motion_payload, indent=2))

        atlas_manifest = [placement.to_dict(self.settings.texture_size) for placement in placements]
        manifest_path = meta_dir / "atlas.json"
        manifest_path.write_text(json.dumps(atlas_manifest, indent=2))

        moc_path = dest / "model.moc3"
        if not moc_path.exists():
            raise PipelineStageError(
                stage="Live2DExport",
                message="model.moc3 not found. Ensure the rigging builder produced a valid Live2D Core asset.",
            )

        model_payload = create_model3_manifest(
            texture_relative=str(texture_path.relative_to(dest)),
            motion_relative=str(motion_path.relative_to(dest)),
            physics_relative=str(physics_path.relative_to(dest)),
            moc_relative=str(moc_path.relative_to(dest)),
            rigging=self.rigging,
            placements=atlas_manifest,
        )
        model_path = dest / "model3.json"
        model_path.write_text(json.dumps(model_payload, indent=2))

        # Copy PSD into sources directory if it exists.
        psd_source = self.workspace.layers_dir / "layers.psd"
        if psd_source.exists():
            psd_target = sources_dir / "layers.psd"
            if psd_target != psd_source:
                psd_target.write_bytes(psd_source.read_bytes())

        outputs = {
            "live2d": dest,
            "texture_atlas": texture_path,
            "model3": model_path,
            "rig": rig_path,
            "physics": physics_path,
            "atlas_manifest": manifest_path,
        }
        if motion_path.exists():
            outputs["motion_idle"] = motion_path
        return outputs


def build_texture_atlas(
    *,
    layers: Iterable[LayerArtifact],
    texture_size: int,
    padding: int,
    background_color: tuple[int, int, int, int],
) -> tuple[Image.Image, list[AtlasPlacement]]:
    """Pack layers into a simple atlas using a row-based packing strategy."""
    atlas = Image.new("RGBA", (texture_size, texture_size), color=background_color)
    placements: list[AtlasPlacement] = []

    x_offset = padding
    y_offset = padding
    row_height = 0

    for layer in layers:
        cropped = layer.image.crop(layer.bbox)
        width, height = cropped.size

        if width <= 0 or height <= 0:
            continue

        if width + padding * 2 > texture_size:
            raise PipelineStageError(
                stage="Live2DExport",
                message=f"Layer {layer.name} width {width}px exceeds atlas size {texture_size}px",
            )
        if x_offset + width + padding > texture_size:
            x_offset = padding
            y_offset += row_height + padding
            row_height = 0

        if y_offset + height + padding > texture_size:
            raise PipelineStageError(
                stage="Live2DExport",
                message="Texture atlas overflow – increase texture_size in the configuration.",
            )

        atlas.paste(cropped, (x_offset, y_offset), mask=cropped.split()[-1])
        placements.append(
            AtlasPlacement(
                name=layer.name,
                texture_index=0,
                position=(x_offset, y_offset),
                size=(width, height),
            )
        )

        x_offset += width + padding
        row_height = max(row_height, height)

    return atlas, placements


def create_model3_manifest(
    *,
    texture_relative: str,
    motion_relative: str,
    physics_relative: str,
    moc_relative: str,
    rigging: RiggingSettings,
    placements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Construct a minimal model3.json manifest."""
    return {
        "Version": 3,
        "FileReferences": {
            "Moc": moc_relative,
            "Textures": [texture_relative],
            "Motions": {
                "Idle": [
                    {
                        "File": motion_relative,
                        "FadeInTime": 0.8,
                        "FadeOutTime": 0.8,
                    }
                ]
            },
            "Physics": physics_relative,
            "UserData": "Meta/rig.json",
        },
        "Groups": [
            {
                "Target": "Parameter",
                "Name": "EyeBlink",
                "Ids": list(rigging.blink_parameters),
            }
        ],
        "HitAreas": [],
        "Parts": placements,
    }


def create_idle_motion() -> dict[str, Any]:
    """Return a lightweight idle motion definition."""
    return {
        "Version": 3,
        "Meta": {"Duration": 2.5, "Fps": 30, "Loop": True},
        "Curves": [
            {
                "Target": "Parameter",
                "Id": "ParamAngleX",
                "Segments": [0.0, 0.0, 1, 0.0, 1.25, 10.0, 2, 0.0, 2.5, -10.0, 2, 0.0],
            },
            {
                "Target": "Parameter",
                "Id": "ParamAngleY",
                "Segments": [0.0, 0.0, 1, 0.0, 1.0, 6.0, 2, 0.0, 2.0, -6.0, 2, 0.0],
            },
            {
                "Target": "Parameter",
                "Id": "ParamBodyAngleZ",
                "Segments": [0.0, 0.0, 1, 0.0, 1.25, 4.0, 2, 0.0, 2.5, -4.0, 2, 0.0],
            },
        ],
        "FadeInTime": 0.5,
        "FadeOutTime": 0.5,
        "Events": [],
    }
