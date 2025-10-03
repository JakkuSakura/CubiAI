"""Concrete pipeline stage implementations."""
from __future__ import annotations

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Iterable, List

import numpy as np
from PIL import Image

from ..config.models import AppConfig, Live2DSettings, RiggingSettings, SegmentationSettings
from ..errors import MissingDependencyError, PipelineStageError
from ..pipeline.artifacts import LayerArtifact
from ..services.hf_segmentation import HFSAMSegmenter
from ..services.sam_hq_local import SamHQLocalSegmenter, SamHQConfig
from ..services.llm_rigging import LLMRiggingClient, RiggingLLMError
from ..services.live2d import Live2DExporter
from ..services.psd_exporter import PSDExporter
from ..workspace import Workspace
from .context import PipelineContext


class PipelineStage(ABC):
    name: str

    @abstractmethod
    def run(self, ctx: PipelineContext) -> None:  # pragma: no cover - runtime behaviour
        """Execute the stage."""


class ImageLoaderStage(PipelineStage):
    name = "ImageLoader"

    def __init__(self, workspace: Workspace) -> None:
        self.workspace = workspace
        self.logger = logging.getLogger(self.name)

    def run(self, ctx: PipelineContext) -> None:
        image = Image.open(self.workspace.input_path).convert("RGBA")
        ctx.data["image"] = image
        ctx.data["canvas_size"] = image.size
        ctx.record_diagnostic(self.name, {"size": image.size, "mode": image.mode})


class SegmentationStage(PipelineStage):
    name = "Segmentation"

    def __init__(self, settings: SegmentationSettings, workspace: Workspace) -> None:
        self.settings = settings
        self.workspace = workspace
        self.logger = logging.getLogger(self.name)

    def run(self, ctx: PipelineContext) -> None:
        image: Image.Image = ctx.data["image"]

        if self.settings.backend == "huggingface-sam":
            layers = self._run_huggingface_stage(image)
        elif self.settings.backend == "sam-hq-local":
            layers = self._run_sam_hq_local(image)
        else:
            layers = self._run_slic_stage(image)

        if not layers:
            raise PipelineStageError(stage=self.name, message="Segmentation produced no layers")

        ctx.data["layers"] = layers
        ctx.record_diagnostic(
            self.name,
            {
                "layer_count": len(layers),
                "settings": self.settings.model_dump(),
                "backend": self.settings.backend,
            },
        )

    def _run_huggingface_stage(self, image: Image.Image) -> List[LayerArtifact]:
        if self.settings.huggingface is None:
            raise PipelineStageError(
                stage=self.name,
                message="Hugging Face configuration missing for huggingface-sam backend.",
            )

        cfg = self.settings.huggingface
        token = os.environ.get(cfg.token_env)
        if not token:
            raise PipelineStageError(
                stage=self.name,
                message=(
                    "Hugging Face token missing. Set the environment variable "
                    f"{cfg.token_env} with a valid token or switch to the `slic` backend."
                ),
            )

        segmenter = HFSAMSegmenter(
            model=cfg.model,
            endpoint=cfg.endpoint,
            token=token,
            max_layers=cfg.max_layers,
            score_threshold=cfg.score_threshold,
        )
        try:
            layers = segmenter.segment(image)
        except Exception as exc:  # noqa: BLE001
            raise PipelineStageError(stage=self.name, message=str(exc)) from exc

        self._persist_layers(layers)
        return layers

    def _run_sam_hq_local(self, image: Image.Image) -> List[LayerArtifact]:
        config = SamHQConfig(
            model_id=os.environ.get(
                "CUBIAI_SAM_HQ_MODEL",
                self.settings.sam_hq_local_model_id or "syscv-community/sam-hq-vit-base",
            ),
            device=os.environ.get("CUBIAI_SAM_HQ_DEVICE", self.settings.sam_hq_local_device),
            max_layers=self.settings.num_segments,
            score_threshold=self.settings.sam_hq_local_score_threshold,
        )
        segmenter = SamHQLocalSegmenter(config)
        layers = [
            layer
            for layer in segmenter.segment(image)
            if layer.metadata.get("area", 0) >= self.settings.min_area_px
        ]
        if not layers:
            raise PipelineStageError(
                stage=self.name,
                message="SAM-HQ produced masks but all were below the minimum area threshold.",
            )
        self._persist_layers(layers)
        return layers

    def _run_slic_stage(self, image: Image.Image) -> List[LayerArtifact]:
        try:
            from skimage import segmentation
            from scipy.ndimage import gaussian_filter
        except ImportError as exc:  # pragma: no cover - optional dependency missing
            raise MissingDependencyError(
                "scikit-image and scipy are required for segmentation. Install with `uv add scikit-image scipy`."
            ) from exc

        rgb = np.array(image.convert("RGB"))
        slic_segments = segmentation.slic(
            rgb,
            n_segments=self.settings.num_segments,
            compactness=self.settings.compactness,
            start_label=1,
        )

        layers = self._build_layers(
            image=image,
            segments=slic_segments,
            gaussian_filter=gaussian_filter,
        )

        self._persist_layers(layers)

        return layers

    def _build_layers(self, image: Image.Image, segments: np.ndarray, gaussian_filter) -> List[LayerArtifact]:
        layers: list[LayerArtifact] = []
        alpha = np.array(image.split()[-1])

        for idx, pixel_count in zip(*np.unique(segments, return_counts=True)):
            if pixel_count < self.settings.min_area_px:
                continue

            mask = (segments == idx).astype(float)
            if self.settings.feather_radius > 0:
                mask = gaussian_filter(mask, sigma=self.settings.feather_radius)
            mask = mask / (mask.max() or 1.0)
            mask_alpha = (mask * 255).astype("uint8")

            composite = np.array(image)
            composite_alpha = (alpha.astype(float) * (mask_alpha / 255.0)).clip(0, 255).astype("uint8")
            composite[..., 3] = composite_alpha
            layer_image = Image.fromarray(composite, mode="RGBA")
            mask_image = Image.fromarray(mask_alpha, mode="L")

            bbox = mask_image.getbbox()
            if bbox is None:
                continue

            name = f"layer_{int(idx):02d}"
            artifact = LayerArtifact(
                name=name,
                image=layer_image,
                mask=mask_image,
                bbox=bbox,
                metadata={"pixel_count": int(pixel_count)},
            )

            layers.append(artifact)

        layers.sort(key=lambda item: item.metadata.get("pixel_count", 0), reverse=True)
        return layers

    def _persist_layers(self, layers: Iterable[LayerArtifact]) -> None:
        for artifact in layers:
            safe_name = artifact.name.replace(" ", "_")
            layer_path = self.workspace.layers_dir / f"{safe_name}.png"
            artifact.image.crop(artifact.bbox).save(layer_path)
            if self.workspace.keep_intermediate:
                mask_path = self.workspace.masks_dir / f"{safe_name}_mask.png"
                artifact.mask.save(mask_path)


class PSDExportStage(PipelineStage):
    name = "PSDExport"

    def __init__(self, workspace: Workspace, enabled: bool, group_name: str) -> None:
        self.workspace = workspace
        self.enabled = enabled
        self.group_name = group_name
        self.logger = logging.getLogger(self.name)

    def run(self, ctx: PipelineContext) -> None:
        if not self.enabled:
            ctx.record_diagnostic(self.name, {"status": "skipped"})
            return
        layers: list[LayerArtifact] = ctx.data.get("layers", [])
        if not layers:
            raise PipelineStageError(stage=self.name, message="No layers available for PSD export")

        exporter = PSDExporter(workspace=self.workspace, group_name=self.group_name)
        psd_path = exporter.export(layers)
        ctx.record_output("psd", psd_path)
        ctx.record_diagnostic(
            self.name,
            {"path": str(psd_path), "layer_count": len(layers)},
        )


class RiggingStage(PipelineStage):
    name = "Rigging"

    def __init__(self, settings: RiggingSettings) -> None:
        self.settings = settings
        self.logger = logging.getLogger(self.name)

    def run(self, ctx: PipelineContext) -> None:
        layers: list[LayerArtifact] = ctx.data.get("layers", [])
        if not layers:
            raise PipelineStageError(stage=self.name, message="No layers available for rigging")
        rig_data = self._generate_rig(layers)
        ctx.data["rig"] = rig_data
        ctx.record_diagnostic(self.name, {"parts": len(rig_data.get("parts", []))})

        builder_outputs = self._run_builder(ctx=ctx, rig_data=rig_data)
        ctx.data["rig_builder"] = builder_outputs
        ctx.record_diagnostic(
            self.name,
            {
                "builder": builder_outputs,
            },
        )

    def _generate_rig(self, layers: Iterable[LayerArtifact]) -> dict[str, object]:
        if self.settings.strategy == "llm":
            return self._generate_rig_via_llm(layers)
        return self._generate_rig_heuristic(layers)

    def _generate_rig_via_llm(self, layers: Iterable[LayerArtifact]) -> dict[str, object]:
        api_key = os.environ.get(self.settings.llm_api_key_env)
        if not api_key:
            raise PipelineStageError(
                stage=self.name,
                message=(
                    "LLM API key missing. Set the environment variable "
                    f"{self.settings.llm_api_key_env} or switch rigging.strategy to 'heuristic'."
                ),
            )

        client = LLMRiggingClient(
            model=self.settings.llm_model,
            base_url=self.settings.llm_base_url,
            api_key=api_key,
        )
        try:
            return client.generate_rig_configuration(layers)
        except RiggingLLMError as exc:
            raise PipelineStageError(stage=self.name, message=str(exc)) from exc

    def _generate_rig_heuristic(self, layers: Iterable[LayerArtifact]) -> dict[str, object]:
        parts = []
        for artifact in layers:
            x0, y0, x1, y1 = artifact.bbox
            parts.append(
                {
                    "id": f"PART_{artifact.name}",
                    "layer": artifact.name,
                    "bounds": [x0, y0, x1, y1],
                    "pivot": [
                        round((x0 + x1) / 2, 2),
                        round((y0 + y1) / 2, 2),
                    ],
                }
            )

        parameters = [
            {"id": "ParamAngleX", "default": 0.0, "min": -30.0, "max": 30.0},
            {"id": "ParamAngleY", "default": 0.0, "min": -30.0, "max": 30.0},
            {"id": "ParamBodyAngleZ", "default": 0.0, "min": -10.0, "max": 10.0},
            {"id": "ParamMouthOpenY", "default": 0.0, "min": 0.0, "max": 1.0},
        ]

        return {
            "template": self.settings.template,
            "parts": parts,
            "parameters": parameters,
            "motions": {"idle": "Motions/idle.motion3.json"},
            "notes": "Generated by CubiAI heuristic rigging pipeline.",
        }

    def _run_builder(self, ctx: PipelineContext, rig_data: dict[str, object]) -> dict[str, object]:
        if not self.settings.enabled:
            return {"status": "skipped"}

        if self.settings.builder is None or not self.settings.builder.command:
            raise PipelineStageError(
                stage=self.name,
                message=(
                    "Rigging builder configuration missing. Define rigging.builder.command in the configuration "
                    "to point at a Live2D-compatible exporter."
                ),
            )

        workspace = ctx.workspace
        psd_path = workspace.layers_dir / "layers.psd"
        if not psd_path.exists():
            raise PipelineStageError(
                stage=self.name,
                message="PSD export must complete before rigging. Expected layers.psd to exist.",
            )

        rig_json_path = workspace.save_metadata("rig_config.json", rig_data)

        builder_cfg = self.settings.builder
        command: list[str] = []
        for arg in builder_cfg.command:
            expanded = os.path.expandvars(arg)
            if "$" in expanded:
                raise PipelineStageError(
                    stage=self.name,
                    message=(
                        "Environment variable placeholder did not resolve in builder command: "
                        f"{arg}"
                    ),
                )
            command.append(
                expanded.format(
                    PSD_PATH=str(psd_path),
                    RIG_JSON=str(rig_json_path),
                    WORKSPACE=str(workspace.root),
                    OUTPUT_DIR=str(workspace.live2d_dir),
                )
            )

        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                timeout=builder_cfg.timeout_seconds,
                cwd=builder_cfg.working_dir,
                text=True,
            )
        except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime dependent
            raise PipelineStageError(
                stage=self.name,
                message=f"Rigging builder failed with exit code {exc.returncode}: {exc.stderr.strip()}",
            ) from exc
        except FileNotFoundError as exc:
            raise PipelineStageError(
                stage=self.name,
                message=f"Rigging builder binary not found: {command[0]}",
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise PipelineStageError(
                stage=self.name,
                message="Rigging builder timed out. Increase timeout or optimise the command.",
            ) from exc

        builder_log = {
            "command": command,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }

        moc_path = workspace.live2d_dir / "model.moc3"
        if not moc_path.exists():
            raise PipelineStageError(
                stage=self.name,
                message=(
                    "Rigging builder did not produce model.moc3 in the workspace Live2D directory. "
                    "Check the command template or builder output."
                ),
            )

        return builder_log


class Live2DExportStage(PipelineStage):
    name = "Live2DExport"

    def __init__(self, workspace: Workspace, live2d: Live2DSettings, rigging: RiggingSettings) -> None:
        self.workspace = workspace
        self.settings = live2d
        self.rigging = rigging
        self.logger = logging.getLogger(self.name)

    def run(self, ctx: PipelineContext) -> None:
        if not self.settings.enabled:
            ctx.record_diagnostic(self.name, {"status": "skipped"})
            return
        layers: list[LayerArtifact] = ctx.data.get("layers", [])
        rig = ctx.data.get("rig")
        if not layers or rig is None:
            raise PipelineStageError(stage=self.name, message="Live2D export requires layers and rig data")

        exporter = Live2DExporter(workspace=self.workspace, settings=self.settings, rigging=self.rigging)
        outputs = exporter.export(layers=layers, rig_data=rig)
        for key, value in outputs.items():
            ctx.record_output(key, value)
        ctx.record_diagnostic(
            self.name,
            {"outputs": {key: str(path) for key, path in outputs.items()}},
        )


class SummaryStage(PipelineStage):
    name = "Summary"

    def run(self, ctx: PipelineContext) -> None:
        layers: list[LayerArtifact] = ctx.data.get("layers", [])
        ctx.record_diagnostic(
            self.name,
            {
                "layer_count": len(layers),
                "outputs": {key: str(val) for key, val in ctx.outputs.items()},
            },
        )


def build_default_stages(config: AppConfig, workspace: Workspace) -> list[PipelineStage]:
    return [
        ImageLoaderStage(workspace=workspace),
        SegmentationStage(settings=config.segmentation, workspace=workspace),
        PSDExportStage(
            workspace=workspace,
            enabled=config.export.psd.enabled,
            group_name=config.export.psd.group_name,
        ),
        RiggingStage(settings=config.rigging),
        Live2DExportStage(
            workspace=workspace,
            live2d=config.export.live2d,
            rigging=config.rigging,
        ),
        SummaryStage(),
    ]
