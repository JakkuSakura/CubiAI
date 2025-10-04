"""Codex CLI-driven annotation helper that emits LabelMe-compatible JSON."""
from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import textwrap
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from ..config import AnnotationLLMSettings

DEFAULT_LABELS: tuple[str, ...] = (
    "hair",
    "face",
    "eyes",
    "mouth",
    "skin",
    "clothes",
    "accessories",
)


class AnnotationLLMError(RuntimeError):
    """Raised when the annotation model call fails or returns invalid data."""


class LabelMeShape(BaseModel):
    """Minimal LabelMe shape schema focusing on polygon segments."""

    label: str = Field(description="Semantic label for the annotated region.")
    points: list[tuple[float, float]] = Field(
        description="Polygon vertices in image pixel coordinates ordered clockwise or counter-clockwise.",
    )
    group_id: int | None = Field(default=None, description="Optional group identifier.")
    shape_type: str = Field(default="polygon", description="LabelMe shape type.")
    flags: dict[str, bool] = Field(default_factory=dict, description="Label-specific boolean flags.")
    description: str | None = Field(default=None, description="Optional free-form notes for reviewers.")


class LabelMeLLMResponse(BaseModel):
    """Schema used to parse the LLM response before merging with image metadata."""

    version: str | None = Field(default=None, description="Optional LabelMe format version supplied by the model.")
    flags: dict[str, bool] = Field(default_factory=dict, description="Global annotation flags.")
    shapes: list[LabelMeShape] = Field(default_factory=list, description="Collection of annotated shapes.")


class LabelMeAnnotation(BaseModel):
    """Full LabelMe annotation payload ready to persist to disk."""

    version: str = Field(default="5.2.1")
    flags: dict[str, bool] = Field(default_factory=dict)
    shapes: list[LabelMeShape] = Field(default_factory=list)
    imagePath: str
    imageData: str | None = None
    imageHeight: int
    imageWidth: int


def _load_image(image_path: Path) -> tuple[Image.Image, bytes, str]:
    with Image.open(image_path) as source:
        image = source.copy()
        fmt = (source.format or "png").lower()
    if fmt not in {"png", "jpeg", "jpg", "webp"}:
        fmt = "png"
    if fmt == "jpg":
        fmt = "jpeg"
    with image_path.open("rb") as fh:
        data = fh.read()
    return image, data, fmt


@dataclass(slots=True, frozen=True)
class AnnotationResult:
    """Aggregate result produced by the Codex annotation flow."""

    annotation: LabelMeAnnotation
    summary: str


@dataclass(slots=True)
class CodexAnnotationTool:
    """Invoke the Codex CLI to generate LabelMe annotations."""

    settings: AnnotationLLMSettings
    labels: Sequence[str] = field(default_factory=lambda: DEFAULT_LABELS)
    include_image_data: bool = False
    _prompt_template: str | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.settings.enabled:
            raise AnnotationLLMError("Annotation LLM is disabled in the configuration.")
        if not self.settings.codex_binary.strip():
            raise AnnotationLLMError("Codex CLI binary path is empty; set annotation.llm.codex_binary.")
        self._prompt_template = self.settings.prompt_template

    def annotate(
            self,
            image_path: Path,
            *,
            output_path: Path | None = None,
            instructions: str | None = None,
            extra_labels: Iterable[str] | None = None,
    ) -> AnnotationResult:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image, byte_data, _ = _load_image(image_path)
        width, height = image.size

        labels = list(dict.fromkeys([*self.labels, *(extra_labels or [])]))
        target_output = output_path or image_path.with_suffix(".labelme.json")
        response_json_path = image_path.with_suffix('.json')
        prompt = self._build_prompt(
            image_path=image_path,
            output_path=target_output,
            width=width,
            height=height,
            labels=labels,
            instructions=instructions,
            response_json_path=response_json_path
        )

        self._invoke_codex(prompt=prompt)

        try:
            payload = json.load(open(response_json_path))
        except json.JSONDecodeError as exc:  # pragma: no cover - depends on remote output quality
            raise AnnotationLLMError(
                f"Codex CLI returned invalid JSON: {exc.msg} (pos {exc.pos})"
            ) from exc

        summary_payload = payload.get("summary")

        if not isinstance(summary_payload, str) or not summary_payload.strip():
            raise AnnotationLLMError("Codex response missing a non-empty summary string.")

        try:
            parsed = LabelMeLLMResponse.model_validate(target_output)
        except ValidationError as exc:  # pragma: no cover - depends on remote output quality
            raise AnnotationLLMError(f"Model returned malformed annotation payload: {exc}") from exc

        filtered_shapes = [
            shape
            for shape in parsed.shapes
            if shape.shape_type != "polygon" or len(shape.points) >= 3
        ]

        annotation = LabelMeAnnotation(
            version=parsed.version or "5.2.1",
            flags=parsed.flags,
            shapes=filtered_shapes,
            imagePath=image_path.name,
            imageData=base64.b64encode(byte_data).decode("ascii") if self.include_image_data else None,
            imageHeight=height,
            imageWidth=width,
        )

        return AnnotationResult(annotation=annotation, summary=summary_payload.strip())

    def _build_prompt(
            self,
            *,
            image_path: Path,
            output_path: Path,
            width: int,
            height: int,
            labels: Sequence[str],
            instructions: str | None,
            response_json_path: Path
    ) -> str:
        labels_text = ", ".join(labels)
        guidance = instructions.strip() if instructions else "None provided."
        include_hint = (
            "Set `imageData` to null; the application will fill it after validation."
            if not self.include_image_data
            else "You may set `imageData` to null; the application can embed data later."
        )

        prompt = textwrap.dedent(
            f"""
            You are an expert anime image annotator. Produce LabelMe-compatible polygons for the supplied image.

            Image metadata:
            - Path: {image_path}
            - Dimensions: width {width}px, height {height}px

            Classes to cover: {labels_text}
            Extra guidance: {guidance}

            Output JSON files (write the JSON payload only, no commentary):
            1. {output_path}
               - Keys: version, flags, shapes, imagePath, imageData, imageWidth, imageHeight
               - Use version "5.2.1" if not specified; default flags to {{}}
               - Each shape must include label, points (>=3 [x, y]), group_id (null or integer), shape_type "polygon", flags {{}}
               - Add description only when it clarifies the region (e.g. "left eye")
               - Use imagePath "{image_path.name}", imageWidth {width}, imageHeight {height}. {include_hint}
               - Annotate every clearly visible class listed above with at least one polygon; multiple polygons are allowed
            2. {response_json_path}
               - Single top-level key "summary" with one concise English sentence describing the image

            Follow the instructions exactly and do not emit any other files or text.
            """
        ).strip()

        return prompt

    def _invoke_codex(self, *, prompt: str):
        command = [
            self.settings.codex_binary,
            "exec",
            "--full-auto",
            prompt,
        ]

        try:
            result = subprocess.run(
                command,
                check=True,
                text=True,
                timeout=self.settings.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise AnnotationLLMError(
                f"Codex CLI binary not found: {self.settings.codex_binary}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AnnotationLLMError(
                "Codex CLI timed out before completing the annotation request."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or exc.stdout or "").strip()
            raise AnnotationLLMError(
                f"Codex CLI failed with exit code {exc.returncode}: {stderr or 'no stderr available'}"
            ) from exc
