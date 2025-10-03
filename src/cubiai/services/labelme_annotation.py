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

        prompt = self._build_prompt(
            image_path=image_path,
            output_path=target_output,
            width=width,
            height=height,
            labels=labels,
            instructions=instructions,
        )

        raw_output = self._invoke_codex(prompt=prompt, image_path=image_path)

        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as exc:  # pragma: no cover - depends on remote output quality
            raise AnnotationLLMError(
                f"Codex CLI returned invalid JSON: {exc.msg} (pos {exc.pos}). Raw output: {raw_output!r}"
            ) from exc

        annotation_payload = payload.get("annotation")
        summary_payload = payload.get("summary")

        if not isinstance(annotation_payload, dict):
            raise AnnotationLLMError("Codex response missing 'annotation' object.")
        if not isinstance(summary_payload, str) or not summary_payload.strip():
            raise AnnotationLLMError("Codex response missing a non-empty summary string.")

        try:
            parsed = LabelMeLLMResponse.model_validate(annotation_payload)
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
    ) -> str:
        if self._prompt_template:
            context = {
                "image_path": str(image_path),
                "output_path": str(output_path),
                "image_width": width,
                "image_height": height,
                "labels": ", ".join(labels),
                "instructions": (instructions or "").strip(),
            }
            return self._prompt_template.format(**context)

        labels_text = ", ".join(labels)
        guidance = instructions.strip() if instructions else "None provided."
        include_hint = (
            "Set `imageData` to null; the application will fill it after validation."
            if not self.include_image_data
            else "You may set `imageData` to null; the application can embed data later."
        )

        prompt = textwrap.dedent(
            f"""
            You are an expert image annotator producing LabelMe-compatible JSON for anime character assets.

            Basic LabelMe annotation structure:
            - `version`: string version identifier (use 5.2.1 if unsure).
            - `flags`: object of boolean flags (use an empty object if none).
            - `shapes`: array of regions. Each shape includes:
              * `label`: semantic class name.
              * `points`: list of [x, y] coordinate pairs in pixel space.
              * `group_id`: null or integer group identifier.
              * `shape_type`: usually "polygon".
              * `flags`: object of optional booleans.
              * `description`: optional notes.
            - `imagePath`: filename only (no directories).
            - `imageData`: null.
            - `imageWidth`: integer pixel width.
            - `imageHeight`: integer pixel height.

            Source image path: {image_path}
            Output JSON path: {output_path}
            Image dimensions: width={width}, height={height}
            Recognised classes: {labels_text}
            Additional guidance: {guidance}

            Produce a JSON object with exactly these top-level keys:
            {{
              "annotation": {{ ...LabelMe annotation as described above... }},
              "summary": "One concise English sentence describing the image"
            }}

            Requirements:
            - Inspect the attached image to infer polygons for the recognised classes.
            - Emit at least one polygon per clearly visible class.
            - Every polygon must contain three or more points.
            - Use `imagePath` = "{image_path.name}", `imageWidth` = {width}, `imageHeight` = {height}.
            - {include_hint}
            - The summary must be a single sentence suitable for human review.
            - Respond with raw JSON only (no markdown fences or prose).
            """
        ).strip()

        return prompt

    def _invoke_codex(self, *, prompt: str, image_path: Path) -> str:
        tmp_path: Path | None = None
        try:
            handle = tempfile.NamedTemporaryFile(mode="w", delete=False)
            tmp_path = Path(handle.name)
            handle.close()
        except OSError as exc:  # pragma: no cover - OS specific
            raise AnnotationLLMError(f"Failed to allocate temporary file for Codex output: {exc}") from exc

        command = [
            self.settings.codex_binary,
            "exec",
            "--model",
            self.settings.model,
            "--output-last-message",
            str(tmp_path),
            "-i",
            str(image_path),
        ]
        if self.settings.extra_cli_args:
            command.extend(self.settings.extra_cli_args)
        command.append(prompt)

        output = ""
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
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
        else:
            if tmp_path and tmp_path.exists():
                try:
                    output = tmp_path.read_text().strip()
                except OSError:
                    output = (result.stdout or "").strip()
            else:
                output = (result.stdout or "").strip()
        finally:
            if tmp_path:
                with suppress(FileNotFoundError):
                    tmp_path.unlink()

        if not output:
            raise AnnotationLLMError(
                "Codex CLI returned no content. Ensure the prompt requests raw JSON output."
            )

        return output
