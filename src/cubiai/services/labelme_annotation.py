"""LangChain-powered annotation helper that emits LabelMe-compatible JSON."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
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


def _data_url(raw_bytes: bytes, extension: str) -> str:
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:image/{extension};base64,{encoded}"


@dataclass(slots=True)
class LangChainAnnotationTool:
    """Wrapper around LangChain's ChatOpenAI for producing LabelMe annotations."""

    settings: AnnotationLLMSettings
    labels: Sequence[str] = field(default_factory=lambda: DEFAULT_LABELS)
    api_key: str | None = None
    include_image_data: bool = False
    _llm: ChatOpenAI = field(init=False, repr=False)
    _parser: PydanticOutputParser = field(init=False, repr=False)
    _system_prompt: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.settings.enabled:
            raise AnnotationLLMError("Annotation LLM is disabled in the configuration.")

        key = self.api_key or os.getenv(self.settings.api_key_env)
        if not key:
            raise AnnotationLLMError(
                f"API key missing; set {self.settings.api_key_env} or pass api_key explicitly."
            )

        llm_kwargs: dict[str, object] = {
            "model": self.settings.model,
            "api_key": key,
            "base_url": self.settings.base_url,
            "temperature": self.settings.temperature,
        }
        if self.settings.max_output_tokens:
            llm_kwargs["max_tokens"] = self.settings.max_output_tokens
        if self.settings.timeout_seconds:
            llm_kwargs["timeout"] = self.settings.timeout_seconds
        if self.settings.default_headers:
            llm_kwargs["default_headers"] = self.settings.default_headers

        try:
            self._llm = ChatOpenAI(**llm_kwargs)
        except Exception as exc:  # pragma: no cover - network/client specific
            raise AnnotationLLMError(f"Failed to initialize chat model: {exc}") from exc

        self._parser = PydanticOutputParser(pydantic_object=LabelMeLLMResponse)

        if self.settings.prompt_template:
            self._system_prompt = self.settings.prompt_template
        else:
            format_instructions = self._parser.get_format_instructions()
            labels_csv = ", ".join(self.labels)
            self._system_prompt = (
                "You are an expert annotator creating polygon labels for anime character assets. "
                "Follow the LabelMe specification and only return JSON. "
                f"Focus on these semantic classes: {labels_csv}.\n\n"
                f"{format_instructions}"
            )

    def annotate(
        self,
        image_path: Path,
        instructions: str | None = None,
        extra_labels: Iterable[str] | None = None,
    ) -> LabelMeAnnotation:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image, byte_data, extension = _load_image(image_path)
        width, height = image.size
        data_url = _data_url(byte_data, extension)

        labels = list(dict.fromkeys([*self.labels, *(extra_labels or [])]))
        labels_text = ", ".join(labels)

        instruction_lines = [
            "Produce polygon annotations in LabelMe JSON format for the provided image.",
            f"Image filename: {image_path.name}",
            f"Image size: width={width}, height={height}",
            f"Recognised classes: {labels_text}",
            "Use `imageData`: null and include at least one polygon for each clearly visible class.",
            "Exclude regions you cannot see or that are obstructed.",
        ]
        if instructions:
            instruction_lines.append(f"Additional guidance: {instructions.strip()}")
        instruction_lines.append(
            "Return polygons with three or more points unless a class is better represented as a point or line."
        )

        human_message = HumanMessage(
            content=[
                {"type": "text", "text": "\n".join(instruction_lines)},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )

        try:
            response = self._llm.invoke(
                [SystemMessage(content=self._system_prompt), human_message]
            )
        except Exception as exc:  # pragma: no cover - network/client specific
            raise AnnotationLLMError(f"Annotation request failed: {exc}") from exc

        try:
            parsed = self._parser.parse(response.content)
        except ValidationError as exc:  # pragma: no cover - depends on remote output quality
            raise AnnotationLLMError("Model returned malformed annotation payload") from exc

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
        return annotation
