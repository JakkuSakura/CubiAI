"""Pydantic models representing processing profiles."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class HuggingFaceSegmentationConfig(BaseModel):
    """Configuration for Hugging Face Inference API segmentation backends."""

    model: str = Field(default="facebook/sam-vit-base", description="Model repository on Hugging Face.")
    endpoint: str | None = Field(
        default=None,
        description="Optional fully qualified endpoint URL. Defaults to the standard inference URL for the model.",
    )
    token_env: str = Field(
        default="HF_API_TOKEN",
        description="Environment variable that stores the API token.",
    )
    max_layers: int = Field(default=16, ge=1, description="Maximum number of layers emitted from the API response.")
    score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum mask confidence for inclusion in the layer stack.",
    )


class SegmentationSettings(BaseModel):
    backend: Literal["slic", "huggingface-sam"] = Field(
        default="slic",
        description="Segmentation backend identifier.",
    )
    num_segments: int = Field(
        default=12,
        ge=2,
        description="Target number of segments produced by the SLIC backend.",
    )
    compactness: float = Field(
        default=10.0,
        ge=0.1,
        description="Compactness parameter for SLIC-like algorithms.",
    )
    min_area_px: int = Field(
        default=4096,
        ge=1,
        description="Minimum area in pixels required for a layer to be emitted.",
    )
    feather_radius: int = Field(
        default=2,
        ge=0,
        description="Feather radius applied to masks to reduce aliasing.",
    )
    huggingface: HuggingFaceSegmentationConfig | None = Field(
        default=None,
        description="Settings for the Hugging Face segmentation backend.",
    )


class PSDExportSettings(BaseModel):
    enabled: bool = True
    group_name: str = Field(default="CubiAI Layers")
    embed_metadata: bool = True


class Live2DSettings(BaseModel):
    enabled: bool = True
    texture_size: int = Field(default=2048, description="Width/height of the texture atlas.")
    padding: int = Field(default=24, ge=0, description="Padding between layers in the atlas.")
    background_color: tuple[int, int, int, int] = Field(
        default=(0, 0, 0, 0), description="Atlas background RGBA color."
    )


class RiggingBuilderSettings(BaseModel):
    command: list[str] = Field(
        default_factory=list,
        description="Command template used to invoke an external Live2D exporter. Supports placeholders {PSD_PATH}, {RIG_JSON}, {WORKSPACE}, {OUTPUT_DIR}.",
    )
    timeout_seconds: int = Field(default=900, ge=1, description="Timeout for the external builder command.")
    working_dir: str | None = Field(
        default=None,
        description="Optional working directory for the builder command.",
    )


class RiggingSettings(BaseModel):
    enabled: bool = True
    template: str = Field(default="humanoid-basic")
    idle_motion: bool = Field(default=True)
    blink_parameters: tuple[str, str] = Field(
        default=("ParamEyeLOpen", "ParamEyeROpen"),
        description="Parameter IDs controlling eye openness.",
    )
    strategy: Literal["llm", "heuristic"] = Field(
        default="llm",
        description="Rigging strategy. 'llm' requires an LLM-backed service for parameter generation.",
    )
    llm_model: str = Field(default="gpt-4.1-mini", description="Model identifier for the LLM rigging provider.")
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the LLM provider compatible with the OpenAI API schema.",
    )
    llm_api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable name that stores the LLM provider API key.",
    )
    builder: RiggingBuilderSettings | None = Field(
        default=None,
        description="External Live2D builder/extractor configuration. Required when rigging is enabled.",
    )


class InlineModelAsset(BaseModel):
    name: str
    path: str
    description: str | None = None

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value:
            msg = "Model asset path cannot be empty"
            raise ValueError(msg)
        return value


class ModelAssets(BaseModel):
    assets: list[InlineModelAsset] = Field(default_factory=list)


class ExportSettings(BaseModel):
    psd: PSDExportSettings = Field(default_factory=PSDExportSettings)
    live2d: Live2DSettings = Field(default_factory=Live2DSettings)


class ProfileMetadata(BaseModel):
    created_by: str | None = None
    notes: str | None = None


class ProfileConfig(BaseModel):
    name: str = Field(default="anime-default")
    description: str = Field(default="Default anime character processing profile.")
    segmentation: SegmentationSettings = Field(default_factory=SegmentationSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    rigging: RiggingSettings = Field(default_factory=RiggingSettings)
    models: ModelAssets = Field(default_factory=ModelAssets)
    metadata: ProfileMetadata = Field(default_factory=ProfileMetadata)

    extras: dict[str, Any] = Field(default_factory=dict, description="Additional profile data.")

    model_config = {
        "extra": "allow",
    }
