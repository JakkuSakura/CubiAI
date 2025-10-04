"""Pydantic models representing the CubiAI configuration."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    backend: Literal["slic", "huggingface-sam", "sam-hq-local"] = Field(
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
    sam_hq_local_model_id: str | None = Field(
        default="syscv-community/sam-hq-vit-base",
        description="Local Hugging Face model identifier for SAM-HQ segmentation.",
    )
    sam_hq_local_device: str | None = Field(
        default=None,
        description="Preferred torch device string (e.g., cuda, cpu). Defaults to auto-detection.",
    )
    sam_hq_local_score_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum mask score returned by SAM-HQ to keep a layer.",
    )


class PSDExportSettings(BaseModel):
    enabled: bool = True
    group_name: str = Field(default="CubiAI Layers")
    embed_metadata: bool = True


class PNGExportSettings(BaseModel):
    enabled: bool = True


class Live2DSettings(BaseModel):
    enabled: bool = False
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
    enabled: bool = False
    template: str | None = None
    idle_motion: bool = False
    blink_parameters: tuple[str, str] | None = None
    strategy: Literal["llm", "heuristic"] | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key_env: str | None = None
    builder: RiggingBuilderSettings | None = None


class AnnotationLLMSettings(BaseModel):
    """Settings describing how the Codex CLI should be invoked for annotations."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = Field(default=False, description="Enable LLM-assisted annotation workflow.")
    model: str = Field(default="gpt-4o-mini", description="Model identifier passed to the Codex CLI.")
    codex_binary: str = Field(
        default="codex",
        description="Codex CLI executable used for non-interactive model calls.",
    )
    timeout_seconds: int = Field(
        default=180,
        ge=30,
        description="Subprocess timeout applied to the Codex invocation.",
    )
    prompt_template: str | None = Field(
        default=None,
        description="Optional template override for generating LabelMe annotations.",
    )
    extra_cli_args: list[str] = Field(
        default_factory=list,
        description="Additional flags passed directly to the Codex CLI (e.g. sandbox overrides).",
    )


class ClusterAnnotationSettings(BaseModel):
    """Settings for the semi-supervised cluster-based annotator."""

    enabled: bool = Field(default=True, description="Enable the semi-supervised cluster annotator.")
    model_path: str = Field(
        default="models/cluster_labeler.joblib",
        description="Path to the trained cluster annotator model produced by train.py.",
    )
    n_clusters: int = Field(
        default=64,
        ge=2,
        description="Number of clusters expected by the trained model (for validation only).",
    )
    superpixels: int = Field(
        default=300,
        ge=32,
        description="Number of SLIC superpixels generated per image during inference.",
    )
    compactness: float = Field(
        default=8.0,
        ge=0.1,
        description="Compactness parameter used for SLIC segmentation.",
    )
    min_probability: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum classifier confidence required to emit a polygon.",
    )
    unknown_label: str = Field(
        default="unknown",
        description="Fallback label assigned when confidence is insufficient.",
    )
    max_points_per_polygon: int = Field(
        default=120,
        ge=16,
        description="Maximum vertices retained after contour simplification.",
    )
    contour_tolerance: float = Field(
        default=2.5,
        ge=0.1,
        description="Tolerance for skimage.measure.approximate_polygon when simplifying contours.",
    )
    dilation_radius: int = Field(
        default=1,
        ge=0,
        description="Optional morphological dilation (in pixels) before contour extraction.",
    )


class AnnotationSettings(BaseModel):
    strategy: Literal["cluster", "codex"] = Field(
        default="cluster",
        description="Default annotation backend strategy.",
    )
    cluster: ClusterAnnotationSettings = Field(default_factory=ClusterAnnotationSettings)
    llm: AnnotationLLMSettings = Field(default_factory=AnnotationLLMSettings)


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
    png: PNGExportSettings = Field(default_factory=PNGExportSettings)
    live2d: Live2DSettings = Field(default_factory=Live2DSettings)


class ConfigMetadata(BaseModel):
    created_by: str | None = None
    notes: str | None = None


class AppConfig(BaseModel):
    name: str = Field(default="anime-default")
    description: str = Field(default="Default anime character processing configuration.")
    segmentation: SegmentationSettings = Field(default_factory=SegmentationSettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    rigging: RiggingSettings = Field(default_factory=RiggingSettings)
    annotation: AnnotationSettings = Field(default_factory=AnnotationSettings)
    models: ModelAssets = Field(default_factory=ModelAssets)
    metadata: ConfigMetadata = Field(default_factory=ConfigMetadata)

    extras: dict[str, Any] = Field(default_factory=dict, description="Additional configuration data.")

    model_config = {
        "extra": "allow",
    }
