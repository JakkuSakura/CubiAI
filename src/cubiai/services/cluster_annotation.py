"""Semi-supervised cluster annotation backend."""
from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
from skimage import color, measure, morphology, segmentation

from .labelme_annotation import AnnotationResult, LabelMeAnnotation, LabelMeShape
from ..config.models import ClusterAnnotationSettings

FEATURE_VERSION = 1


class ClusterAnnotationError(RuntimeError):
    """Raised when the cluster-based annotator fails."""


@dataclass(slots=True)
class _SegmentRecord:
    segment_id: int
    centroid: tuple[float, float]
    area: float
    mean_color: tuple[float, float, float]


def _safe_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_features(
    image: np.ndarray,
    segments: np.ndarray,
) -> tuple[np.ndarray, list[_SegmentRecord]]:
    """Extract per-segment feature vectors and metadata."""

    height, width = segments.shape
    lab = color.rgb2lab(image)
    coords_y, coords_x = np.indices((height, width), dtype=np.float64)

    features: list[np.ndarray] = []
    records: list[_SegmentRecord] = []

    total_pixels = float(height * width)

    for segment_id in np.unique(segments):
        mask = segments == segment_id
        pixel_count = int(mask.sum())
        if pixel_count == 0:
            continue

        lab_pixels = lab[mask]
        rgb_pixels = image[mask]

        mean_lab = _safe_array(lab_pixels.mean(axis=0))
        std_lab = _safe_array(lab_pixels.std(axis=0))
        mean_rgb = _safe_array(rgb_pixels.mean(axis=0) / 255.0)

        centroid_x = float(coords_x[mask].mean() / width)
        centroid_y = float(coords_y[mask].mean() / height)
        area_ratio = float(pixel_count / total_pixels)

        feature = np.concatenate(
            [
                mean_lab,
                std_lab,
                mean_rgb,
                np.array([centroid_x, centroid_y, area_ratio], dtype=np.float64),
            ]
        )
        features.append(feature)

        record = _SegmentRecord(
            segment_id=int(segment_id),
            centroid=(centroid_x, centroid_y),
            area=area_ratio,
            mean_color=(float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2])),
        )
        records.append(record)

    if not features:
        raise ClusterAnnotationError("SLIC segmentation produced no usable segments.")

    feature_matrix = np.stack(features, axis=0)
    return feature_matrix.astype(np.float64), records


def _mask_to_polygon(
    mask: np.ndarray,
    *,
    tolerance: float,
    max_points: int,
) -> list[tuple[float, float]] | None:
    contours = measure.find_contours(mask.astype(np.float32), 0.5)
    if not contours:
        return None

    contour = max(contours, key=len)
    simplified = measure.approximate_polygon(contour, tolerance)
    if simplified.shape[0] < 3:
        return None

    points: list[tuple[float, float]] = []
    for row, col in simplified:
        points.append((float(col), float(row)))

    if points[0] == points[-1]:
        points.pop()
    if len(points) < 3:
        return None

    if len(points) > max_points:
        step = max(1, len(points) // max_points)
        points = points[::step]
        if len(points) < 3:
            return None

    return points


@dataclass(slots=True)
class ClusterAnnotationModel:
    settings: ClusterAnnotationSettings
    include_image_data: bool = False

    def __post_init__(self) -> None:
        self._load_bundle()

    def _load_bundle(self) -> None:
        model_path = Path(self.settings.model_path)
        if not model_path.exists():
            raise ClusterAnnotationError(
                f"Cluster model not found at {model_path}. Run train.py to prepare the model."
            )

        try:
            bundle = joblib.load(model_path)
        except Exception as exc:  # pragma: no cover - joblib specific
            raise ClusterAnnotationError(f"Failed to load cluster model: {exc}") from exc

        feature_version = bundle.get("feature_version")
        if feature_version != FEATURE_VERSION:
            raise ClusterAnnotationError(
                f"Unsupported feature version {feature_version}; expected {FEATURE_VERSION}."
            )

        self._kmeans = bundle.get("kmeans")
        if self._kmeans is None:
            raise ClusterAnnotationError("Model bundle missing kmeans estimator.")

        self._scaler = bundle.get("scaler")
        self._classifier = bundle.get("classifier")
        self._labels: list[str] = bundle.get("labels", [])
        self._cluster_to_label: dict[int, str] = {
            int(key): value for key, value in bundle.get("cluster_to_label", {}).items()
        }
        config = bundle.get("config", {})
        self._superpixels = int(config.get("superpixels", self.settings.superpixels))
        self._compactness = float(config.get("compactness", self.settings.compactness))
        self._n_clusters = int(config.get("n_clusters", self.settings.n_clusters))

        if self._n_clusters != self.settings.n_clusters:
            # Not fatal but warn via exception detail to prompt update.
            raise ClusterAnnotationError(
                "Configured n_clusters does not match trained model. Update settings or retrain."
            )

    def annotate(self, image_path: Path) -> AnnotationResult:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as src:
            image = src.convert("RGB")
        rgb = np.array(image)
        segments = segmentation.slic(
            rgb,
            n_segments=self._superpixels,
            compactness=self._compactness,
            start_label=0,
        )

        features, records = _compute_features(rgb, segments)
        cluster_ids = self._kmeans.predict(features)

        probabilities: np.ndarray | None = None
        if self._classifier is not None and self._scaler is not None and self._labels:
            features_scaled = self._scaler.transform(features)
            probabilities = self._classifier.predict_proba(features_scaled)

        shapes: list[LabelMeShape] = []
        height, width = segments.shape
        struct_elem = None
        if self.settings.dilation_radius > 0:
            struct_elem = morphology.disk(self.settings.dilation_radius)

        for idx, record in enumerate(records):
            cluster_id = int(cluster_ids[idx])
            mask = segments == record.segment_id
            if struct_elem is not None:
                mask = morphology.binary_dilation(mask, struct_elem)

            label: str | None = self._cluster_to_label.get(cluster_id)
            confidence: float | None = None

            if probabilities is not None:
                cluster_probs = probabilities[idx]
                best_idx = int(np.argmax(cluster_probs))
                best_prob = float(cluster_probs[best_idx])
                predicted_label = self._labels[best_idx]
                if best_prob >= self.settings.min_probability:
                    label = predicted_label
                    confidence = best_prob
                elif label is None:
                    label = self.settings.unknown_label if self.settings.unknown_label else None
                    confidence = best_prob
                else:
                    confidence = best_prob
            elif label is None:
                if self.settings.unknown_label:
                    label = self.settings.unknown_label
                    confidence = 0.0
                else:
                    continue

            if label is None:
                continue

            polygon = _mask_to_polygon(
                mask,
                tolerance=self.settings.contour_tolerance,
                max_points=self.settings.max_points_per_polygon,
            )
            if not polygon:
                continue

            description = None
            if confidence is not None:
                description = json.dumps({"confidence": round(confidence, 3), "cluster": cluster_id})
            else:
                description = json.dumps({"cluster": cluster_id})

            shape = LabelMeShape(
                label=label,
                points=polygon,
                group_id=None,
                shape_type="polygon",
                flags={},
                description=description,
            )
            shapes.append(shape)

        image_data: str | None = None
        if self.include_image_data:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_data = base64.b64encode(buffer.getvalue()).decode("ascii")

        annotation = LabelMeAnnotation(
            version="5.2.1",
            flags={},
            shapes=shapes,
            imagePath=image_path.name,
            imageData=image_data,
            imageHeight=height,
            imageWidth=width,
        )

        summary = (
            f"Cluster annotator produced {len(shapes)} polygon(s) using semi-supervised pseudo labels."
        )
        return AnnotationResult(annotation=annotation, summary=summary)


@dataclass(slots=True)
class ClusterAnnotationTool:
    settings: ClusterAnnotationSettings
    include_image_data: bool = False
    _model: ClusterAnnotationModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._model = ClusterAnnotationModel(
            settings=self.settings,
            include_image_data=self.include_image_data,
        )

    def annotate(self, image_path: Path, **_: object) -> AnnotationResult:
        return self._model.annotate(image_path=image_path)

__all__ = [
    "ClusterAnnotationModel",
    "ClusterAnnotationTool",
    "ClusterAnnotationError",
    "FEATURE_VERSION",
    "compute_feature_matrix",
]


def compute_feature_matrix(image: np.ndarray, segments: np.ndarray) -> tuple[np.ndarray, list[_SegmentRecord]]:
    """Public helper to extract feature matrix and segment metadata."""
    return _compute_features(image, segments)
