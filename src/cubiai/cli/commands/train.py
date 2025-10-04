"""Training utilities for the semi-supervised cluster annotator."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import typer
from PIL import Image
from rich.console import Console
from skimage import segmentation
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ...services.cluster_annotation import FEATURE_VERSION, compute_feature_matrix

console = Console()
app = typer.Typer(help="Prepare and refine the semi-supervised cluster annotator")


IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
FEATURES_FILE = "features.npy"
CLUSTER_IDS_FILE = "cluster_ids.npy"
METADATA_FILE = "metadata.json"
SUMMARY_FILE = "cluster_summary.json"
ASSIGNMENTS_FILE = "cluster_assignments.json"
MODEL_FILE = "cluster_labeler.joblib"


def _iter_images(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if ".preview" in path.name:
            continue
        yield path


def _load_group_labels(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text())
    mapping: dict[int, str] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            mapping[int(key)] = str(value)
        return mapping
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            if "cluster" in item and "label" in item:
                mapping[int(item["cluster"])] = str(item["label"])
        return mapping
    raise typer.BadParameter(
        "Group labels must be a mapping or a list of {\"cluster\": int, \"label\": str}."
    )


@app.command()
def prepare(
    images_dir: Path = typer.Argument(..., help="Directory containing training images (excluding *.preview.* files)."),
    workdir: Path = typer.Argument(..., help="Directory where intermediate artifacts and the model are written."),
    n_clusters: int = typer.Option(64, "--clusters", min=2, help="Number of k-means clusters."),
    superpixels: int = typer.Option(300, "--superpixels", min=32, help="SLIC superpixels per image."),
    compactness: float = typer.Option(8.0, "--compactness", min=0.1, help="SLIC compactness parameter."),
    batch_size: int = typer.Option(2048, "--batch-size", min=32, help="MiniBatchKMeans batch size."),
    random_state: int = typer.Option(42, "--random-state", help="Random seed for reproducibility."),
) -> None:
    """Run clustering over images and persist artifacts for later labelling."""

    if not images_dir.exists() or not images_dir.is_dir():
        raise typer.BadParameter(f"Image directory not found: {images_dir}")

    image_paths = list(_iter_images(images_dir))
    if not image_paths:
        raise typer.BadParameter("No eligible images found (accepted: png, jpg, webp, bmp).")

    workdir.mkdir(parents=True, exist_ok=True)
    model_target = workdir / MODEL_FILE

    console.print(f"[bold]Extracting features from {len(image_paths)} image(s)...[/bold]")

    feature_rows: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []

    for image_path in image_paths:
        with Image.open(image_path) as src:
            image = src.convert("RGB")
        rgb = np.array(image)
        segments = segmentation.slic(
            rgb,
            n_segments=superpixels,
            compactness=compactness,
            start_label=0,
        )
        features, records = compute_feature_matrix(rgb, segments)
        feature_rows.extend(features)
        for record in records:
            metadata.append(
                {
                    "image": image_path.name,
                    "image_path": str(image_path),
                    "segment_id": record.segment_id,
                    "centroid": [round(record.centroid[0], 6), round(record.centroid[1], 6)],
                    "area": record.area,
                }
            )

    if not feature_rows:
        raise typer.BadParameter("Failed to compute any segment features.")

    features_matrix = np.stack(feature_rows, axis=0)

    console.print("[bold]Running MiniBatchKMeans clustering...[/bold]")
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=random_state,
        n_init="auto",
        max_iter=100,
    )
    cluster_ids = kmeans.fit_predict(features_matrix)

    summary_entries: list[dict[str, object]] = []
    for cluster_id in range(n_clusters):
        indices = np.where(cluster_ids == cluster_id)[0]
        if len(indices) == 0:
            continue
        coverage = float(sum(metadata[idx]["area"] for idx in indices))
        sample = [
            {
                "image": metadata[idx]["image"],
                "segment_id": metadata[idx]["segment_id"],
                "centroid": metadata[idx]["centroid"],
                "area": round(float(metadata[idx]["area"]), 5),
            }
            for idx in indices[: min(5, len(indices))]
        ]
        summary_entries.append(
            {
                "cluster_id": int(cluster_id),
                "count": int(len(indices)),
                "coverage": coverage,
                "sample_segments": sample,
            }
        )

    assignments = [
        {
            "image": metadata[idx]["image"],
            "segment_id": metadata[idx]["segment_id"],
            "cluster_id": int(cluster_ids[idx]),
            "centroid": metadata[idx]["centroid"],
            "area": round(float(metadata[idx]["area"]), 5),
        }
        for idx in range(len(metadata))
    ]

    summary = {
        "prepared_at": datetime.utcnow().isoformat(timespec="seconds"),
        "images": len(image_paths),
        "segments": len(metadata),
        "clusters": n_clusters,
        "superpixels": superpixels,
        "compactness": compactness,
        "entries": summary_entries,
    }

    np.save(workdir / FEATURES_FILE, features_matrix)
    np.save(workdir / CLUSTER_IDS_FILE, cluster_ids)
    (workdir / METADATA_FILE).write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    (workdir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (workdir / ASSIGNMENTS_FILE).write_text(json.dumps(assignments, indent=2, ensure_ascii=False))

    bundle = {
        "feature_version": FEATURE_VERSION,
        "kmeans": kmeans,
        "scaler": None,
        "classifier": None,
        "labels": [],
        "cluster_to_label": {},
        "config": {
            "superpixels": superpixels,
            "compactness": compactness,
            "n_clusters": n_clusters,
            "prepared_at": datetime.utcnow().isoformat(timespec="seconds"),
        },
    }
    joblib.dump(bundle, model_target)

    console.print(f"[bold green]Clustering complete. Model saved to {model_target}[/bold green]")
    console.print(
        f"[dim]Next steps:[/dim] Review {SUMMARY_FILE} / {ASSIGNMENTS_FILE}, create a group-label JSON, then run `cubiai train label`."
    )


@app.command()
def label(
    workdir: Path = typer.Argument(..., help="Working directory produced by the prepare step."),
    group_labels: Path = typer.Argument(..., help="JSON mapping of cluster ids to semantic labels."),
    model_path: Path | None = typer.Option(
        None,
        "--model-path",
        help="Override the model file to update (defaults to <workdir>/cluster_labeler.joblib).",
    ),
    max_iter: int = typer.Option(1000, "--max-iter", min=100, help="Max iterations for logistic regression."),
) -> None:
    """Attach group labels and fit the pseudo-label classifier for refinement."""

    workdir = workdir.resolve()
    if not workdir.exists():
        raise typer.BadParameter(f"Workdir not found: {workdir}")

    features_path = workdir / FEATURES_FILE
    cluster_ids_path = workdir / CLUSTER_IDS_FILE
    metadata_path = workdir / METADATA_FILE
    model_target = model_path or (workdir / MODEL_FILE)

    for required in (features_path, cluster_ids_path, metadata_path, model_target):
        if not required.exists():
            raise typer.BadParameter(f"Required artifact missing: {required}")

    cluster_to_label = _load_group_labels(group_labels)
    if not cluster_to_label:
        raise typer.BadParameter("Group labels file contained no label assignments.")

    features_matrix = np.load(features_path)
    cluster_ids = np.load(cluster_ids_path)

    bundle = joblib.load(model_target)
    bundle["cluster_to_label"] = {str(k): v for k, v in cluster_to_label.items()}

    train_indices = [idx for idx, cid in enumerate(cluster_ids) if int(cid) in cluster_to_label]
    if len(train_indices) < 2:
        raise typer.BadParameter(
            "Need at least two labelled clusters to train the classifier. Label more clusters first."
        )

    y_labels = [cluster_to_label[int(cluster_ids[idx])] for idx in train_indices]
    unique_labels = sorted(set(y_labels))
    if len(unique_labels) < 2:
        raise typer.BadParameter(
            "At least two distinct labels are required to train the classifier."
        )

    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_to_index[label] for label in y_labels])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_matrix[train_indices])

    classifier = LogisticRegression(max_iter=max_iter, multi_class="multinomial")
    classifier.fit(X_scaled, y)

    bundle["scaler"] = scaler
    bundle["classifier"] = classifier
    bundle["labels"] = unique_labels
    joblib.dump(bundle, model_target)

    console.print(
        f"[bold green]Updated model saved to {model_target} with {len(unique_labels)} labelled classes.[/bold green]"
    )


__all__ = ["app"]
