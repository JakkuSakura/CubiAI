"""Training utilities for the semi-supervised cluster annotator."""
from __future__ import annotations

import json

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import typer
from rich.console import Console
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ...services.cluster_prepare import PrepareOptions, run_prepare
from ...viewers import cluster_review

console = Console()
app = typer.Typer(help="Prepare, review, and label cluster annotations")


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
                mapping[int(item["cluster"])] = str(item["label"] )
        return mapping
    raise typer.BadParameter(
        "Group labels must be a mapping or a list of {\"cluster\": int, \"label\": str}."
    )


@app.command()
def prepare(
    images_dir: Path = typer.Argument(..., help="Directory containing training images (excluding *.preview.* files)."),
    workdir: Path = typer.Argument(..., help="Directory where clustering artifacts are written."),
    n_clusters: int = typer.Option(64, "--clusters", min=2, help="Number of k-means clusters."),
    superpixels: int = typer.Option(300, "--superpixels", min=32, help="SLIC superpixels per image."),
    compactness: float = typer.Option(8.0, "--compactness", min=0.1, help="SLIC compactness parameter."),
    batch_size: int = typer.Option(2048, "--batch-size", min=32, help="MiniBatchKMeans batch size."),
    random_state: int = typer.Option(42, "--random-state", help="Random seed for reproducibility."),
    workers: Optional[int] = typer.Option(None, "--workers", min=1, help="Number of worker threads (default: CPU count)."),
) -> None:
    """Run clustering over images and persist artifacts for later labelling."""

    options = PrepareOptions(
        images_dir=images_dir,
        workdir=workdir,
        n_clusters=n_clusters,
        superpixels=superpixels,
        compactness=compactness,
        batch_size=batch_size,
        random_state=random_state,
        workers=workers,
    )

    def log(message: str) -> None:
        console.print(message)

    try:
        run_prepare(options, log=log)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def review(
    workdir: Path = typer.Argument(..., help="Path to the prepare() output directory."),
    superpixels: int | None = typer.Option(None, help="Override SLIC superpixels for previews."),
    compactness: float | None = typer.Option(None, help="Override SLIC compactness for previews."),
    image_root: Path | None = typer.Option(None, help="Override base path for original images."),
) -> None:
    """Launch the PySide6 cluster reviewer."""

    cluster_review.launch(
        workdir=workdir,
        superpixels=superpixels,
        compactness=compactness,
        image_root=image_root,
    )


@app.command()
def ui(
    workdir: Path = typer.Argument(..., help="Directory where clustering artifacts are written."),
    prepare_images: Path | None = typer.Option(
        None,
        "--prepare-images",
        help="Optional image directory to run prepare before launching the UI.",
    ),
    n_clusters: int = typer.Option(64, "--clusters", min=2, help="Clusters for optional prepare."),
    superpixels: int = typer.Option(300, "--superpixels", min=32, help="SLIC superpixels for prepare and viewer."),
    compactness: float = typer.Option(8.0, "--compactness", min=0.1, help="SLIC compactness."),
    batch_size: int = typer.Option(2048, "--batch-size", min=32, help="MiniBatchKMeans batch size."),
    random_state: int = typer.Option(42, "--random-state", help="Random seed for clustering."),
    workers: Optional[int] = typer.Option(None, "--workers", min=1, help="Worker threads for prepare."),
    image_root: Path | None = typer.Option(None, help="Override base path for original images."),
) -> None:
    """Run (optional) prepare and open the desktop reviewer."""

    if prepare_images is not None:
        options = PrepareOptions(
            images_dir=prepare_images,
            workdir=workdir,
            n_clusters=n_clusters,
            superpixels=superpixels,
            compactness=compactness,
            batch_size=batch_size,
            random_state=random_state,
            workers=workers,
        )
        console.print("[bold]Running prepare before launching the viewer…[/bold]")
        try:
            run_prepare(options, log=lambda msg: console.print(msg))
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        if image_root is None:
            image_root = prepare_images
    elif image_root is None:
        image_root = workdir

    cluster_review.launch(
        workdir=workdir,
        superpixels=superpixels,
        compactness=compactness,
        image_root=image_root,
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

    features_path = workdir / "features.npy"
    cluster_ids_path = workdir / "cluster_ids.npy"
    metadata_path = workdir / "metadata.json"
    model_target = model_path or (workdir / "cluster_labeler.joblib")

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
