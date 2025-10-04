"""Shared utilities for running the SLIC-based clustering prepare step."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image
from skimage import segmentation
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from .cluster_annotation import compute_feature_matrix

IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass(slots=True)
class PrepareOptions:
    images_dir: Path
    workdir: Path
    n_clusters: int = 64
    superpixels: int = 300
    compactness: float = 8.0
    batch_size: int = 2048
    random_state: int = 42
    workers: int | None = None


def _iter_images(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if ".preview" in path.name:
            continue
        yield path


def _process_image(
    image_path: Path,
    *,
    superpixels: int,
    compactness: float,
) -> tuple[np.ndarray | None, list[dict[str, object]]]:
    try:
        with Image.open(image_path) as src:
            image = src.convert("RGB")
    except Exception as exc:  # pragma: no cover
        return None, [{"warning": f"failed to read: {exc}"}]

    rgb = np.array(image)
    rgb_float = np.ascontiguousarray(rgb, dtype=np.float64) / 255.0
    try:
        segments = segmentation.slic(
            rgb_float,
            n_segments=superpixels,
            compactness=compactness,
            start_label=0,
            channel_axis=-1,
        )
    except Exception as exc:  # pragma: no cover
        return None, [{"warning": f"slic_failed: {exc}"}]

    try:
        features, records = compute_feature_matrix(rgb, segments)
    except Exception as exc:  # pragma: no cover
        return None, [{"warning": f"feature_failed: {exc}"}]

    metadata = [
        {
            "image": image_path.name,
            "image_path": str(image_path),
            "segment_id": record.segment_id,
            "centroid": [round(record.centroid[0], 6), round(record.centroid[1], 6)],
            "area": record.area,
        }
        for record in records
    ]
    return features, metadata


def run_prepare(
    options: PrepareOptions,
    *,
    log: Callable[[str], None] | None = None,
) -> None:
    images_dir = options.images_dir
    workdir = options.workdir

    if not images_dir.exists() or not images_dir.is_dir():
        raise ValueError(f"Image directory not found: {images_dir}")

    image_paths = list(_iter_images(images_dir))
    if not image_paths:
        raise ValueError("No eligible images found (accepted: png, jpg, webp, bmp).")

    workdir.mkdir(parents=True, exist_ok=True)

    def emit(message: str) -> None:
        if log:
            log(message)

    emit(f"Extracting features from {len(image_paths)} image(s)...")

    feature_rows: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    processed_images = 0

    max_workers = options.workers or os.cpu_count() or 1

    if max_workers <= 1:
        iterator = tqdm(image_paths, desc="Images", unit="img")
        for image_path in iterator:
            features, meta = _process_image(
                image_path,
                superpixels=options.superpixels,
                compactness=options.compactness,
            )
            if features is None:
                warning = meta[0].get("warning") if meta else "unknown error"
                emit(f"Skipping {image_path}: {warning}")
                continue
            feature_rows.extend(features)
            metadata.extend(meta)
            processed_images += 1
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_image,
                    path,
                    superpixels=options.superpixels,
                    compactness=options.compactness,
                ): path
                for path in image_paths
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Images", unit="img"):
                image_path = futures[future]
                try:
                    features, meta = future.result()
                except Exception as exc:  # pragma: no cover
                    emit(f"Skipping {image_path}: {exc}")
                    continue
                if features is None:
                    warning = meta[0].get("warning") if meta else "unknown error"
                    emit(f"Skipping {image_path}: {warning}")
                    continue
                feature_rows.extend(features)
                metadata.extend(meta)
                processed_images += 1

    if not feature_rows:
        raise ValueError("Failed to compute any segment features.")
    if processed_images == 0:
        raise ValueError("No images produced usable SLIC segments. Check the dataset or parameters.")

    features_matrix = np.stack(feature_rows, axis=0)

    emit("Running MiniBatchKMeans clustering...")
    kmeans = MiniBatchKMeans(
        n_clusters=options.n_clusters,
        batch_size=options.batch_size,
        random_state=options.random_state,
        n_init="auto",
        max_iter=100,
    )
    cluster_ids = kmeans.fit_predict(features_matrix)

    summary_entries: list[dict[str, object]] = []
    for cluster_id in range(options.n_clusters):
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
            "image_path": metadata[idx]["image_path"],
            "segment_id": metadata[idx]["segment_id"],
            "cluster_id": int(cluster_ids[idx]),
            "centroid": metadata[idx]["centroid"],
            "area": round(float(metadata[idx]["area"]), 5),
        }
        for idx in range(len(metadata))
    ]

    summary = {
        "prepared_at": datetime.utcnow().isoformat(timespec="seconds"),
        "images": processed_images,
        "segments": len(metadata),
        "clusters": options.n_clusters,
        "superpixels": options.superpixels,
        "compactness": options.compactness,
        "entries": summary_entries,
        "config": {
            "superpixels": options.superpixels,
            "compactness": options.compactness,
            "n_clusters": options.n_clusters,
        },
    }

    np.save(options.workdir / "features.npy", features_matrix)
    np.save(options.workdir / "cluster_ids.npy", cluster_ids)
    (options.workdir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    (options.workdir / "cluster_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (options.workdir / "cluster_assignments.json").write_text(json.dumps(assignments, indent=2, ensure_ascii=False))

    emit(f"Clustering complete. Artifacts saved to {options.workdir}")
