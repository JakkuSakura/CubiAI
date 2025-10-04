"""Streamlit app for reviewing SLIC clustering results."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import streamlit as st
from PIL import Image
from skimage import segmentation



_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("workdir", nargs="?", default="data")
_parser.add_argument("--superpixels", type=int, dest="superpixels")
_parser.add_argument("--compactness", type=float, dest="compactness")
_parser.add_argument("--image-root", type=str, dest="image_root")
_cli_args, _ = _parser.parse_known_args()

@dataclass(slots=True)
class ClusterSummary:
    cluster_id: int
    count: int
    coverage: float
    sample_segments: list[dict[str, object]]


@dataclass(slots=True)
class Assignment:
    image_path: Path
    image_name: str
    segment_id: int
    cluster_id: int
    centroid: tuple[float, float]
    area: float


@st.cache_resource(show_spinner=False)
def _load_metadata_map(workdir: Path) -> dict[tuple[str, int], Path]:
    metadata_path = workdir / "metadata.json"
    mapping: dict[tuple[str, int], Path] = {}
    if not metadata_path.exists():
        return mapping
    try:
        payload = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return mapping
    for entry in payload:
        image_name = entry.get("image")
        image_path = entry.get("image_path")
        segment_id = entry.get("segment_id")
        if image_name is None or image_path is None or segment_id is None:
            continue
        try:
            key = (str(image_name), int(segment_id))
        except (TypeError, ValueError):
            continue
        mapping[key] = Path(image_path)
        mapping.setdefault((str(image_path), int(segment_id)), Path(image_path))
    return mapping


@st.cache_resource(show_spinner=False)
def _load_summary(workdir: Path) -> tuple[list[ClusterSummary], dict[str, object]]:
    summary_path = workdir / "cluster_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    payload = json.loads(summary_path.read_text())
    entries = [
        ClusterSummary(
            cluster_id=int(entry["cluster_id"]),
            count=int(entry.get("count", 0)),
            coverage=float(entry.get("coverage", 0.0)),
            sample_segments=list(entry.get("sample_segments", [])),
        )
        for entry in payload.get("entries", [])
    ]
    return entries, payload


@st.cache_resource(show_spinner=False)
def _load_assignments(workdir: Path) -> list[Assignment]:
    assignments_path = workdir / "cluster_assignments.json"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Assignments file not found: {assignments_path}")
    payload = json.loads(assignments_path.read_text())
    assignments: list[Assignment] = []
    for item in payload:
        try:
            image_path_raw = item.get("image_path")
            image_name = item.get("image") or (Path(image_path_raw).name if image_path_raw else "")
            image_path = Path(image_path_raw) if image_path_raw else Path(image_name)
            assignments.append(
                Assignment(
                    image_path=image_path,
                    image_name=str(image_name),
                    segment_id=int(item["segment_id"]),
                    cluster_id=int(item["cluster_id"]),
                    centroid=tuple(float(c) for c in item.get("centroid", (0.0, 0.0))),
                    area=float(item.get("area", 0.0)),
                )
            )
        except (KeyError, ValueError, TypeError):
            continue
    return assignments


@st.cache_resource(show_spinner=False)
def _load_image(path: Path) -> np.ndarray:
    with Image.open(path) as src:
        img = src.convert("RGB")
    return np.array(img)


def _render_segment(
    image_path: Path,
    segment_id: int,
    *,
    superpixels: int,
    compactness: float,
) -> Image.Image | None:
    if not image_path.exists():
        st.warning(f"Image missing: {image_path}")
        return None

    rgb = _load_image(image_path)
    rgb_float = np.ascontiguousarray(rgb, dtype=np.float64) / 255.0
    try:
        segments = segmentation.slic(
            rgb_float,
            n_segments=superpixels,
            compactness=compactness,
            start_label=0,
            channel_axis=-1,
        )
    except Exception as exc:
        st.error(f"SLIC failed for {image_path.name}: {exc}")
        return None

    if segment_id >= segments.max() + 1:
        st.warning(f"Segment {segment_id} not found in recomputed SLIC for {image_path.name}.")
        return None

    mask = segments == segment_id
    if not np.any(mask):
        st.warning(f"Segment {segment_id} empty for {image_path.name}.")
        return None

    overlay = rgb.copy()
    overlay[~mask] = (overlay[~mask] * 0.25).astype(np.uint8)

    return Image.fromarray(overlay)


def main() -> None:
    st.set_page_config(page_title="Cluster Reviewer", layout="wide")
    st.title("CubiAI Cluster Reviewer")

    workdir_input = st.sidebar.text_input("Workdir", value=str(_cli_args.workdir))
    image_root_input = st.sidebar.text_input("Image root", value=str(_cli_args.image_root or ""))
    image_root = Path(image_root_input).expanduser().resolve() if image_root_input else None
    workdir = Path(workdir_input).expanduser().resolve()

    if not workdir.exists():
        st.error(f"Workdir not found: {workdir}")
        return

    try:
        summaries, summary_payload = _load_summary(workdir)
        assignments = _load_assignments(workdir)
        metadata_map = _load_metadata_map(workdir)
    except FileNotFoundError as missing:
        st.error(str(missing))
        return
    except json.JSONDecodeError as exc:
        st.error(f"Failed to parse JSON: {exc}")
        return

    if not summaries:
        st.warning("No clusters found in summary.")
        return

    config = summary_payload.get("config", {})
    default_superpixels = int(config.get("superpixels", 300))
    default_compactness = float(config.get("compactness", 8.0))

    init_superpixels = _cli_args.superpixels or default_superpixels
    init_compactness = _cli_args.compactness or default_compactness

    superpixels = st.sidebar.number_input("SLIC superpixels", min_value=32, value=init_superpixels, step=8)
    compactness = st.sidebar.number_input("SLIC compactness", min_value=0.1, value=init_compactness, step=0.1)
    sample_limit = st.sidebar.slider("Samples per cluster", min_value=1, max_value=24, value=6)

    cluster_ids = sorted(summary.cluster_id for summary in summaries)
    selected_cluster = st.sidebar.selectbox("Cluster", cluster_ids, index=0)

    summary_map = {summary.cluster_id: summary for summary in summaries}
    target_summary = summary_map.get(selected_cluster)

    st.subheader(f"Cluster {selected_cluster}")
    col1, col2 = st.columns([1, 1])
    if target_summary:
        with col1:
            st.metric("Segments", target_summary.count)
        with col2:
            st.metric("Coverage", f"{target_summary.coverage * 100:.2f}%")

    cluster_assignments = [a for a in assignments if a.cluster_id == selected_cluster]
    if not cluster_assignments:
        st.info("No assignments found for this cluster.")
        return

    def resolve_image_path(assignment: Assignment) -> Path | None:
        if assignment.image_path.exists():
            return assignment.image_path
        return metadata_map.get((assignment.image_name, assignment.segment_id))

    cluster_assignments.sort(key=lambda item: item.area, reverse=True)
    display_assignments = cluster_assignments[:sample_limit]

    for assignment in display_assignments:
        image_path = resolve_image_path(assignment)
        if image_path is None:
            st.warning(f"Path unavailable for {assignment.image_name}")
            continue
        preview = _render_segment(
            image_path,
            assignment.segment_id,
            superpixels=superpixels,
            compactness=compactness,
        )
        if preview is not None:
            st.image(preview, use_container_width=True)
        st.caption(
            f"Centroid: ({assignment.centroid[0]:.3f}, {assignment.centroid[1]:.3f}) · "
            f"Area ratio: {assignment.area:.5f}"
        )

    st.divider()
    with st.expander("Assignments table"):
        st.dataframe(
            {
                "image": [a.image_name for a in cluster_assignments],
                "segment_id": [a.segment_id for a in cluster_assignments],
                "area": [a.area for a in cluster_assignments],
                "centroid_x": [a.centroid[0] for a in cluster_assignments],
                "centroid_y": [a.centroid[1] for a in cluster_assignments],
            },
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
