"""PySide6-based reviewer for SLIC clustering results with integrated prepare controls."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageQt
from skimage import segmentation

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..services.cluster_prepare import PrepareOptions, run_prepare


@dataclass(slots=True)
class ClusterSummary:
    cluster_id: int
    count: int
    coverage: float


@dataclass(slots=True)
class Assignment:
    image_name: str
    image_path: Path
    segment_id: int
    cluster_id: int
    centroid: tuple[float, float]
    area: float


def _load_summary(workdir: Path) -> tuple[list[ClusterSummary], dict[str, object]]:
    summary_path = workdir / "cluster_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    payload = json.loads(summary_path.read_text())
    summaries = [
        ClusterSummary(
            cluster_id=int(entry["cluster_id"]),
            count=int(entry.get("count", 0)),
            coverage=float(entry.get("coverage", 0.0)),
        )
        for entry in payload.get("entries", [])
    ]
    return summaries, payload


def _load_assignments(workdir: Path) -> list[Assignment]:
    assignments_path = workdir / "cluster_assignments.json"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Assignments file not found: {assignments_path}")
    payload = json.loads(assignments_path.read_text())
    assignments: list[Assignment] = []
    for item in payload:
        try:
            image_path_raw = item.get("image_path") or item.get("image")
            image_name = item.get("image") or Path(image_path_raw or "").name
            image_path = Path(image_path_raw) if image_path_raw else Path(image_name)
            assignments.append(
                Assignment(
                    image_name=str(image_name),
                    image_path=image_path,
                    segment_id=int(item["segment_id"]),
                    cluster_id=int(item["cluster_id"]),
                    centroid=tuple(float(c) for c in item.get("centroid", (0.0, 0.0))),
                    area=float(item.get("area", 0.0)),
                )
            )
        except (KeyError, ValueError, TypeError):  # pragma: no cover - defensive parsing
            continue
    return assignments


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
            resolved = Path(image_path)
        except (TypeError, ValueError):
            continue
        mapping[key] = resolved
        mapping.setdefault((resolved.name, int(segment_id)), resolved)
    return mapping


def _resolve_image_path(
    assignment: Assignment,
    metadata_map: dict[tuple[str, int], Path],
    image_root: Path | None,
) -> Path | None:
    candidates: list[Path] = []
    if assignment.image_path:
        candidates.append(assignment.image_path)
    key = (assignment.image_name, assignment.segment_id)
    if key in metadata_map:
        candidates.append(metadata_map[key])
    alt_key = (assignment.image_path.name, assignment.segment_id)
    if alt_key in metadata_map:
        candidates.append(metadata_map[alt_key])
    if image_root is not None:
        candidates.append(image_root / assignment.image_path.name)
        candidates.append(image_root / assignment.image_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _highlight_segment(
    image_path: Path,
    segment_id: int,
    *,
    superpixels: int,
    compactness: float,
) -> np.ndarray | None:
    with Image.open(image_path) as src:
        rgb_image = src.convert("RGB")
    rgb = np.array(rgb_image)
    rgb_float = np.ascontiguousarray(rgb, dtype=np.float64) / 255.0
    try:
        segments = segmentation.slic(
            rgb_float,
            n_segments=superpixels,
            compactness=compactness,
            start_label=0,
            channel_axis=-1,
        )
    except Exception:
        return None

    if segment_id < 0 or segment_id > segments.max():
        return None

    mask = segments == segment_id
    if not np.any(mask):
        return None

    overlay = rgb.copy()
    overlay[~mask] = (overlay[~mask] * 0.25).astype(np.uint8)
    return overlay


def _to_pixmap(array: np.ndarray, max_size: int = 240) -> QPixmap:
    image = Image.fromarray(array)
    qimage = ImageQt.ImageQt(image)
    pixmap = QPixmap.fromImage(qimage)
    return pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class ReviewTab(QWidget):
    def __init__(
        self,
        *,
        workdir: Path,
        summaries: list[ClusterSummary],
        assignments: list[Assignment],
        metadata_map: dict[tuple[str, int], Path],
        default_superpixels: int,
        default_compactness: float,
        default_clusters: int,
        image_root: Path | None,
    ) -> None:
        super().__init__()
        self.workdir = workdir
        self.metadata_map = metadata_map
        self.image_root = image_root
        self.default_clusters = default_clusters
        self.sample_pixmaps: list[QPixmap] = []

        self.assignments_by_cluster: Dict[int, list[Assignment]] = defaultdict(list)
        for assignment in assignments:
            self.assignments_by_cluster[assignment.cluster_id].append(assignment)
        self.summaries = {summary.cluster_id: summary for summary in summaries}

        self._build_ui(default_superpixels, default_compactness)
        self.reload_data(image_root=image_root, summaries=summaries, assignments=assignments)

    def _build_ui(self, default_superpixels: int, default_compactness: float) -> None:
        layout = QHBoxLayout(self)

        # Controls column
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        form = QFormLayout()

        self.image_root_line = QLineEdit(str(self.image_root or ""))
        browse_root = QPushButton("Browse…")
        browse_root.clicked.connect(self._select_image_root)
        root_row = QWidget()
        root_row_layout = QHBoxLayout(root_row)
        root_row_layout.setContentsMargins(0, 0, 0, 0)
        root_row_layout.addWidget(self.image_root_line)
        root_row_layout.addWidget(browse_root)
        form.addRow("Image root", root_row)

        self.cluster_spin = QSpinBox()
        self.cluster_spin.setRange(2, 512)
        self.cluster_spin.setValue(self.default_clusters)
        form.addRow("Clusters (prepare)", self.cluster_spin)

        self.superpixels_spin = QSpinBox()
        self.superpixels_spin.setRange(32, 4096)
        self.superpixels_spin.setSingleStep(8)
        self.superpixels_spin.setValue(default_superpixels)
        form.addRow("SLIC superpixels", self.superpixels_spin)

        self.compactness_spin = QDoubleSpinBox()
        self.compactness_spin.setRange(0.1, 64.0)
        self.compactness_spin.setSingleStep(0.1)
        self.compactness_spin.setDecimals(1)
        self.compactness_spin.setValue(default_compactness)
        form.addRow("SLIC compactness", self.compactness_spin)

        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(1, 48)
        self.sample_spin.setValue(6)
        form.addRow("Samples per cluster", self.sample_spin)

        controls_layout.addLayout(form)

        self.cluster_list = QListWidget()
        controls_layout.addWidget(QLabel("Clusters"))
        controls_layout.addWidget(self.cluster_list)
        controls_layout.addStretch()

        # Review area
        review_widget = QWidget()
        review_layout = QVBoxLayout(review_widget)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        review_layout.addWidget(self.summary_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.samples_container = QWidget()
        self.samples_layout = QGridLayout(self.samples_container)
        self.samples_layout.setAlignment(Qt.AlignTop)
        self.samples_layout.setHorizontalSpacing(12)
        self.samples_layout.setVerticalSpacing(12)
        self.scroll_area.setWidget(self.samples_container)
        review_layout.addWidget(self.scroll_area)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        review_layout.addWidget(self.status_label)

        layout.addWidget(controls_widget, 0)
        layout.addWidget(review_widget, 1)

        self.cluster_list.currentItemChanged.connect(self._on_cluster_changed)
        self.superpixels_spin.valueChanged.connect(self._refresh_current_cluster)
        self.compactness_spin.valueChanged.connect(self._refresh_current_cluster)
        self.sample_spin.valueChanged.connect(self._refresh_current_cluster)

    def _select_image_root(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select image directory", str(self.image_root or self.workdir))
        if directory:
            self.set_image_root(Path(directory))

    def set_image_root(self, image_root: Path) -> None:
        self.image_root = image_root
        self.image_root_line.setText(str(image_root))
        self._refresh_current_cluster()

    def update_default_clusters(self, value: int) -> None:
        self.default_clusters = max(2, value)
        self.cluster_spin.setValue(self.default_clusters)

    def reload_data(
        self,
        *,
        image_root: Path | None = None,
        summaries: list[ClusterSummary] | None = None,
        assignments: list[Assignment] | None = None,
    ) -> None:
        if image_root is not None:
            self.set_image_root(image_root)

        if summaries is None or assignments is None:
            try:
                summaries, _ = _load_summary(self.workdir)
                assignments = _load_assignments(self.workdir)
                self.metadata_map = _load_metadata_map(self.workdir)
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                self.cluster_list.clear()
                self.samples_layout.addWidget(QLabel(str(exc)), 0, 0)
                return

        self.assignments_by_cluster = defaultdict(list)
        for assignment in assignments:
            self.assignments_by_cluster[assignment.cluster_id].append(assignment)
        self.summaries = {summary.cluster_id: summary for summary in summaries}

        self.cluster_list.clear()
        cluster_ids = sorted(self.assignments_by_cluster.keys())
        for cluster_id in cluster_ids:
            summary = self.summaries.get(cluster_id)
            label = f"Cluster {cluster_id}" if summary is None else f"Cluster {cluster_id} ({summary.count})"
            QListWidgetItem(label, self.cluster_list)

        if cluster_ids:
            self.cluster_list.setCurrentRow(0)
        else:
            self.samples_layout.addWidget(QLabel("No clusters available."), 0, 0)

    def _on_cluster_changed(self, *_: object) -> None:
        self._refresh_current_cluster()

    def _refresh_current_cluster(self) -> None:
        current_item = self.cluster_list.currentItem()
        if current_item is None:
            return
        cluster_id_text = current_item.text().split()[1]
        cluster_id = int(cluster_id_text)
        self._render_cluster(cluster_id)

    def _render_cluster(self, cluster_id: int) -> None:
        assignments = self.assignments_by_cluster.get(cluster_id, [])
        summary = self.summaries.get(cluster_id)
        if summary:
            self.summary_label.setText(
                f"Segments: {summary.count}\nCoverage: {summary.coverage * 100:.2f}%"
            )
        else:
            self.summary_label.setText("")

        while self.samples_layout.count():
            item = self.samples_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.sample_pixmaps.clear()

        if not assignments:
            self.samples_layout.addWidget(QLabel("No assignments found for this cluster."), 0, 0)
            return

        assignments = sorted(assignments, key=lambda a: a.area, reverse=True)
        assignments = assignments[: self.sample_spin.value()]

        superpixels = self.superpixels_spin.value()
        compactness = self.compactness_spin.value()

        for index, assignment in enumerate(assignments):
            image_path = _resolve_image_path(assignment, self.metadata_map, self.image_root)
            container = QWidget()
            card_layout = QVBoxLayout(container)
            title = QLabel(
                f"{assignment.image_name} — segment {assignment.segment_id}\n"
                f"Centroid: ({assignment.centroid[0]:.3f}, {assignment.centroid[1]:.3f}) · "
                f"Area ratio: {assignment.area:.5f}"
            )
            title.setWordWrap(True)
            card_layout.addWidget(title)

            if image_path is None:
                card_layout.addWidget(QLabel("[Image path unavailable]"))
            else:
                overlay = _highlight_segment(
                    image_path,
                    assignment.segment_id,
                    superpixels=superpixels,
                    compactness=compactness,
                )
                if overlay is None:
                    card_layout.addWidget(QLabel("[Failed to generate preview]"))
                else:
                    pixmap = _to_pixmap(overlay)
                    self.sample_pixmaps.append(pixmap)
                    label = QLabel()
                    label.setPixmap(pixmap)
                    label.setAlignment(Qt.AlignCenter)
                    card_layout.addWidget(label)

            container.setStyleSheet("#sample-card { border: 1px solid #444; padding: 6px; }")
            container.setObjectName("sample-card")
            row = index // 4
            col = index % 4
            self.samples_layout.addWidget(container, row, col)


class PrepareTab(QWidget):
    def __init__(
        self,
        *,
        workdir: Path,
        default_clusters: int,
        default_superpixels: int,
        default_compactness: float,
        default_image_root: Path | None,
        review_tab: ReviewTab,
    ) -> None:
        super().__init__()
        self.workdir = workdir
        self.review_tab = review_tab
        self.default_image_root = default_image_root

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.images_line = QLineEdit(str(default_image_root or ""))
        browse_images = QPushButton("Browse…")
        browse_images.clicked.connect(self._select_images)
        img_row = QWidget()
        img_row_layout = QHBoxLayout(img_row)
        img_row_layout.setContentsMargins(0, 0, 0, 0)
        img_row_layout.addWidget(self.images_line)
        img_row_layout.addWidget(browse_images)
        form.addRow("Images directory", img_row)

        self.workdir_label = QLabel(str(workdir))
        form.addRow("Workdir", self.workdir_label)

        self.cluster_spin = QSpinBox()
        self.cluster_spin.setRange(2, 512)
        self.cluster_spin.setValue(default_clusters)
        form.addRow("Clusters", self.cluster_spin)

        self.superpixels_spin = QSpinBox()
        self.superpixels_spin.setRange(32, 4096)
        self.superpixels_spin.setSingleStep(8)
        self.superpixels_spin.setValue(default_superpixels)
        form.addRow("SLIC superpixels", self.superpixels_spin)

        self.compactness_spin = QDoubleSpinBox()
        self.compactness_spin.setRange(0.1, 64.0)
        self.compactness_spin.setSingleStep(0.1)
        self.compactness_spin.setDecimals(1)
        self.compactness_spin.setValue(default_compactness)
        form.addRow("SLIC compactness", self.compactness_spin)

        self.run_button = QPushButton("Run prepare")
        self.run_button.clicked.connect(self._run_prepare)
        form.addRow("", self.run_button)

        layout.addLayout(form)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        layout.addWidget(self.log_output)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch()

    def _select_images(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select image directory", str(self.default_image_root or self.workdir))
        if directory:
            self.images_line.setText(directory)

    def _append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)
        self.log_output.ensureCursorVisible()
        QApplication.processEvents()

    def _run_prepare(self) -> None:
        images_dir_text = self.images_line.text().strip()
        if not images_dir_text and self.default_image_root is not None:
            images_dir_text = str(self.default_image_root)
        if not images_dir_text:
            QMessageBox.warning(self, "Missing images", "Select an image directory before running prepare.")
            return

        images_dir = Path(images_dir_text)
        options = PrepareOptions(
            images_dir=images_dir,
            workdir=self.workdir,
            n_clusters=self.cluster_spin.value(),
            superpixels=self.superpixels_spin.value(),
            compactness=self.compactness_spin.value(),
            batch_size=2048,
            random_state=42,
            workers=None,
        )

        self.run_button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.log_output.clear()
        self.status_label.setText("Running prepare…")
        try:
            run_prepare(options, log=self._append_log)
            self.review_tab.update_default_clusters(options.n_clusters)
            self.review_tab.reload_data(image_root=images_dir)
            self.status_label.setText("Prepare completed.")
        except ValueError as exc:
            QMessageBox.critical(self, "Prepare failed", str(exc))
            self.status_label.setText(str(exc))
        finally:
            QApplication.restoreOverrideCursor()
            self.run_button.setEnabled(True)


class ClusterReviewMainWindow(QMainWindow):
    def __init__(
        self,
        *,
        workdir: Path,
        summaries: list[ClusterSummary],
        assignments: list[Assignment],
        metadata_map: dict[tuple[str, int], Path],
        default_superpixels: int,
        default_compactness: float,
        default_clusters: int,
        image_root: Path | None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("CubiAI Cluster Reviewer")
        self.workdir = workdir

        self.review_tab = ReviewTab(
            workdir=workdir,
            summaries=summaries,
            assignments=assignments,
            metadata_map=metadata_map,
            default_superpixels=default_superpixels,
            default_compactness=default_compactness,
            default_clusters=default_clusters,
            image_root=image_root,
        )
        self.prepare_tab = PrepareTab(
            workdir=workdir,
            default_clusters=default_clusters,
            default_superpixels=default_superpixels,
            default_compactness=default_compactness,
            default_image_root=image_root or workdir,
            review_tab=self.review_tab,
        )

        tabs = QTabWidget()
        tabs.addTab(self.prepare_tab, "Prepare")
        tabs.addTab(self.review_tab, "Review")

        self.setCentralWidget(tabs)
        self.resize(1280, 820)


def launch(
    *,
    workdir: Path,
    superpixels: int | None = None,
    compactness: float | None = None,
    image_root: Path | None = None,
) -> None:
    try:
        summaries, payload = _load_summary(workdir)
        assignments = _load_assignments(workdir)
        metadata_map = _load_metadata_map(workdir)
    except (FileNotFoundError, json.JSONDecodeError):
        summaries = []
        payload = {"config": {}}
        assignments = []
        metadata_map = {}

    config = payload.get("config", {})
    default_superpixels = superpixels or int(config.get("superpixels", 300))
    default_compactness = compactness or float(config.get("compactness", 8.0))
    default_clusters = int(config.get("n_clusters", len(summaries) or 64))

    app = QApplication.instance()
    owns_app = False
    if app is None:
        app = QApplication(sys.argv)
        owns_app = True

    window = ClusterReviewMainWindow(
        workdir=workdir,
        summaries=summaries,
        assignments=assignments,
        metadata_map=metadata_map,
        default_superpixels=default_superpixels,
        default_compactness=default_compactness,
        default_clusters=default_clusters,
        image_root=image_root,
    )
    window.show()

    if owns_app:
        app.exec()


__all__ = ["launch"]
