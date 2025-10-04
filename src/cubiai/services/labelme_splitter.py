"""Split image layers based on LabelMe annotations using Skia."""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import skia

from .labelme_annotation import LabelMeAnnotation

TRANSPARENT = skia.ColorSetARGB(0, 0, 0, 0)


@dataclass(frozen=True)
class LayerSlice:
    """Represents an exported layer produced from a polygon annotation."""

    label: str
    index: int
    image_path: Path
    mask_path: Path | None
    area: float
    polygon: Sequence[tuple[float, float]]


def _load_annotation(path: Path) -> LabelMeAnnotation:
    data = json.loads(path.read_text())
    return LabelMeAnnotation.model_validate(data)


def _make_path(points: Iterable[tuple[float, float]]) -> skia.Path:
    iterator = iter(points)
    try:
        start = next(iterator)
    except StopIteration:  # pragma: no cover - guarded by caller
        return skia.Path()

    path = skia.Path()
    path.moveTo(float(start[0]), float(start[1]))
    for x, y in iterator:
        path.lineTo(float(x), float(y))
    path.close()
    return path


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "shape"


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    pts = list(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]):
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def _subset(image: skia.Image, bounds: skia.Rect) -> skia.Image:
    left = max(int(math.floor(bounds.left())), 0)
    top = max(int(math.floor(bounds.top())), 0)
    right = min(int(math.ceil(bounds.right())), image.width())
    bottom = min(int(math.ceil(bounds.bottom())), image.height())
    if right <= left or bottom <= top:
        return image
    rect = skia.IRect.MakeLTRB(left, top, right, bottom)
    subset = image.subset(rect)
    return subset or image


def _render_layer(
    *,
    base_image: skia.Image,
    path: skia.Path,
    layer_name: str,
    include_mask: bool,
    output_dir: Path,
) -> tuple[Path, Path | None]:
    width, height = base_image.width(), base_image.height()

    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    canvas.clear(TRANSPARENT)
    canvas.save()
    canvas.clipPath(path, antialias=True)
    canvas.drawImage(base_image, 0, 0)
    canvas.restore()
    snapshot = surface.makeImageSnapshot()

    bounds = path.computeTightBounds()
    cropped = _subset(snapshot, bounds)

    layer_path = output_dir / f"{layer_name}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    cropped.save(str(layer_path), skia.kPNG)

    mask_path: Path | None = None
    if include_mask:
        mask_surface = skia.Surface(width, height)
        mask_canvas = mask_surface.getCanvas()
        mask_canvas.clear(TRANSPARENT)
        paint = skia.Paint(Color=skia.ColorWHITE, Style=skia.Paint.kFill_Style, AntiAlias=True)
        mask_canvas.drawPath(path, paint)
        mask_snapshot = mask_surface.makeImageSnapshot()
        mask_cropped = _subset(mask_snapshot, bounds)
        mask_path = output_dir / f"{layer_name}_mask.png"
        mask_cropped.save(str(mask_path), skia.kPNG)

    return layer_path, mask_path


def split_annotation_layers(
    image_path: Path,
    annotation_path: Path,
    output_dir: Path,
    *,
    include_mask: bool = True,
) -> list[LayerSlice]:
    """Generate per-polygon image layers for the supplied annotation."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation JSON not found: {annotation_path}")

    annotation = _load_annotation(annotation_path)
    base_image = skia.Image.open(str(image_path))

    slices: list[LayerSlice] = []
    counts: dict[str, int] = {}

    for shape in annotation.shapes:
        if shape.shape_type != "polygon" or len(shape.points) < 3:
            continue

        points = [(float(x), float(y)) for x, y in shape.points]
        path = _make_path(points)
        if path.isEmpty():
            continue

        area = _polygon_area(points)
        label_slug = _slugify(shape.label)
        counts[label_slug] = counts.get(label_slug, 0) + 1
        layer_index = counts[label_slug]
        layer_name = f"{layer_index:02d}_{label_slug}"

        image_file, mask_file = _render_layer(
            base_image=base_image,
            path=path,
            layer_name=layer_name,
            include_mask=include_mask,
            output_dir=output_dir,
        )

        slice_info = LayerSlice(
            label=shape.label,
            index=layer_index,
            image_path=image_file,
            mask_path=mask_file,
            area=area,
            polygon=points,
        )
        slices.append(slice_info)

    slices.sort(key=lambda item: item.area, reverse=True)
    return slices
