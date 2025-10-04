"""Utilities for rendering annotation previews using Skia."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import skia

from .labelme_annotation import LabelMeAnnotation


@dataclass(frozen=True)
class PreviewStyle:
    """Visual parameters controlling the annotation preview rendering."""

    stroke_width: float = 2.5
    fill_opacity: float = 0.35
    label_background_opacity: float = 0.85
    label_text_size: float = 16.0


_PRESET_COLORS: dict[str, tuple[int, int, int]] = {
    "hair": (244, 114, 114),
    "face": (251, 191, 36),
    "eyes": (96, 165, 250),
    "mouth": (192, 132, 252),
    "skin": (252, 211, 77),
    "clothes": (74, 222, 128),
    "accessories": (14, 165, 233),
}


def _hash_color(label: str) -> tuple[int, int, int]:
    value = abs(hash(label))
    r = 80 + (value & 0x7F)
    g = 80 + ((value >> 7) & 0x7F)
    b = 80 + ((value >> 14) & 0x7F)
    return r, g, b


def _color_for_label(label: str) -> tuple[int, int, int]:
    key = label.lower()
    return _PRESET_COLORS.get(key, _hash_color(label))


def _load_annotation(annotation_path: Path) -> LabelMeAnnotation:
    data = json.loads(annotation_path.read_text())
    return LabelMeAnnotation.model_validate(data)


def _build_path(points: Iterable[tuple[float, float]]) -> skia.Path:
    iterator = iter(points)
    try:
        first = next(iterator)
    except StopIteration:  # pragma: no cover - malformed annotation
        return skia.Path()

    path = skia.Path()
    path.moveTo(float(first[0]), float(first[1]))
    for x, y in iterator:
        path.lineTo(float(x), float(y))
    path.close()
    return path


def _draw_label(canvas: skia.Canvas, label: str, anchor: tuple[float, float], color: tuple[int, int, int], *, style: PreviewStyle, image_bounds: tuple[int, int]) -> None:
    text = label.strip()
    if not text:
        return

    font = skia.Font(None, style.label_text_size)
    metrics = font.getMetrics()
    text_width = font.measureText(text)

    ascent = metrics.fAscent if hasattr(metrics, "fAscent") else metrics[0]
    descent = metrics.fDescent if hasattr(metrics, "fDescent") else metrics[1]

    text_height = abs(ascent) + abs(descent)
    padding = style.label_text_size * 0.4

    x = float(anchor[0])
    y = float(anchor[1])

    max_x = image_bounds[0] - 1
    max_y = image_bounds[1] - 1

    x = min(max(x, 0.0), max_x - text_width - padding)
    y = min(max(y, text_height + padding), max_y - padding)

    background_rect = skia.Rect.MakeXYWH(
        x - padding,
        y - text_height - padding * 0.5,
        text_width + padding * 2,
        text_height + padding,
    )

    r, g, b = color
    bg_color = skia.ColorSetARGB(int(style.label_background_opacity * 255), r, g, b)
    text_color = skia.ColorWHITE

    bg_paint = skia.Paint(Color=bg_color, Style=skia.Paint.kFill_Style, AntiAlias=True)
    text_paint = skia.Paint(Color=text_color, AntiAlias=True)

    canvas.drawRect(background_rect, bg_paint)
    canvas.drawString(text, x, y - descent, font, text_paint)


def render_annotation_preview(
    image_path: Path,
    annotation_path: Path,
    output_path: Path,
    *,
    style: PreviewStyle | None = None,
) -> Path:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not annotation_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    preview_style = style or PreviewStyle()

    annotation = _load_annotation(annotation_path)
    image = skia.Image.open(str(image_path))

    surface = skia.Surface(image.width(), image.height())
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorBLACK)
    canvas.drawImage(image, 0, 0)

    for shape in annotation.shapes:
        if shape.shape_type != "polygon" or len(shape.points) < 3:
            continue

        r, g, b = _color_for_label(shape.label)
        fill_color = skia.ColorSetARGB(int(preview_style.fill_opacity * 255), r, g, b)
        stroke_color = skia.ColorSetARGB(255, r, g, b)

        path = _build_path(shape.points)
        if path.isEmpty():
            continue

        fill_paint = skia.Paint(
            Color=fill_color,
            Style=skia.Paint.kFill_Style,
            AntiAlias=True,
        )
        stroke_paint = skia.Paint(
            Color=stroke_color,
            Style=skia.Paint.kStroke_Style,
            StrokeWidth=preview_style.stroke_width,
            AntiAlias=True,
        )

        canvas.drawPath(path, fill_paint)
        canvas.drawPath(path, stroke_paint)

        anchor = shape.points[0]
        _draw_label(
            canvas,
            label=shape.label,
            anchor=anchor,
            color=(r, g, b),
            style=preview_style,
            image_bounds=(image.width(), image.height()),
        )

    snapshot = surface.makeImageSnapshot()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save(str(output_path), skia.kPNG)
    return output_path
