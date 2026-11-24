"""Small geometry helpers shared by the API, the exporters and the evaluation.

Pure Python on purpose: no framework imports, easy to unit test.
"""
from __future__ import annotations

from typing import Sequence

Point = Sequence[float]
BBox = Sequence[float]  # [x, y, w, h]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bbox_from_points(points: Sequence[Point]) -> list[float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, y0 = min(xs), min(ys)
    return [x0, y0, max(xs) - x0, max(ys) - y0]


def polygon_area(points: Sequence[Point]) -> float:
    """Shoelace formula. Returns the absolute area."""
    n = len(points)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[(i + 1) % n][0], points[(i + 1) % n][1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def clip_bbox(b: BBox, width: float, height: float) -> list[float]:
    x0 = clamp(b[0], 0, width)
    y0 = clamp(b[1], 0, height)
    x1 = clamp(b[0] + b[2], 0, width)
    y1 = clamp(b[1] + b[3], 0, height)
    return [x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0)]


def clip_points(points: Sequence[Point], width: float, height: float) -> list[list[float]]:
    return [[clamp(p[0], 0, width), clamp(p[1], 0, height)] for p in points]


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0.0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0
