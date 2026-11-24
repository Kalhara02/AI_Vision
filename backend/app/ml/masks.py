"""Mask <-> polygon conversion and mask IoU. NumPy + OpenCV only, unit tested."""
from __future__ import annotations

import cv2
import numpy as np


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected region (SAM sometimes adds small specks)."""
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:
        return m.astype(bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == k


def mask_to_polygon(mask: np.ndarray, epsilon: float = 1.0, min_area: int = 4) -> list[list[float]] | None:
    """Outer contour of the largest region, simplified with Douglas-Peucker."""
    m = largest_component(mask)
    if m.sum() < min_area:
        return None
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
    if len(c) < 3:
        return None
    return [[float(x), float(y)] for x, y in c]


def polygon_to_mask(points: list[list[float]], height: int, width: int) -> np.ndarray:
    m = np.zeros((height, width), np.uint8)
    pts = np.round(np.asarray(points, np.float64)).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0
