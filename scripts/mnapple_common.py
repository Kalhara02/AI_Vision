"""Shared helpers for the MinneApple preparation scripts.

MinneApple stores one mask PNG per image: 0 = background, every apple has its
own integer id. These helpers turn such an instance mask into COCO polygons.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# reuse the app's COCO writer and geometry, so ground truth and exports share one code path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))  # on the host
sys.path.insert(0, str(REPO))  # inside the backend container (/app/app)
from app.export.formats import ExpAnnotation, ExpImage, to_coco  # noqa: E402
from app.geometry import bbox_from_points, polygon_area  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def load_instance_mask(path: Path) -> np.ndarray:
    """Return a 2-D integer array of instance ids (0 = background)."""
    m = np.array(Image.open(path))
    if m.ndim == 3:
        if np.all(m[..., 0] == m[..., 1]) and np.all(m[..., 0] == m[..., 2]):
            m = m[..., 0]
        else:  # colour-coded instances: map every distinct colour to an id
            flat = m[..., :3].reshape(-1, 3).astype(np.int64)
            codes = flat[:, 0] << 16 | flat[:, 1] << 8 | flat[:, 2]
            _, ids = np.unique(codes, return_inverse=True)
            m = ids.reshape(m.shape[:2])  # background colour (black) sorts first -> 0
    return m.astype(np.int32)


def instance_to_polygon(binary: np.ndarray, epsilon: float = 1.0) -> tuple[list[list[float]] | None, int]:
    """Largest outer contour of a binary mask, simplified. Returns (points, n_components)."""
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, 0
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2)
    if len(c) < 3:
        return None, len(contours)
    return [[float(x), float(y)] for x, y in c], len(contours)


def mask_to_annotations(mask: np.ndarray, image_id: int, min_area: int = 20,
                        full_areas: dict[int, int] | None = None) -> tuple[list[ExpAnnotation], dict]:
    """Convert every instance in `mask` into a polygon annotation.

    `full_areas` (instance id -> pixel area in the uncropped image) marks
    instances cut by a crop border as truncated.
    """
    anns, stats = [], {"dropped_small": 0, "dropped_degenerate": 0, "multi_part": 0}
    for inst in np.unique(mask):
        if inst == 0:
            continue
        binary = mask == inst
        area_px = int(binary.sum())
        if area_px < min_area:
            stats["dropped_small"] += 1
            continue
        pts, parts = instance_to_polygon(binary)
        if pts is None or polygon_area(pts) <= 0:
            stats["dropped_degenerate"] += 1
            continue
        if parts > 1:
            stats["multi_part"] += 1
        a = ExpAnnotation(image_id=image_id, category=0, kind="polygon", bbox=bbox_from_points(pts), points=pts)
        a.source = "ground_truth"
        # kept for evaluation: the pixel-exact mask beats the simplified polygon
        a.instance_id = int(inst)  # type: ignore[attr-defined]
        a.pixel_area = area_px  # type: ignore[attr-defined]
        if full_areas is not None:
            a.truncated = area_px < full_areas.get(int(inst), area_px)  # type: ignore[attr-defined]
        anns.append(a)
    return anns, stats


def write_coco(path: Path, images: list[ExpImage], anns: list[ExpAnnotation], extra: dict[int, dict],
               description: str) -> None:
    coco = to_coco(images, anns, ["apple"], description=description)
    for ann_json, a in zip(coco["annotations"], anns):
        ann_json["attributes"] = {
            "source": "ground_truth",
            "instance_id": getattr(a, "instance_id", None),
            "pixel_area": getattr(a, "pixel_area", None),
            "truncated": getattr(a, "truncated", False),
        }
    for img_json in coco["images"]:
        img_json.update(extra.get(img_json["id"], {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coco))
