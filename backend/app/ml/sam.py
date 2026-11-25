"""Interactive segmentation with SAM 2.1 (Ravi et al., 2024), run locally, plus our focus mode.

Plain mode: the image encoder runs once per image (features cached, LRU), every
click only runs the light prompt encoder + mask decoder on the whole image.

Focus mode (for small objects, inspired by the focus view of FocalClick,
Chen et al. 2022): SAM resizes every input to 1024 px, so a 20 px apple in a
512 px crop stays tiny. We first decode on the whole image to find roughly
where the object is, then run SAM again on a window around it, about 3x the
object size. That window is upscaled to 1024 px, so the object fills SAM's
view. The window is fixed by the first click (or the box) and is cached, so
further clicks on the same object reuse its encoding and the previous mask.

Library: the official `sam2` package (facebookresearch/sam2). Weights come from
the Hugging Face Hub on first use (cached in HF_HOME, the data volume).
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import SAM_MODEL_ID
from .masks import mask_to_polygon

FEATURE_CACHE = 24  # encoded images / focus windows kept in memory
IMAGE_CACHE = 8

# focus mode parameters (pixels in the original image)
FOCUS_MAX_OBJECT = 160  # objects larger than this are segmented in plain mode
FOCUS_CONTEXT = 3.0  # window side = object size x this
FOCUS_MIN_SIDE = 96
FOCUS_MAX_SIDE = 384
FOCUS_POINT_MARGIN = 16  # every click stays at least this far inside the window
FOCUS_GRID = 16  # window corners snap to this grid, so small changes reuse the cache


class SamUnavailable(RuntimeError):
    pass


def _no_grad():
    try:
        import torch
        return torch.inference_mode()
    except ImportError:  # unit tests run a fake predictor without PyTorch
        import contextlib
        return contextlib.nullcontext()


def focus_window(region: tuple[float, float, float, float], points: list[tuple[float, float]],
                 height: int, width: int) -> tuple[int, int, int, int]:
    """Window (x0, y0, x1, y1) around `region` with context, containing every point, inside the image."""
    rx0, ry0, rx1, ry1 = region
    cx, cy = (rx0 + rx1) / 2, (ry0 + ry1) / 2
    side = max(rx1 - rx0, ry1 - ry0, 1.0) * FOCUS_CONTEXT
    side = min(max(side, FOCUS_MIN_SIDE), FOCUS_MAX_SIDE)
    x0, y0, x1, y1 = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
    for px, py in points:
        x0, y0 = min(x0, px - FOCUS_POINT_MARGIN), min(y0, py - FOCUS_POINT_MARGIN)
        x1, y1 = max(x1, px + FOCUS_POINT_MARGIN), max(y1, py + FOCUS_POINT_MARGIN)
    g = FOCUS_GRID
    x0 = max(0, int(math.floor(x0 / g) * g))
    y0 = max(0, int(math.floor(y0 / g) * g))
    x1 = min(width, int(math.ceil(x1 / g) * g))
    y1 = min(height, int(math.ceil(y1 / g) * g))
    return x0, y0, x1, y1


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


class SamService:
    def __init__(self, model_id: str = SAM_MODEL_ID):
        self.model_id = model_id
        self._predictor = None
        self._lock = threading.Lock()
        self._features: OrderedDict[str, dict] = OrderedDict()
        self._images: OrderedDict[str, np.ndarray] = OrderedDict()
        self._current: str | None = None
        self.device = None

    # ---------------------------------------------------------------- loading and caching
    def _load(self):
        if self._predictor is None:
            try:
                import torch
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as exc:
                raise SamUnavailable(f"SAM 2 is not installed: {exc}") from exc
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._predictor = SAM2ImagePredictor.from_pretrained(self.model_id, device=self.device)
        return self._predictor

    @property
    def loaded(self) -> bool:
        return self._predictor is not None

    def image(self, key: str, path: Path) -> np.ndarray:
        if key in self._images:
            self._images.move_to_end(key)
            return self._images[key]
        arr = np.array(Image.open(path).convert("RGB"))
        self._images[key] = arr
        while len(self._images) > IMAGE_CACHE:
            self._images.popitem(last=False)
        return arr

    def _ensure(self, key: str, image: np.ndarray) -> float:
        """Make `image` (cached under `key`) the predictor's current input. Returns encoder ms (0 if cached)."""
        p = self._load()
        if self._current == key:
            return 0.0
        if key in self._features:
            self._features.move_to_end(key)
            st = self._features[key]
            p._features, p._orig_hw, p._is_image_set = st["features"], st["orig_hw"], True
            self._current = key
            return 0.0
        t0 = time.perf_counter()
        with _no_grad():
            p.set_image(image)
        self._features[key] = {"features": p._features, "orig_hw": p._orig_hw}
        while len(self._features) > FEATURE_CACHE:
            self._features.popitem(last=False)
        self._current = key
        return (time.perf_counter() - t0) * 1000

    def _decode(self, shape: tuple[int, int], coords, labels, box_xyxy, multimask: bool, mask_input):
        """One prompt on the current input. Returns (mask HxW bool, score, low-res logits 1x256x256)."""
        p = self._predictor
        with _no_grad():
            masks, scores, low = p.predict(point_coords=coords, point_labels=labels, box=box_xyxy,
                                           mask_input=mask_input, multimask_output=multimask)
        scores = np.asarray(scores).reshape(-1)
        k = int(np.argmax(scores))
        mask = np.asarray(masks).reshape(-1, *shape)[k] > 0.5
        low = np.asarray(low)
        logits = low.reshape(-1, *low.shape[-2:])[k][None]
        return mask, float(scores[k]), logits

    # ---------------------------------------------------------------- core
    def segment(self, key: str, image: np.ndarray, points: list[list[float]], box: list[float] | None = None,
                focus: bool = False, prev: dict | None = None) -> dict:
        """Segment one object. points: [[x, y, 1|0], ...]; box: [x, y, w, h].

        `prev` is the result of the previous call for the same object; its mask logits are fed
        back in (as in the usual iterative click protocol) when the input window is unchanged.
        Not thread-safe on its own: callers hold the lock.
        """
        h, w = image.shape[:2]
        pts = np.array([[q[0], q[1]] for q in points], np.float32) if points else None
        lbl = np.array([int(q[2]) for q in points], np.int32) if points else None
        box_xyxy = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], np.float32) if box else None
        enc = self._ensure(key, image)
        t0 = time.perf_counter()

        window = None
        if focus:
            if box_xyxy is not None:
                region = tuple(float(v) for v in box_xyxy)
            else:  # where is the object? one cheap decode with the first click on the whole image
                anchor, _, _ = self._decode((h, w), pts[:1], lbl[:1], None, True, None)
                region = _mask_bbox(anchor) or (pts[0, 0] - 8, pts[0, 1] - 8, pts[0, 0] + 8, pts[0, 1] + 8)
            if max(region[2] - region[0], region[3] - region[1]) <= FOCUS_MAX_OBJECT:
                window = focus_window(region, [(float(x), float(y)) for x, y in (pts if pts is not None else [])], h, w)

        if window is None:  # plain mode
            reuse = prev is not None and prev.get("window") is None
            multimask = box_xyxy is None and len(points) <= 1 and not reuse
            mask, score, logits = self._decode((h, w), pts, lbl, box_xyxy, multimask,
                                               prev["logits"] if reuse else None)
            dec = (time.perf_counter() - t0) * 1000
            return {"mask": mask, "score": score, "logits": logits, "window": None,
                    "encode_ms": round(enc, 1), "decode_ms": round(dec, 1)}

        x0, y0, x1, y1 = window
        crop_enc = self._ensure(f"{key}|{x0},{y0},{x1},{y1}", np.ascontiguousarray(image[y0:y1, x0:x1]))
        enc += crop_enc
        off = np.array([x0, y0], np.float32)
        reuse = prev is not None and prev.get("window") == window
        multimask = box_xyxy is None and len(points) <= 1 and not reuse
        mask_c, score, logits = self._decode(
            (y1 - y0, x1 - x0),
            pts - off if pts is not None else None, lbl,
            box_xyxy - np.concatenate([off, off]) if box_xyxy is not None else None,
            multimask, prev["logits"] if reuse else None)
        mask = np.zeros((h, w), bool)
        mask[y0:y1, x0:x1] = mask_c
        dec = (time.perf_counter() - t0) * 1000 - crop_enc  # decoding only, window encoding counted in enc
        return {"mask": mask, "score": score, "logits": logits, "window": window,
                "encode_ms": round(enc, 1), "decode_ms": round(dec, 1)}

    # ---------------------------------------------------------------- API used by the app
    def prepare(self, key: str, path: Path) -> float:
        with self._lock:
            return self._ensure(key, self.image(key, path))

    def predict(self, key: str, path: Path, points: list[list[float]], box: list[float] | None = None,
                focus: bool = False) -> dict:
        with self._lock:
            r = self.segment(key, self.image(key, path), points, box, focus)
        return {"points": mask_to_polygon(r["mask"]), "score": r["score"], "focus_window": r["window"],
                "encode_ms": r["encode_ms"], "decode_ms": r["decode_ms"]}

    def boxes_to_polygons(self, key: str, path: Path, boxes: list[list[float]]) -> dict:
        """Many boxes (accepted detector suggestions) to polygons in one batched decode (plain mode)."""
        if not boxes:
            return {"polygons": [], "scores": []}
        with self._lock:
            image = self.image(key, path)
            enc = self._ensure(key, image)
            t0 = time.perf_counter()
            xyxy = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes], np.float32)
            with _no_grad():
                masks, scores, _ = self._predictor.predict(box=xyxy, multimask_output=False)
            masks = np.asarray(masks)
            masks = masks.reshape(len(boxes), -1, *masks.shape[-2:])[:, 0] > 0.5
            scores = np.asarray(scores).reshape(len(boxes), -1)[:, 0]
            dec = (time.perf_counter() - t0) * 1000
        return {"polygons": [mask_to_polygon(m) for m in masks], "scores": [float(s) for s in scores],
                "encode_ms": round(enc, 1), "decode_ms": round(dec, 1)}


sam_service = SamService()
