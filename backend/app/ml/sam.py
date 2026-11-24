"""Interactive segmentation with SAM 2.1 (Ravi et al., 2024), run locally.

The heavy image encoder runs once per image; its features are cached (LRU), so
every further click only runs the light prompt encoder + mask decoder.

Library: the official `sam2` package (facebookresearch/sam2). Weights are
downloaded from the Hugging Face Hub on first use into HF_HOME (data volume).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import SAM_MODEL_ID
from .masks import mask_to_polygon

CACHE_SIZE = 12


class SamUnavailable(RuntimeError):
    pass


class SamService:
    def __init__(self, model_id: str = SAM_MODEL_ID):
        self.model_id = model_id
        self._predictor = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._current: str | None = None
        self.device = None

    # ---------------------------------------------------------------- loading
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

    def _set_image(self, key: str, path: Path) -> float:
        """Make `path` the predictor's current image. Returns encoder time in ms (0 if cached)."""
        import torch

        p = self._load()
        if self._current == key:
            return 0.0
        if key in self._cache:
            self._cache.move_to_end(key)
            st = self._cache[key]
            p._features, p._orig_hw, p._is_image_set = st["features"], st["orig_hw"], True
            self._current = key
            return 0.0
        t0 = time.perf_counter()
        image = np.array(Image.open(path).convert("RGB"))
        with torch.inference_mode():
            p.set_image(image)
        self._cache[key] = {"features": p._features, "orig_hw": p._orig_hw}
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        self._current = key
        return (time.perf_counter() - t0) * 1000

    # ---------------------------------------------------------------- API
    def prepare(self, key: str, path: Path) -> float:
        with self._lock:
            return self._set_image(key, path)

    def predict(self, key: str, path: Path, points: list[list[float]], box: list[float] | None = None) -> dict:
        """points: [[x, y, 1|0], ...] (1 = object, 0 = background); box: [x, y, w, h]."""
        import torch

        with self._lock:
            enc_ms = self._set_image(key, path)
            p = self._predictor
            t0 = time.perf_counter()
            coords = np.array([[q[0], q[1]] for q in points], np.float32) if points else None
            labels = np.array([int(q[2]) for q in points], np.int32) if points else None
            box_xyxy = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]], np.float32) if box else None
            # one click is ambiguous (apple vs. cluster vs. branch): ask for 3 masks, keep the best
            multimask = box is None and len(points) <= 1
            with torch.inference_mode():
                masks, scores, _ = p.predict(point_coords=coords, point_labels=labels, box=box_xyxy,
                                             multimask_output=multimask)
            masks = np.asarray(masks)
            scores = np.asarray(scores).reshape(-1)
            k = int(np.argmax(scores))
            mask = masks.reshape(-1, *masks.shape[-2:])[k] > 0.5
            dec_ms = (time.perf_counter() - t0) * 1000
        return {"points": mask_to_polygon(mask), "score": float(scores[k]),
                "encode_ms": round(enc_ms, 1), "decode_ms": round(dec_ms, 1)}

    def boxes_to_polygons(self, key: str, path: Path, boxes: list[list[float]]) -> dict:
        """Turn many boxes (e.g. accepted detector proposals) into polygons in one batched call."""
        import torch

        if not boxes:
            return {"polygons": [], "scores": []}
        with self._lock:
            enc_ms = self._set_image(key, path)
            p = self._predictor
            t0 = time.perf_counter()
            xyxy = np.array([[b[0], b[1], b[0] + b[2], b[1] + b[3]] for b in boxes], np.float32)
            with torch.inference_mode():
                masks, scores, _ = p.predict(box=xyxy, multimask_output=False)
            masks = np.asarray(masks)
            masks = masks.reshape(len(boxes), -1, *masks.shape[-2:])[:, 0] > 0.5
            scores = np.asarray(scores).reshape(len(boxes), -1)[:, 0]
            dec_ms = (time.perf_counter() - t0) * 1000
        return {"polygons": [mask_to_polygon(m) for m in masks], "scores": [float(s) for s in scores],
                "encode_ms": round(enc_ms, 1), "decode_ms": round(dec_ms, 1)}


sam_service = SamService()
