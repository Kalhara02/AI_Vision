"""Zero-shot suggestions for any class: an open-vocabulary detector prompted with the project's labels.

Before a project has any finished images there is nothing to fine-tune on, and a COCO detector only
knows COCO's 80 classes. OWLv2 (Minderer, Gritsenko and Houlsby, 2023) scores image regions against
free-text queries, so a project with the classes "pothole, crack" or "tumor" gets suggestions from the
first image. The weights come from Hugging Face and run locally on the CPU, like SAM.

What is ours: prompt building, decoding the raw outputs into pixel boxes (OWLv2 pads the image to a
square, OWL-ViT v1 stretches it), per-class non-maximum suppression, score rescaling for the UI and
an LRU cache, so moving the threshold slider does not run the model again.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

import numpy as np

from ..config import ZERO_SHOT_ENABLED, ZERO_SHOT_MID, ZERO_SHOT_MODEL

RAW_MIN = 0.03  # raw OWL score kept in the cache; the UI threshold filters later
MAX_DETS = 300


class ZeroShotUnavailable(RuntimeError):
    pass


def prompt_for(name: str) -> str:
    """'traffic_light' -> 'a photo of a traffic light' (the CLIP-style prompt OWL was trained with)."""
    return f"a photo of a {name.replace('_', ' ').replace('-', ' ').strip().lower()}"


def calibrate(raw: float, mid: float = ZERO_SHOT_MID) -> float:
    """Map a raw OWL score to the 0..1 scale of the UI slider: `mid` -> 0.5, 2*mid and above -> 1."""
    return float(min(1.0, raw / (2 * mid)))


def decode(logits: np.ndarray, boxes: np.ndarray, width: int, height: int, padded_square: bool,
           raw_min: float = RAW_MIN) -> list[dict]:
    """Turn OWL outputs into pixel boxes. logits (P, Q) are per box and query, boxes (P, 4) are
    normalised cx, cy, w, h. OWLv2 normalises against the padded square, OWL-ViT v1 against the image."""
    scores_all = 1 / (1 + np.exp(-logits))
    q = scores_all.argmax(1)
    s = scores_all[np.arange(len(q)), q]
    keep = s >= raw_min
    sx, sy = (max(width, height),) * 2 if padded_square else (width, height)
    out = []
    for (cx, cy, bw, bh), qi, si in zip(boxes[keep].tolist(), q[keep].tolist(), s[keep].tolist()):
        # .tolist() gives plain Python floats: NumPy float32 cannot be sent as JSON
        x0, y0 = max(0.0, (cx - bw / 2) * sx), max(0.0, (cy - bh / 2) * sy)
        x1, y1 = min(float(width), (cx + bw / 2) * sx), min(float(height), (cy + bh / 2) * sy)
        if x1 - x0 > 1 and y1 - y0 > 1:
            out.append({"bbox": [x0, y0, x1 - x0, y1 - y0], "query": int(qi), "raw": float(si)})
    return out


def nms(dets: list[dict], iou: float = 0.5) -> list[dict]:
    """Greedy non-maximum suppression per query (class), highest score first."""
    from .metrics import box_iou_xywh

    kept: list[dict] = []
    for d in sorted(dets, key=lambda d: -d["raw"]):
        same = [k for k in kept if k["query"] == d["query"]]
        if same and box_iou_xywh(np.array([d["bbox"]]), np.array([k["bbox"] for k in same])).max() > iou:
            continue
        kept.append(d)
        if len(kept) >= MAX_DETS:
            break
    return kept


class ZeroShotService:
    def __init__(self, model_id: str = ZERO_SHOT_MODEL, cache_size: int = 64):
        self.model_id = model_id
        self._model = self._processor = None
        self._padded = "owlv2" in model_id.lower()
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple, list[dict]] = OrderedDict()
        self._cache_size = cache_size

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if not ZERO_SHOT_ENABLED:
            raise ZeroShotUnavailable("zero-shot suggestions are disabled (ZERO_SHOT_ENABLED=0)")
        if self._model is None:
            try:
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            except ImportError as exc:  # pragma: no cover - depends on the image
                raise ZeroShotUnavailable(f"transformers is not installed: {exc}")
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).eval()

    def detect(self, key: str, image: np.ndarray, names: list[str]) -> tuple[list[dict], float]:
        """Raw detections for `names` on an RGB image: [{bbox, query, raw}], query = index into names."""
        import torch

        ck = (key, tuple(names))
        with self._lock:
            if ck in self._cache:
                self._cache.move_to_end(ck)
                return self._cache[ck], 0.0
            self._load()
            t0 = time.perf_counter()
            inputs = self._processor(text=[[prompt_for(n) for n in names]], images=image, return_tensors="pt")
            with torch.inference_mode():
                out = self._model(**inputs)
            h, w = image.shape[:2]
            dets = nms(decode(out.logits[0].float().numpy(), out.pred_boxes[0].float().numpy(), w, h, self._padded))
            ms = (time.perf_counter() - t0) * 1000
            self._cache[ck] = dets
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return dets, ms


zero_shot_service = ZeroShotService()
