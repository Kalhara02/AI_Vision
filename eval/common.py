"""Shared helpers for the evaluation scripts.

Run the scripts inside the backend container (it has PyTorch, SAM 2 and pycocotools):
    docker compose run --rm backend python eval/<script>.py --help
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (REPO, REPO / "backend"):  # container: /app ; host: <repo>/backend
    if (p / "app").is_dir():
        sys.path.insert(0, str(p))
        break

RESULTS = HERE / "results"


def results_dir(sub: str = "") -> Path:
    d = RESULTS / sub if sub else RESULTS
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_coco(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def anns_by_image(coco: dict) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {im["id"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        out.setdefault(a["image_id"], []).append(a)
    return out


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.int32)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def size_bucket(area: float) -> str:
    """COCO size buckets."""
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def mean_sd(xs) -> str:
    xs = np.asarray(list(xs), float)
    if not len(xs):
        return "n/a"
    return f"{xs.mean():.2f} ± {xs.std(ddof=1) if len(xs) > 1 else 0:.2f}"
