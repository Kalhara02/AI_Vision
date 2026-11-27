"""Score a detector the app trained (e.g. the released v5) on the MinneApple test crops.

The learning curve in detector_curve.py trains its own models offline. This script answers the
other question: how good is the model the app actually deployed after its training rounds?
Same test crops, same pycocotools metrics, so the numbers are directly comparable.

    docker compose run --rm backend python eval/eval_app_model.py --project 5
    docker compose run --rm backend python eval/eval_app_model.py --project 5 --version 3   # the starting model

Writes eval/results/app_model_p<project>_v<version>.json (about 25 minutes on a laptop CPU).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import load_coco, load_rgb, results_dir
from detector_curve import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=int, required=True)
    ap.add_argument("--version", type=int, help="default: the latest version of that project")
    ap.add_argument("--data", type=Path, default=Path("data/minneapple"))
    args = ap.parse_args()

    import torch

    from app.config import MODEL_DIR
    from app.ml.detector import build_finetune, detect, device

    src = Path(MODEL_DIR) / f"project_{args.project}"
    versions = sorted(int(p.stem.split("_v")[1]) for p in src.glob("detector_v*.pt"))
    if not versions:
        raise SystemExit(f"no detector in {src}")
    v = args.version or versions[-1]
    model = build_finetune()
    model.load_state_dict(torch.load(src / f"detector_v{v}.pt", map_location="cpu"))
    model = model.to(device()).eval()

    test_path = args.data / "coco_crops_test.json"
    test = load_coco(test_path)
    img_dir = args.data / "crops_test" / "images"
    t0, dets = time.time(), []
    for k, im in enumerate(test["images"], 1):
        for d in detect(model, load_rgb(img_dir / im["file_name"]), 1, threshold=0.05):
            dets.append({"image_id": im["id"], "category_id": 1, "bbox": d["bbox"], "score": d["score"]})
        if k % 50 == 0:
            print(f"{k}/{len(test['images'])} crops", flush=True)
    m = {k: round(val, 4) for k, val in evaluate(test_path, dets).items()}
    res = {"project": args.project, "version": v, "test_crops": len(test["images"]), **m,
           "seconds": round(time.time() - t0, 1)}
    out = results_dir() / f"app_model_p{args.project}_v{v}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
