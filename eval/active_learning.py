"""Does the active-learning queue help? Uncertainty sampling vs. random sampling.

Simulates the app's loop with ground truth as the annotator: start with
--start random crops, then each round add --step crops chosen either randomly
or by the app's own uncertainty score (sum of binary entropies of detector
scores, app/ml/metrics.py), retrain with the app's training loop, and measure
COCO AP50 on test crops. Same seed and same start set for both strategies.

    docker compose run --rm backend python eval/active_learning.py --rounds 4

Heavy on CPU (one training per round and strategy); use the GPU compose file if you can.
Writes: eval/results/active_learning.csv, active_learning.png
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from common import anns_by_image, load_coco, load_rgb, results_dir
from detector_curve import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/minneapple"))
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=300, help="random pool subset scored each round")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from app.ml.detector import Sample, TrainConfig, detect, train_detector
    from app.ml.metrics import image_uncertainty

    out = results_dir()
    pool = load_coco(args.data / "coco_crops_pool.json")
    test = load_coco(args.data / "coco_crops_test.json")
    pool_dir, test_dir = args.data / "crops_pool" / "images", args.data / "crops_test" / "images"
    by_img = anns_by_image(pool)
    samples = [Sample(pool_dir / im["file_name"], [a["bbox"] for a in by_img[im["id"]]])
               for im in pool["images"] if by_img[im["id"]]]
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    samples = samples[: args.candidates + args.start]
    start = samples[: args.start]
    rest = samples[args.start:]

    test_images = test["images"][: args.max_test]
    keep = {im["id"] for im in test_images}
    sub = {**test, "images": test_images, "annotations": [a for a in test["annotations"] if a["image_id"] in keep]}
    test_path = out / "_al_test_subset.json"
    test_path.write_text(json.dumps(sub))
    test_imgs = {im["id"]: load_rgb(test_dir / im["file_name"]) for im in test_images}

    def test_ap(model) -> dict:
        dets = [{"image_id": iid, "category_id": 1, "bbox": d["bbox"], "score": d["score"]}
                for iid, img in test_imgs.items() for d in detect(model, img, 1, 0.05)]
        return evaluate(test_path, dets)

    rows = []
    for strategy in ("random", "uncertainty"):
        labeled, unlabeled = list(start), list(rest)
        srng = random.Random(args.seed + 1)
        for r in range(args.rounds + 1):
            model, meta = train_detector(labeled, TrainConfig(epochs=args.epochs, seed=args.seed))
            m = test_ap(model)
            rows.append({"strategy": strategy, "round": r, "n_labeled": len(labeled),
                         "AP50": round(m["AP50"], 4), "mAP": round(m["mAP"], 4), "train_seconds": meta["train_seconds"]})
            print(rows[-1])
            if r == args.rounds or not unlabeled:
                break
            if strategy == "random":
                srng.shuffle(unlabeled)
                pick = unlabeled[: args.step]
            else:
                scored = [(image_uncertainty([d["score"] for d in detect(model, load_rgb(s.path), 1, 0.05)]), k)
                          for k, s in enumerate(unlabeled)]
                top = {k for _, k in sorted(scored, reverse=True)[: args.step]}
                pick = [s for k, s in enumerate(unlabeled) if k in top]
            labeled += pick
            unlabeled = [s for s in unlabeled if s not in pick]
            with open(out / "active_learning.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

    with open(out / "active_learning.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for strategy, style in (("random", "o--"), ("uncertainty", "s-")):
        rs = [r for r in rows if r["strategy"] == strategy]
        ax.plot([r["n_labeled"] for r in rs], [r["AP50"] for r in rs], style, label=strategy)
    ax.set_xlabel("labeled crops")
    ax.set_ylabel("AP50 on test crops")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "active_learning.png", dpi=160)


if __name__ == "__main__":
    main()
