"""Detector quality vs. number of labeled images (the learning curve).

Trains the app's own fine-tuning loop on n ground-truth crops from the `pool`
split and evaluates COCO box mAP on the `test` crops. n = 0 is the pretrained
COCO detector restricted to its "apple" class (the app's cold start).

    docker compose run --rm backend python eval/detector_curve.py --sizes 0 25 50 100 200

Needs: data/minneapple/coco_crops_pool.json, coco_crops_test.json (+ image folders),
made with scripts/make_crops.py (see eval/README.md).
Writes: eval/results/detector_curve.csv, detector_curve.png, detections_n<k>.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from common import anns_by_image, load_coco, load_rgb, results_dir


def evaluate(gt_path: Path, dets: list[dict]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(gt_path))
    if not dets:
        return {k: 0.0 for k in ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AR100")}
    dt = gt.loadRes(dets)
    ev = COCOeval(gt, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    s = ev.stats
    return {"mAP": s[0], "AP50": s[1], "AP75": s[2], "AP_small": s[3], "AP_medium": s[4], "AR100": s[8]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/minneapple"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[0, 25, 50, 100, 200])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=0, help="limit test crops (0 = all), for a quick run")
    args = ap.parse_args()

    from app.ml.detector import Sample, TrainConfig, build_pretrained, detect, device, train_detector

    out = results_dir()
    pool = load_coco(args.data / "coco_crops_pool.json")
    test_path = args.data / "coco_crops_test.json"
    test = load_coco(test_path)
    pool_dir, test_dir = args.data / "crops_pool" / "images", args.data / "crops_test" / "images"
    test_images = test["images"][: args.max_test] if args.max_test else test["images"]
    if args.max_test:  # evaluate against the same subset
        keep = {im["id"] for im in test_images}
        test = {**test, "images": test_images, "annotations": [a for a in test["annotations"] if a["image_id"] in keep]}
        test_path = out / "_test_subset.json"
        test_path.write_text(json.dumps(test))

    by_img = anns_by_image(pool)
    pool_samples = [Sample(pool_dir / im["file_name"], [a["bbox"] for a in by_img[im["id"]]])
                    for im in pool["images"] if by_img[im["id"]]]
    rng = random.Random(args.seed)
    rng.shuffle(pool_samples)
    test_imgs = {im["id"]: load_rgb(test_dir / im["file_name"]) for im in test_images}

    rows = []
    for n in args.sizes:
        t0 = time.time()
        if n == 0:
            model, label = build_pretrained()
            model = model.to(device())
            meta = {"train_seconds": 0}
        else:
            if n > len(pool_samples):
                print(f"skip n={n}: only {len(pool_samples)} pool crops")
                continue
            model, meta = train_detector(pool_samples[:n], TrainConfig(epochs=args.epochs, seed=args.seed))
            label = 1
        dets = []
        for iid, img in test_imgs.items():
            for d in detect(model, img, label, threshold=0.05):
                dets.append({"image_id": iid, "category_id": 1, "bbox": d["bbox"], "score": d["score"]})
        (out / f"detections_n{n}.json").write_text(json.dumps(dets))
        m = evaluate(test_path, dets)
        row = {"n_train_images": n, **{k: round(v, 4) for k, v in m.items()},
               "train_seconds": meta["train_seconds"], "total_seconds": round(time.time() - t0, 1)}
        rows.append(row)
        print(row)
        with open(out / "detector_curve.csv", "w", newline="") as f:  # rewrite after each step
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [r["n_train_images"] for r in rows]
    ax.plot(xs, [r["AP50"] for r in rows], "o-", label="AP50")
    ax.plot(xs, [r["mAP"] for r in rows], "s-", label="mAP@[.5:.95]")
    ax.set_xlabel("labeled training crops (0 = pretrained COCO 'apple')")
    ax.set_ylabel("box AP on test crops")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "detector_curve.png", dpi=160)
    print(f"wrote {out / 'detector_curve.csv'} and .png")


if __name__ == "__main__":
    main()
