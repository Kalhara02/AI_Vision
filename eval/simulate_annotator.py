"""Simulated annotator: estimated labeling time per crop for three strategies.

A reproducible complement to the user study: the ground truth "labels" every
test crop, and every action gets a time cost.

  manual          draw each apple as a polygon, vertex by vertex
  sam             click each apple with the NoC protocol until IoU >= 0.85 (clicks from sam_noc.py)
  detector+sam    accept all suggestions, refine them with SAM (R), delete false positives,
                  click in missed apples with SAM; done once per detector model given

Action costs come from your own user-study logs when --logs is given (median
times between the matching events), otherwise from the defaults below. The costs
used are written to the output, so the report can state them.

    docker compose run --rm backend python eval/simulate_annotator.py \
        --detections eval/results/detections_n0.json eval/results/detections_n100.json \
        --logs data/logs.json

Writes: eval/results/simulated_effort.csv, simulated_effort.json, simulated_effort.png
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import anns_by_image, load_coco, load_mask, mean_sd, results_dir

DEFAULT_COSTS = {  # seconds; assumptions unless calibrated from logs
    "vertex": 1.2,      # place one polygon vertex
    "object": 1.5,      # per manual object: choose class, close the polygon
    "sam_click": 1.5,   # aim + click + look at the SAM preview
    "sam_accept": 0.8,  # press Enter on a good preview
    "accept_all": 1.0,  # press A once per image
    "verify": 0.4,      # glance at one accepted suggestion
    "refine": 1.5,      # press R once per image and wait
    "delete": 1.5,      # find and remove one false positive
}


def calibrate(logs: list[dict]) -> dict:
    """Median gap between consecutive events of the same kind on the same image (gaps > 10 s ignored)."""
    ev = sorted(logs, key=lambda e: e["t"])
    gaps = defaultdict(list)
    last_by_img: dict[tuple, dict] = {}
    for e in ev:
        key = (e.get("payload", {}).get("participant") or e["session_id"], e.get("image_id"))
        prev = last_by_img.get(key)
        if prev is not None:
            dt = (e["t"] - prev["t"]) / 1000
            if 0 < dt <= 10:
                if e["event"] == "poly_point" and prev["event"] == "poly_point" and e["mode"] == "manual":
                    gaps["vertex"].append(dt)
                elif e["event"] == "sam_click" and prev["event"] in ("sam_click", "sam_predict"):
                    gaps["sam_click"].append(dt)
                elif e["event"] == "sam_accept":
                    gaps["sam_accept"].append(dt)
                elif e["event"] in ("proposal_rejected", "ann_deleted"):
                    gaps["delete"].append(dt)
        last_by_img[key] = e
    return {k: round(float(np.median(v)), 2) for k, v in gaps.items() if len(v) >= 5}


def manual_vertices(mask: np.ndarray, epsilon: float) -> int:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 0
    c = max(cs, key=cv2.contourArea)
    return max(3, len(cv2.approxPolyDP(c, epsilon, True)))


def iou_xywh(a, b) -> float:
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/minneapple"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--detections", type=Path, nargs="*", default=[])
    ap.add_argument("--threshold", type=float, default=0.5, help="suggestion score threshold (as in the app)")
    ap.add_argument("--logs", type=Path, help="user-study logs to calibrate action costs")
    ap.add_argument("--epsilon", type=float, default=2.0, help="polygon simplification for manual vertex count (px)")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--sam-mode", default="focus", choices=["plain", "focus"],
                    help="which SAM mode's clicks to use from sam_noc_objects.csv (the app uses focus)")
    args = ap.parse_args()
    out = results_dir()

    costs = dict(DEFAULT_COSTS)
    calibrated = {}
    if args.logs:
        calibrated = calibrate(json.loads(args.logs.read_text()))
        costs.update(calibrated)
    print("action costs (s):", costs, "| calibrated from logs:", sorted(calibrated) or "none")

    coco = load_coco(args.data / f"coco_crops_{args.split}.json")
    images = coco["images"][: args.max_images] if args.max_images else coco["images"]
    by_img = anns_by_image(coco)
    mask_dir = args.data / f"crops_{args.split}" / "masks"

    # SAM clicks per object from sam_noc.py (objects it did not sample get the median)
    noc, box_iou = {}, {}
    sam_csv = out / "sam_noc_objects.csv"
    if sam_csv.exists():
        for r in csv.DictReader(open(sam_csv)):
            if r.get("mode", args.sam_mode) != args.sam_mode:
                continue
            noc[(r["image"], int(r["instance_id"]))] = int(r["noc85"])
            box_iou[(r["image"], int(r["instance_id"]))] = float(r["box_iou"])
    else:
        print("note: run sam_noc.py first for measured clicks; using 2 clicks per apple")
    noc_med = float(np.median(list(noc.values()))) if noc else 2.0
    det_sets = {}
    for p in args.detections:
        m = re.search(r"n(\d+)", p.stem)
        dets = defaultdict(list)
        for d in json.loads(p.read_text()):
            if d["score"] >= args.threshold:
                dets[d["image_id"]].append(d)
        det_sets[f"detector+sam (n={m.group(1) if m else p.stem})"] = dets

    rows = []
    for im in images:
        inst = load_mask(mask_dir / im["mask_file"])
        gts = by_img[im["id"]]
        row = {"image": im["file_name"], "apples": len(gts)}
        clicks = lambda a: noc.get((im["file_name"], a["attributes"]["instance_id"]), noc_med)  # noqa: E731
        # manual
        verts = sum(manual_vertices(inst == a["attributes"]["instance_id"], args.epsilon) for a in gts)
        row["manual"] = verts * costs["vertex"] + len(gts) * costs["object"]
        # SAM only
        row["sam"] = sum(clicks(a) * costs["sam_click"] + costs["sam_accept"] for a in gts)
        # detector + SAM, per model
        for name, dets in det_sets.items():
            ds = sorted(dets.get(im["id"], []), key=lambda d: -d["score"])
            matched, fp = set(), 0
            for d in ds:
                best, bj = 0.0, None
                for j, g in enumerate(gts):
                    if j not in matched:
                        v = iou_xywh(d["bbox"], g["bbox"])
                        if v > best:
                            best, bj = v, j
                if best >= 0.5:
                    matched.add(bj)
                else:
                    fp += 1
            t = (costs["accept_all"] + costs["refine"]) if ds else 0.0
            t += len(ds) * costs["verify"] + fp * costs["delete"]
            for j, g in enumerate(gts):
                key = (im["file_name"], g["attributes"]["instance_id"])
                if j not in matched:  # missed by the detector: segment with SAM clicks
                    t += clicks(g) * costs["sam_click"] + costs["sam_accept"]
                elif box_iou.get(key, 1.0) < 0.85:  # box-prompt outline not good enough: fix with clicks
                    t += max(1.0, clicks(g) - 1) * costs["sam_click"]
            row[name] = t
        rows.append(row)

    strategies = ["manual", "sam", *det_sets]
    with open(out / "simulated_effort.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "apples", *strategies])
        w.writeheader()
        w.writerows([{k: (round(v, 1) if isinstance(v, float) else v) for k, v in r.items()} for r in rows])
    man = np.mean([r["manual"] for r in rows])
    summary = {"costs_seconds": costs, "calibrated_from_logs": sorted(calibrated), "n_images": len(rows),
               "threshold": args.threshold, "strategies": {}}
    for s in strategies:
        vals = [r[s] for r in rows]
        summary["strategies"][s] = {"seconds_per_crop": mean_sd(vals),
                                    "mean": round(float(np.mean(vals)), 1),
                                    "saved_vs_manual_percent": round(100 * (1 - np.mean(vals) / man), 1)}
    (out / "simulated_effort.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary["strategies"], indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    means = [summary["strategies"][s]["mean"] for s in strategies]
    sds = [np.std([r[s] for r in rows], ddof=1) for s in strategies]
    ax.bar(range(len(strategies)), means, yerr=sds, capsize=5,
           color=["#999999", "#56B4E9"] + ["#0072B2"] * len(det_sets))
    ax.set_xticks(range(len(strategies)), [s.replace("detector+sam ", "det+SAM\n") for s in strategies])
    ax.set_ylabel("estimated seconds per crop")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "simulated_effort.png", dpi=160)


if __name__ == "__main__":
    main()
