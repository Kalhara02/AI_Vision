"""Failure cases for the report, collected automatically.

- SAM: the apples with the lowest IoU after 5 clicks (from sam_noc_objects.csv)
- Detector: missed apples (FN) and false alarms (FP) at score >= 0.5 for one model

    docker compose run --rm backend python eval/failure_gallery.py --detections eval/results/detections_n100.json

Writes: eval/results/failures_sam.png, failures_detector.png (green = ground truth,
red = false positive, yellow = missed). Sort the examples into groups yourself
(leaf occlusion, merged clusters, green apples, border cuts, shadow) for the report.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from common import anns_by_image, load_coco, load_mask, load_rgb, results_dir


def tile(img: np.ndarray, size: int = 300) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def grid(tiles: list[np.ndarray], cols: int = 4) -> np.ndarray:
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def iou_xywh(a, b) -> float:
    iw = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = a[2] * a[3] + b[2] * b[3] - inter
    return inter / u if u else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/minneapple"))
    ap.add_argument("--detections", type=Path, help="detections_n<k>.json from detector_curve.py")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sam-mode", default="focus", choices=["plain", "focus"])
    args = ap.parse_args()
    out = results_dir()
    coco = load_coco(args.data / "coco_crops_test.json")
    imgs = {im["id"]: im for im in coco["images"]}
    by_name = {im["file_name"]: im for im in coco["images"]}
    img_dir, mask_dir = args.data / "crops_test" / "images", args.data / "crops_test" / "masks"

    sam_csv = out / "sam_noc_objects.csv"
    if sam_csv.exists():
        rows = [r for r in csv.DictReader(open(sam_csv)) if r.get("mode", args.sam_mode) == args.sam_mode]
        rows = sorted(rows, key=lambda r: float(r["iou5"]))[: args.n]
        tiles = []
        for r in rows:
            im = by_name[r["image"]]
            rgb = load_rgb(img_dir / im["file_name"]).copy()
            gt = (load_mask(mask_dir / im["mask_file"]) == int(r["instance_id"])).astype(np.uint8)
            cs, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cs, -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(gt)
            pad = max(w, h) * 2 + 20
            cx, cy = x + w // 2, y + h // 2
            crop = rgb[max(0, cy - pad):cy + pad, max(0, cx - pad):cx + pad]
            t = tile(crop)
            cv2.putText(t, f"IoU@5 {float(r['iou5']):.2f}  NoC90 {r['noc90']}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            tiles.append(t)
        if tiles:
            cv2.imwrite(str(out / "failures_sam.png"), cv2.cvtColor(grid(tiles), cv2.COLOR_RGB2BGR))
            print("wrote failures_sam.png")

    if args.detections:
        dets = json.loads(args.detections.read_text())
        by_img = anns_by_image(coco)
        scored = []
        for iid, gts in by_img.items():
            ds = [d for d in dets if d["image_id"] == iid and d["score"] >= 0.5]
            fp = [d for d in ds if all(iou_xywh(d["bbox"], g["bbox"]) < 0.5 for g in gts)]
            fn = [g for g in gts if all(iou_xywh(d["bbox"], g["bbox"]) < 0.5 for d in ds)]
            scored.append((len(fp) + len(fn), iid, fp, fn, gts))
        tiles = []
        for _, iid, fp, fn, gts in sorted(scored, reverse=True)[: args.n]:
            rgb = load_rgb(img_dir / imgs[iid]["file_name"]).copy()
            for g in gts:
                x, y, w, h = map(int, g["bbox"])
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 1)
            for g in fn:
                x, y, w, h = map(int, g["bbox"])
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 220, 0), 3)
            for d in fp:
                x, y, w, h = map(int, d["bbox"])
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 3)
            t = tile(rgb)
            cv2.putText(t, f"FP {len(fp)}  FN {len(fn)}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            tiles.append(t)
        if tiles:
            cv2.imwrite(str(out / "failures_detector.png"), cv2.cvtColor(grid(tiles), cv2.COLOR_RGB2BGR))
            print("wrote failures_detector.png")


if __name__ == "__main__":
    main()
