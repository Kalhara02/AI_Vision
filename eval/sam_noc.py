"""Interactive segmentation quality of SAM 2.1 on apples: plain vs. focus mode (NoC protocol).

Uses exactly the app's code path (app/ml/sam.py, SamService.segment), so the
numbers describe the tool, not a separate script.

Click simulation as in RITM (Sofiiuk et al. 2022): the first click goes to the
point inside the apple farthest from its border; each next click goes to the
centre of the largest error region, positive for missed apple, negative for
wrongly included background. The previous mask logits are fed back in.

Reports NoC@80/85/90 (clicks to reach that IoU; the maximum counts as a
failure penalty), mean IoU after k clicks, and IoU from a box prompt (the
detector -> SAM path), per mode and per apple size.

    docker compose run --rm backend python eval/sam_noc.py --max-objects 400

Writes: eval/results/sam_noc_objects.csv, sam_noc_summary.json, sam_miou_curve.png
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np

from common import anns_by_image, load_coco, load_mask, load_rgb, results_dir, size_bucket

THRESHOLDS = (0.80, 0.85, 0.90)


def farthest_point(region: np.ndarray) -> tuple[int, int, float]:
    d = cv2.distanceTransform(np.pad(region, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(x), int(y), float(d[y, x])


def next_click(gt: np.ndarray, pred: np.ndarray) -> list:
    xn, yn, dn = farthest_point(gt & ~pred)
    xp, yp, dp = farthest_point(pred & ~gt)
    return [xn, yn, 1] if dn >= dp else [xp, yp, 0]


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def run_object(sam, key, image, gt, focus: bool, max_clicks: int) -> tuple[list[float], float, float]:
    pred = np.zeros_like(gt)
    clicks, ious, prev, t0 = [], [], None, time.perf_counter()
    for _ in range(max_clicks):
        clicks.append(next_click(gt, pred))
        prev = sam.segment(key, image, clicks, focus=focus, prev=prev)
        pred = prev["mask"]
        ious.append(iou(pred, gt))
        if ious[-1] >= max(THRESHOLDS):  # every threshold reached: later clicks would not count
            ious += [ious[-1]] * (max_clicks - len(ious))
            break
    ms_per_click = (time.perf_counter() - t0) * 1000 / len(clicks)
    ys, xs = np.nonzero(gt)
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1 - xs.min()), float(ys.max() + 1 - ys.min())]
    box_iou = iou(sam.segment(key, image, [], box=box, focus=focus)["mask"], gt)
    return ious, box_iou, ms_per_click


def summarize(rs: list[dict], max_clicks: int) -> dict:
    if not rs:
        return {}
    curve = np.mean([r["ious"] for r in rs], axis=0)
    out = {"n": len(rs)}
    for t in THRESHOLDS:
        k = int(round(t * 100))
        out[f"NoC@{k}"] = round(float(np.mean([r[f"noc{k}"] for r in rs])), 2)
        out[f"fail@{k}"] = round(float(np.mean([r["ious"][-1] < t for r in rs])), 3)
    out.update({"mIoU@1": round(float(curve[0]), 4), "mIoU@3": round(float(curve[2]), 4),
                "mIoU@5": round(float(curve[4]), 4),
                "box_prompt_mIoU": round(float(np.mean([r["box_iou"] for r in rs])), 4),
                "ms_per_click": round(float(np.median([r["ms_per_click"] for r in rs])), 1),
                "curve": [round(float(v), 4) for v in curve]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/minneapple"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--modes", nargs="+", default=["plain", "focus"], choices=["plain", "focus"])
    ap.add_argument("--max-clicks", type=int, default=20)
    ap.add_argument("--max-objects", type=int, default=400, help="random sample of apples (0 = all)")
    ap.add_argument("--include-truncated", action="store_true", help="also apples cut by the crop border")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from app.ml.sam import SamService

    sam = SamService(args.model)
    coco = load_coco(args.data / f"coco_crops_{args.split}.json")
    img_dir = args.data / f"crops_{args.split}" / "images"
    mask_dir = args.data / f"crops_{args.split}" / "masks"
    by_img = anns_by_image(coco)
    objects = [(im, a) for im in coco["images"] for a in by_img[im["id"]]
               if args.include_truncated or not a.get("attributes", {}).get("truncated")]
    random.Random(args.seed).shuffle(objects)
    if args.max_objects:
        objects = objects[: args.max_objects]
    objects.sort(key=lambda o: o[0]["id"])  # group by crop so each crop is encoded once
    print(f"{len(objects)} apples on {len({o[0]['id'] for o in objects})} crops, modes {args.modes}")

    rows, cur_id, image, inst = [], None, None, None
    t_start = time.time()
    for n, (im, a) in enumerate(objects):
        if im["id"] != cur_id:
            image = load_rgb(img_dir / im["file_name"])
            inst = load_mask(mask_dir / im["mask_file"])
            cur_id = im["id"]
        gt = inst == a["attributes"]["instance_id"]
        if not gt.any():
            continue
        for mode in args.modes:
            ious, box_iou, ms = run_object(sam, f"crop{im['id']}", image, gt, mode == "focus", args.max_clicks)
            row = {"mode": mode, "image": im["file_name"], "instance_id": a["attributes"]["instance_id"],
                   "area": int(gt.sum()), "size": size_bucket(gt.sum()),
                   "iou1": round(ious[0], 4), "iou3": round(ious[2], 4), "iou5": round(ious[4], 4),
                   "box_iou": round(box_iou, 4), "ms_per_click": round(ms, 1), "ious": ious}
            for t in THRESHOLDS:
                k = int(round(t * 100))
                row[f"noc{k}"] = next((i + 1 for i, v in enumerate(ious) if v >= t), args.max_clicks)
            rows.append(row)
        if (n + 1) % 25 == 0:
            print(f"{n + 1}/{len(objects)} apples  {time.time() - t_start:.0f} s")

    out = results_dir()
    with open(out / "sam_noc_objects.csv", "w", newline="") as f:
        cols = [c for c in rows[0] if c != "ious"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = {"model": args.model, "device": sam.device, "max_clicks": args.max_clicks, "modes": {}}
    for mode in args.modes:
        rs = [r for r in rows if r["mode"] == mode]
        summary["modes"][mode] = {"all": summarize(rs, args.max_clicks)}
        for b in ("small", "medium", "large"):
            summary["modes"][mode][b] = summarize([r for r in rs if r["size"] == b], args.max_clicks)
    (out / "sam_noc_summary.json").write_text(json.dumps(summary, indent=1))

    print("\n| mode | apples | NoC@80 | NoC@85 | NoC@90 | mIoU@1 | mIoU@3 | mIoU@5 | box prompt IoU | ms / click |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for mode in args.modes:
        s = summary["modes"][mode]["all"]
        print(f"| {mode} | {s['n']} | {s['NoC@80']} | {s['NoC@85']} | {s['NoC@90']} | {s['mIoU@1']} | "
              f"{s['mIoU@3']} | {s['mIoU@5']} | {s['box_prompt_mIoU']} | {s['ms_per_click']} |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4))
    styles = {"plain": "--", "focus": "-"}
    for mode in args.modes:
        for size, marker in (("all", "o"), ("small", ".")):
            s = summary["modes"][mode].get(size)
            if s:
                ax.plot(range(1, args.max_clicks + 1), s["curve"], styles[mode], marker=marker, markersize=4,
                        label=f"{mode}, {size} (n={s['n']})")
    for t in THRESHOLDS:
        ax.axhline(t, lw=0.6, color="grey", ls=":")
    ax.set_xlabel("number of clicks")
    ax.set_ylabel("mean IoU")
    ax.set_ylim(0.4, 1)
    ax.set_xlim(1, args.max_clicks)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sam_miou_curve.png", dpi=160)


if __name__ == "__main__":
    main()
