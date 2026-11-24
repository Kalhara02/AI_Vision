"""Interactive segmentation quality of SAM 2.1 on apples (NoC protocol).

Standard click simulation (as in RITM, Sofiiuk et al. 2022): the first click goes
to the point inside the object farthest from its border; every next click goes
to the centre of the largest error region, positive for missed object, negative
for wrongly included background. The previous mask logits are fed back in.

Reports NoC@85 / NoC@90 (clicks to reach the IoU, max = penalty when never
reached), mIoU after k clicks, and IoU from a box prompt (detector -> SAM path).

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


def farthest_point(region: np.ndarray) -> tuple[int, int, float]:
    """(x, y, distance) of the pixel deepest inside `region`."""
    d = cv2.distanceTransform(np.pad(region, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(d)), d.shape)
    return int(x), int(y), float(d[y, x])


def next_click(gt: np.ndarray, pred: np.ndarray) -> tuple[int, int, int]:
    fn = gt & ~pred
    fp = pred & ~gt
    xn, yn, dn = farthest_point(fn)
    xp, yp, dp = farthest_point(fp)
    return (xn, yn, 1) if dn >= dp else (xp, yp, 0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/minneapple"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="facebook/sam2.1-hiera-tiny")
    ap.add_argument("--max-clicks", type=int, default=20)
    ap.add_argument("--max-objects", type=int, default=400, help="random sample of apples (0 = all)")
    ap.add_argument("--include-truncated", action="store_true", help="also apples cut by the crop border")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2ImagePredictor.from_pretrained(args.model, device=dev)

    coco = load_coco(args.data / f"coco_crops_{args.split}.json")
    img_dir = args.data / f"crops_{args.split}" / "images"
    mask_dir = args.data / f"crops_{args.split}" / "masks"
    by_img = anns_by_image(coco)
    objects = [(im, a) for im in coco["images"] for a in by_img[im["id"]]
               if args.include_truncated or not a.get("attributes", {}).get("truncated")]
    random.Random(args.seed).shuffle(objects)
    if args.max_objects:
        objects = objects[: args.max_objects]
    objects.sort(key=lambda o: o[0]["id"])  # group by image so each image is encoded once
    print(f"{len(objects)} apples on {len({o[0]['id'] for o in objects})} crops, device {dev}")

    rows, cur_id, inst_mask = [], None, None
    t_start = time.time()
    for n, (im, a) in enumerate(objects):
        if im["id"] != cur_id:
            with torch.inference_mode():
                predictor.set_image(load_rgb(img_dir / im["file_name"]))
            inst_mask = load_mask(mask_dir / im["mask_file"])
            cur_id = im["id"]
        gt = inst_mask == a["attributes"]["instance_id"]
        if gt.sum() == 0:
            continue

        pred = np.zeros_like(gt)
        pts, lbl, logits, ious = [], [], None, []
        for k in range(args.max_clicks):
            x, y, pos = next_click(gt, pred)
            pts.append([x, y])
            lbl.append(pos)
            with torch.inference_mode():
                masks, scores, low = predictor.predict(
                    point_coords=np.array(pts, np.float32), point_labels=np.array(lbl, np.int32),
                    mask_input=logits, multimask_output=(k == 0))
            best = int(np.argmax(np.asarray(scores).reshape(-1)))
            masks = np.asarray(masks).reshape(-1, *gt.shape)
            pred = masks[best] > 0.5
            logits = np.asarray(low).reshape(-1, *np.asarray(low).shape[-2:])[best][None]
            ious.append(iou(pred, gt))
            if ious[-1] >= 0.90:  # both thresholds reached: later clicks would not count
                ious += [ious[-1]] * (args.max_clicks - len(ious))
                break

        ys, xs = np.nonzero(gt)
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
        with torch.inference_mode():
            bm, _, _ = predictor.predict(box=box, multimask_output=False)
        box_iou = iou(np.asarray(bm).reshape(-1, *gt.shape)[0] > 0.5, gt)

        noc = lambda t: next((i + 1 for i, v in enumerate(ious) if v >= t), args.max_clicks)  # noqa: E731
        rows.append({"image": im["file_name"], "instance_id": a["attributes"]["instance_id"],
                     "area": int(gt.sum()), "size": size_bucket(gt.sum()),
                     "noc85": noc(0.85), "noc90": noc(0.90),
                     "iou1": round(ious[0], 4), "iou3": round(ious[2], 4), "iou5": round(ious[4], 4),
                     "box_iou": round(box_iou, 4), "ious": ious})
        if (n + 1) % 50 == 0:
            print(f"{n + 1}/{len(objects)}  {time.time() - t_start:.0f} s")

    out = results_dir()
    with open(out / "sam_noc_objects.csv", "w", newline="") as f:
        cols = [c for c in rows[0] if c != "ious"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def summarize(rs):
        if not rs:
            return {}
        curve = np.mean([r["ious"] for r in rs], axis=0)
        return {"n": len(rs),
                "NoC@85": round(float(np.mean([r["noc85"] for r in rs])), 2),
                "NoC@90": round(float(np.mean([r["noc90"] for r in rs])), 2),
                "fail@85_within_max": round(float(np.mean([r["noc85"] == args.max_clicks and r["ious"][-1] < 0.85
                                                           for r in rs])), 3),
                "mIoU@1": round(float(curve[0]), 4), "mIoU@3": round(float(curve[2]), 4),
                "mIoU@5": round(float(curve[4]), 4), "box_prompt_mIoU": round(float(np.mean([r["box_iou"] for r in rs])), 4),
                "curve": [round(float(v), 4) for v in curve]}

    summary = {"model": args.model, "device": dev, "max_clicks": args.max_clicks, "all": summarize(rows)}
    for b in ("small", "medium", "large"):
        summary[b] = summarize([r for r in rows if r["size"] == b])
    (out / "sam_noc_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "curve"} if isinstance(v, dict) else v
                      for k, v in summary.items()}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("all", "small", "medium"):
        if summary.get(key):
            ax.plot(range(1, args.max_clicks + 1), summary[key]["curve"], marker=".", label=f"{key} (n={summary[key]['n']})")
    ax.axhline(0.85, ls="--", lw=0.8, color="grey")
    ax.axhline(0.90, ls=":", lw=0.8, color="grey")
    ax.set_xlabel("number of clicks")
    ax.set_ylabel("mean IoU")
    ax.set_ylim(0, 1)
    ax.set_xlim(1, args.max_clicks)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "sam_miou_curve.png", dpi=160)


if __name__ == "__main__":
    main()
