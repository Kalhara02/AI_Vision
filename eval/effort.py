"""Effort and quality: manual vs. assisted, from the app's own interaction logs.

1. Download the logs of your study project:  http://localhost:8000/api/projects/<id>/logs  -> data/logs.json
2. Export that project as COCO (without images is fine) -> data/study_export.json
3. docker compose run --rm backend python eval/effort.py --logs data/logs.json \
       --export data/study_export.json --gt data/minneapple/coco_crops_study.json

Per finished image: active time (gaps longer than --idle seconds are not counted),
pointer clicks, keyboard shortcuts, objects. Quality against MinneApple ground
truth: one-to-one matching (Hungarian) on mask IoU, TP at IoU >= 0.5.
Also: detector suggestion acceptance rate per model version.

Writes: eval/results/effort_images.csv, effort_summary.md, effort_time.png, acceptance.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import anns_by_image, load_coco, load_mask, mean_sd, results_dir
from app.analytics import acceptance, per_image_effort  # same code as the in-app Insights panel


# ---------------------------------------------------------------- quality vs ground truth
def quality(export: dict, gt: dict, gt_mask_dir: Path) -> dict[int, dict]:
    from scipy.optimize import linear_sum_assignment

    from app.ml.masks import mask_iou, polygon_to_mask

    gt_by_name = {im["file_name"]: im for im in gt["images"]}
    gt_anns = anns_by_image(gt)
    ex_anns = anns_by_image(export)
    out = {}
    for im in export["images"]:
        original = im["file_name"].split("_", 1)[1] if "_" in im["file_name"] else im["file_name"]
        g = gt_by_name.get(original)
        if g is None:
            continue
        inst = load_mask(gt_mask_dir / g["mask_file"])
        h, w = inst.shape
        gts = [inst == a["attributes"]["instance_id"] for a in gt_anns[g["id"]]]
        preds = []
        for a in ex_anns.get(im["id"], []):
            if a["segmentation"]:
                s = a["segmentation"][0]
                preds.append(polygon_to_mask(list(zip(s[0::2], s[1::2])), h, w))
            else:
                x, y, bw, bh = a["bbox"]
                preds.append(polygon_to_mask([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]], h, w))
        if not gts or not preds:
            out[im["id"]] = {"tp": 0, "precision": 0.0 if preds else None, "recall": 0.0, "mean_iou_tp": None}
            continue
        m = np.array([[mask_iou(p, q) for q in gts] for p in preds])
        r, c = linear_sum_assignment(-m)
        tp_ious = [m[i, j] for i, j in zip(r, c) if m[i, j] >= 0.5]
        out[im["id"]] = {"tp": len(tp_ious), "precision": round(len(tp_ious) / len(preds), 4),
                         "recall": round(len(tp_ious) / len(gts), 4),
                         "mean_iou_tp": round(float(np.mean(tp_ious)), 4) if tp_ious else None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--export", type=Path, help="COCO export of the study project")
    ap.add_argument("--gt", type=Path, help="ground truth COCO of the crops (coco_crops_study.json)")
    ap.add_argument("--gt-masks", type=Path, help="default: <gt folder>/crops_study/masks")
    ap.add_argument("--idle", type=float, default=60.0, help="gaps longer than this (s) are not active time")
    args = ap.parse_args()

    out = results_dir()
    events = json.loads(args.logs.read_text())
    rows = per_image_effort(events, args.idle)
    if not rows:
        raise SystemExit("no finished images (image_done events) in the logs")

    if args.export and args.gt:
        mask_dir = args.gt_masks or args.gt.parent / "crops_study" / "masks"
        q = quality(load_coco(args.export), load_coco(args.gt), mask_dir)
        for r in rows:
            r.update(q.get(r["image_id"], {}))

    with open(out / "effort_images.csv", "w", newline="") as f:
        cols = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    acc = acceptance(events)
    if acc:
        with open(out / "acceptance.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(acc[0]))
            w.writeheader()
            w.writerows(acc)

    # ---- summary per mode
    lines = ["| mode | images | active s / image | clicks / object | recall | precision | IoU (TP) |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)
    for mode, rs in sorted(by_mode.items()):
        cpo = [r["clicks"] / r["objects"] for r in rs if r["objects"]]
        lines.append(f"| {mode} | {len(rs)} | {mean_sd(r['active_s'] for r in rs)} | {mean_sd(cpo)} | "
                     f"{mean_sd(r['recall'] for r in rs if r.get('recall') is not None)} | "
                     f"{mean_sd(r['precision'] for r in rs if r.get('precision') is not None)} | "
                     f"{mean_sd(r['mean_iou_tp'] for r in rs if r.get('mean_iou_tp') is not None)} |")

    # paired test over participants who did both modes
    per_p = defaultdict(dict)
    for (p, m), vals in _group(rows, ("participant", "mode"), "active_s").items():
        per_p[p][m] = float(np.mean(vals))
    pairs = [(d["manual"], d["assisted"]) for d in per_p.values() if "manual" in d and "assisted" in d]
    if pairs:
        man, ast = np.array(pairs).T
        saved = 100 * (1 - ast.mean() / man.mean())
        lines += ["", f"Participants with both modes: {len(pairs)}. "
                      f"Mean active time per image: manual {man.mean():.1f} s, assisted {ast.mean():.1f} s "
                      f"(**{saved:.0f}% less time** with assistance)."]
        if len(pairs) >= 3:
            from scipy.stats import wilcoxon
            p = wilcoxon(man, ast).pvalue
            lines.append(f"Paired Wilcoxon signed-rank test on per-participant means: p = {p:.3f} "
                         f"(n = {len(pairs)}, a pilot study, not proof).")
    if acc:
        lines += ["", "| detector version | suggested | accepted | rejected | acceptance rate |", "| --- | --- | --- | --- | --- |"]
        lines += [f"| v{a['version']} | {a['shown']} | {a['accepted']} | {a['rejected']} | {a['acceptance_rate']} |" for a in acc]
    (out / "effort_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = sorted(by_mode)
    means = [np.mean([r["active_s"] for r in by_mode[m]]) for m in modes]
    sds = [np.std([r["active_s"] for r in by_mode[m]], ddof=1) if len(by_mode[m]) > 1 else 0 for m in modes]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(modes, means, yerr=sds, capsize=6, color=["#999999" if m == "manual" else "#0072B2" for m in modes])
    ax.set_ylabel("active seconds per crop (mean ± sd)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "effort_time.png", dpi=160)


def _group(rows, keys, value):
    g = defaultdict(list)
    for r in rows:
        g[tuple(r[k] for k in keys)].append(r[value])
    return g


if __name__ == "__main__":
    main()
