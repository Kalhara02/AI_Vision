"""Does the AI help outside orchards? Zero-shot, the learning loop and SAM on a second domain.

Runs the app's own code (app/ml/zeroshot.py, app/ml/detector.py, app/ml/sam.py) on a COCO-format
dataset with several user-chosen classes (by default the COCO subset from scripts/make_coco_subset.py).
Nothing in it is specific to that dataset: point --data at any COCO export, LabelLoop's own included.

    docker compose run --rm backend python eval/generalize.py                 # everything (~3 h on a CPU)
    docker compose run --rm backend python eval/generalize.py --stages zeroshot sam
    docker compose run --rm backend python eval/generalize.py --sizes 25 50 --epochs 6

Stages (each saved as soon as it finishes, so an interrupted run resumes):
  zeroshot  OWLv2 prompted with the class names, before any labels      -> mAP, AP50, AP50 per class
  trained   Faster R-CNN fine-tuned on n pool images (all classes)      -> same metrics, per n
            "app" = trained model + zero-shot for classes with < 10 training boxes (what the user sees)
  sam       SAM 2.1 plain mode, NoC protocol on test instances          -> NoC@80/85/90, IoU after 1/3/5 clicks,
                                                                          IoU from a box prompt, per size
Writes eval/results/generalize_<tag>.json (tag = data folder name).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from common import anns_by_image, load_coco, load_rgb, results_dir, size_bucket


def coco_eval(gt_path: Path, dets: list[dict], cat_ids: list[int]) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO(str(gt_path))
    out = {"mAP": 0.0, "AP50": 0.0, "AP50_per_class": {}}
    if not dets:
        return out
    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    ev.params.catIds = cat_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    out["mAP"], out["AP50"] = round(float(ev.stats[0]), 4), round(float(ev.stats[1]), 4)
    prec = ev.eval["precision"][0, :, :, 0, 2]  # IoU 0.5, all recalls, per class, all areas, 100 dets
    for k, cid in enumerate(cat_ids):
        p = prec[:, k]
        out["AP50_per_class"][gt.cats[cid]["name"]] = round(float(p[p > -1].mean()) if (p > -1).any() else 0.0, 4)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/coco_sub"))
    ap.add_argument("--stages", nargs="+", default=["zeroshot", "trained", "sam"], choices=["zeroshot", "trained", "sam"])
    ap.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-objects", type=int, default=300, help="SAM: random test instances")
    ap.add_argument("--min-area", type=int, default=100, help="SAM: skip instances smaller than this (px)")
    ap.add_argument("--max-clicks", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from app.ml.detector import MIN_TRAINED_INSTANCES, Sample, TrainConfig, detect, train_detector
    from app.ml.zeroshot import calibrate, zero_shot_service

    out_path = results_dir() / f"generalize_{args.data.name}.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    save = lambda: out_path.write_text(json.dumps(res, indent=1))  # noqa: E731

    test_path, pool = args.data / "test.json", load_coco(args.data / "pool.json")
    test = load_coco(test_path)
    img_dir = args.data / "images"
    cats = test["categories"]
    cat_ids, names = [c["id"] for c in cats], [c["name"] for c in cats]
    res["dataset"] = {"data": str(args.data), "classes": names, "test_images": len(test["images"]),
                      "pool_images": len(pool["images"]),
                      "test_objects": sum(1 for a in test["annotations"] if not a.get("iscrowd"))}
    print(res["dataset"])
    test_imgs = {im["id"]: load_rgb(img_dir / im["file_name"]) for im in test["images"]}

    # zero-shot detections for every class, computed once (also reused as the app's fallback)
    zs_dets: dict[int, list[dict]] = {}
    if {"zeroshot", "trained"} & set(args.stages):
        t0 = time.time()
        for k, (iid, img) in enumerate(test_imgs.items(), 1):
            raw, _ = zero_shot_service.detect(f"test{iid}", img, names)
            zs_dets[iid] = [{"image_id": iid, "category_id": cat_ids[d["query"]], "bbox": d["bbox"],
                             "score": calibrate(d["raw"])} for d in raw]
            if k % 25 == 0:
                print(f"zero-shot {k}/{len(test_imgs)}  {time.time() - t0:.0f} s", flush=True)
        zs_ms = (time.time() - t0) * 1000 / max(1, len(test_imgs))
    if "zeroshot" in args.stages:
        m = coco_eval(test_path, [d for ds in zs_dets.values() for d in ds], cat_ids)
        res["zeroshot"] = {**m, "model": zero_shot_service.model_id, "ms_per_image": round(zs_ms)}
        save()
        print("zero-shot", res["zeroshot"])

    if "trained" in args.stages:
        by_img = anns_by_image(pool)
        samples = []
        for im in pool["images"]:
            anns = [a for a in by_img[im["id"]] if not a.get("iscrowd")]
            if anns:
                samples.append(Sample(img_dir / im["file_name"], [a["bbox"] for a in anns], [a["category_id"] for a in anns]))
        random.Random(args.seed).shuffle(samples)
        res.setdefault("trained", {})
        for n in args.sizes:
            if str(n) in res["trained"] or n > len(samples):
                print(f"skip n={n}")
                continue
            model, meta = train_detector(samples[:n], TrainConfig(epochs=args.epochs, seed=args.seed), class_ids=cat_ids)
            weak = [c for c in cat_ids if meta["boxes_per_class"].get(str(c), 0) < MIN_TRAINED_INSTANCES]
            dets = []
            for iid, img in test_imgs.items():
                for d in detect(model, img, None, 0.05):
                    dets.append({"image_id": iid, "category_id": cat_ids[d["label"] - 1], "bbox": d["bbox"], "score": d["score"]})
            app = [d for d in dets if d["category_id"] not in weak] + \
                  [d for ds in zs_dets.values() for d in ds if d["category_id"] in weak]
            res["trained"][str(n)] = {"trained_only": coco_eval(test_path, dets, cat_ids),
                                      "app": coco_eval(test_path, app, cat_ids),
                                      "weak_classes": [names[cat_ids.index(c)] for c in weak],
                                      "boxes_per_class": {names[cat_ids.index(int(c))]: v
                                                          for c, v in meta["boxes_per_class"].items()},
                                      "epochs": args.epochs, "train_seconds": meta["train_seconds"]}
            save()
            print(f"n={n}", json.dumps(res["trained"][str(n)]))

    if "sam" in args.stages:
        from pycocotools.coco import COCO

        from app.ml.sam import SamService
        from sam_noc import THRESHOLDS, run_object, summarize

        sam, gt = SamService(), COCO(str(test_path))
        objs = [a for a in test["annotations"] if not a.get("iscrowd") and a["area"] >= args.min_area]
        random.Random(args.seed).shuffle(objs)
        objs = sorted(objs[: args.max_objects], key=lambda a: a["image_id"])
        rows, t0 = [], time.time()
        for k, a in enumerate(objs, 1):
            m = gt.annToMask(a).astype(bool)
            if not m.any():
                continue
            ious, box_iou, ms = run_object(sam, f"test{a['image_id']}", test_imgs[a["image_id"]], m, "plain", args.max_clicks)
            row = {"ious": ious, "box_iou": box_iou, "ms_per_click": ms, "size": size_bucket(m.sum()),
                   "class": gt.cats[a["category_id"]]["name"]}
            for t in THRESHOLDS:
                kk = int(round(t * 100))
                row[f"noc{kk}"] = next((i + 1 for i, v in enumerate(ious) if v >= t), args.max_clicks)
            rows.append(row)
            if k % 25 == 0:
                print(f"SAM {k}/{len(objs)}  {time.time() - t0:.0f} s", flush=True)
        res["sam"] = {"mode": "plain", "all": summarize(rows, args.max_clicks),
                      **{s: summarize([r for r in rows if r["size"] == s], args.max_clicks) for s in ("small", "medium", "large")},
                      "per_class_noc85": {c: round(float(np.mean([r["noc85"] for r in rows if r["class"] == c])), 2)
                                          for c in names if any(r["class"] == c for r in rows)}}
        save()
        print("SAM", {k: v for k, v in res["sam"]["all"].items() if k != "curve"})

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
