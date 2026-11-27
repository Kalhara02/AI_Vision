"""Build a small multi-class benchmark from COCO val2017 to test LabelLoop outside orchards.

COCO (Lin et al. 2014) is used here only as a second, very different image domain (street scenes,
homes, animals) with several classes and instance masks, so the same evaluation as for MinneApple
can be run on it. Download once, by hand, into data/raw/coco/:
    http://images.cocodataset.org/zips/val2017.zip                     (~0.8 GB)
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip  (only instances_val2017.json is used)

    docker compose run --rm backend python scripts/make_coco_subset.py
    docker compose run --rm backend python scripts/make_coco_subset.py --classes person bicycle car dog bottle chair

Writes data/coco_sub/{pool,test}.json + images/: images that contain at least one of the classes,
split by image (seed fixed), annotations restricted to those classes (crowd regions kept as COCO
"ignore" areas for evaluation, never used for training).
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco", type=Path, default=Path("data/raw/coco"))
    ap.add_argument("--out", type=Path, default=Path("data/coco_sub"))
    ap.add_argument("--classes", nargs="+", default=["person", "bicycle", "car", "dog", "bottle", "chair"])
    ap.add_argument("--pool", type=int, default=150, help="images available for training")
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ann_file = next(iter(args.coco.rglob("instances_val2017.json")), None)
    img_dir = next((p for p in args.coco.rglob("val2017") if p.is_dir()), None)
    if ann_file is None or img_dir is None:
        raise SystemExit(f"put val2017/ and annotations/instances_val2017.json under {args.coco}")
    coco = json.loads(ann_file.read_text())
    cats = [c for c in coco["categories"] if c["name"] in args.classes]
    missing = set(args.classes) - {c["name"] for c in cats}
    if missing:
        raise SystemExit(f"not COCO classes: {sorted(missing)}")
    keep = {c["id"] for c in cats}
    anns = [a for a in coco["annotations"] if a["category_id"] in keep]
    with_obj = {a["image_id"] for a in anns if not a.get("iscrowd")}
    images = sorted((im for im in coco["images"] if im["id"] in with_obj), key=lambda im: im["id"])
    random.Random(args.seed).shuffle(images)
    need = args.pool + args.test
    if len(images) < need:
        raise SystemExit(f"only {len(images)} images contain these classes, need {need}")
    split = {"test": images[: args.test], "pool": images[args.test: need]}

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    for name, ims in split.items():
        ids = {im["id"] for im in ims}
        sub = {"info": {"description": f"COCO val2017 subset ({name}) for LabelLoop", "classes": args.classes},
               "licenses": coco.get("licenses", []), "categories": cats,
               "images": [{k: im[k] for k in ("id", "file_name", "width", "height")} for im in ims],
               "annotations": [a for a in anns if a["image_id"] in ids]}
        (args.out / f"{name}.json").write_text(json.dumps(sub))
        for im in ims:
            dst = args.out / "images" / im["file_name"]
            if not dst.exists():
                shutil.copy2(img_dir / im["file_name"], dst)
        n_obj = sum(1 for a in sub["annotations"] if not a.get("iscrowd"))
        per = {c["name"]: sum(1 for a in sub["annotations"] if a["category_id"] == c["id"] and not a.get("iscrowd"))
               for c in cats}
        print(f"{name}: {len(ims)} images, {n_obj} objects {per}")


if __name__ == "__main__":
    main()
