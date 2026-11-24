"""Step 1.4 + 1.5: fixed split of the MinneApple training images and COCO ground truth per split.

Usage:
    python scripts/minneapple_to_coco.py --images <.../detection/train/images> \
                                         --masks  <.../detection/train/masks> \
                                         --out data/minneapple

Writes:
    data/minneapple/splits.json             which image goes to which split (seed 42, never changes)
    data/minneapple/coco_<split>.json       ground truth polygons per split
    data/minneapple/stats.json              counts for the report (dropped tiny instances etc.)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

from mnapple_common import IMAGE_EXTS, ExpImage, load_instance_mask, mask_to_annotations, write_coco

SPLITS = {"val": 70, "test": 100, "study": 100}  # everything else goes to "pool"


def make_split(names: list[str], seed: int) -> dict[str, list[str]]:
    names = sorted(names)
    rng = random.Random(seed)
    rng.shuffle(names)
    if len(names) < sum(SPLITS.values()) + 10:
        raise SystemExit(f"only {len(names)} images, need more than {sum(SPLITS.values()) + 10}")
    out, i = {}, 0
    for split, n in SPLITS.items():
        out[split] = sorted(names[i:i + n])
        i += n
    out["pool"] = sorted(names[i:])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/minneapple"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-area", type=int, default=20, help="drop instances smaller than this (pixels)")
    args = ap.parse_args()

    images = {p.name: p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    masks = {p.name: p for p in args.masks.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    paired = sorted(set(images) & set(masks))
    if missing := sorted(set(images) ^ set(masks)):
        print(f"warning: {len(missing)} files without a partner, e.g. {missing[:3]}")
    print(f"{len(paired)} image/mask pairs")

    args.out.mkdir(parents=True, exist_ok=True)
    splits_file = args.out / "splits.json"
    if splits_file.exists():
        splits = json.loads(splits_file.read_text())
        print(f"reusing existing {splits_file} (delete it to re-split)")
    else:
        splits = make_split(paired, args.seed)
        splits_file.write_text(json.dumps(splits, indent=1))

    id_of = {name: i + 1 for i, name in enumerate(paired)}  # stable image ids across all splits
    stats = {}
    for split, names in splits.items():
        exp_images, exp_anns, extra = [], [], {}
        totals = {"images": 0, "instances": 0, "dropped_small": 0, "dropped_degenerate": 0, "multi_part": 0}
        for name in names:
            with Image.open(images[name]) as im:
                w, h = im.size
            mask = load_instance_mask(masks[name])
            if mask.shape != (h, w):
                raise SystemExit(f"{name}: mask {mask.shape} does not match image {(h, w)}")
            iid = id_of[name]
            anns, st = mask_to_annotations(mask, iid, args.min_area)
            exp_images.append(ExpImage(id=iid, file_name=name, width=w, height=h))
            extra[iid] = {"mask_file": name, "split": split}
            exp_anns += anns
            totals["images"] += 1
            totals["instances"] += len(anns)
            for k, v in st.items():
                totals[k] += v
        write_coco(args.out / f"coco_{split}.json", exp_images, exp_anns, extra, f"MinneApple {split} (ground truth)")
        totals["instances_per_image"] = round(totals["instances"] / max(totals["images"], 1), 1)
        stats[split] = totals
        print(f"{split:6s} {totals}")

    (args.out / "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
