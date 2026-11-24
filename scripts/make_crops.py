"""Step 1.6: cut labeling-sized crops with their own ground truth.

Full MinneApple images hold ~40 apples, too many for one labeling session. This
cuts a grid of square crops, crops the instance MASK (not the polygons) and
re-extracts polygons, so objects cut by the border get clean outlines.

Usage:
    python scripts/make_crops.py --images <.../train/images> --masks <.../train/masks> \
                                 --data data/minneapple --split study
    python scripts/make_crops.py ... --split pool

Writes:
    data/minneapple/crops_<split>/images/*.png      crops to upload into the app
    data/minneapple/crops_<split>/masks/*.png       matching instance masks (16 bit) for evaluation
    data/minneapple/coco_crops_<split>.json         ground truth for the crops
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mnapple_common import ExpImage, load_instance_mask, mask_to_annotations, write_coco


def grid(length: int, size: int) -> list[int]:
    """Start offsets covering [0, length); the last crop is aligned to the edge."""
    if length <= size:
        return [0]
    starts = list(range(0, length - size + 1, size))
    if starts[-1] + size < length:
        starts.append(length - size)
    return starts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--masks", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/minneapple"))
    ap.add_argument("--split", required=True, choices=["pool", "val", "test", "study"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--min-apples", type=int, default=8)
    ap.add_argument("--max-apples", type=int, default=15)
    ap.add_argument("--min-area", type=int, default=20)
    args = ap.parse_args()

    splits = json.loads((args.data / "splits.json").read_text())
    out_dir = args.data / f"crops_{args.split}"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    exp_images, exp_anns, extra = [], [], {}
    seen = kept = 0
    for name in splits[args.split]:
        img = np.array(Image.open(args.images / name).convert("RGB"))
        mask = load_instance_mask(args.masks / name)
        ids, counts = np.unique(mask, return_counts=True)
        full_areas = {int(i): int(c) for i, c in zip(ids, counts) if i != 0}
        h, w = mask.shape
        for y in grid(h, args.size):
            for x in grid(w, args.size):
                seen += 1
                cm = mask[y:y + args.size, x:x + args.size]
                cid = len(exp_images) + 1
                anns, _ = mask_to_annotations(cm, cid, args.min_area, full_areas)
                if not (args.min_apples <= len(anns) <= args.max_apples):
                    continue
                kept += 1
                stem = f"{Path(name).stem}_x{x}_y{y}"
                Image.fromarray(img[y:y + args.size, x:x + args.size]).save(out_dir / "images" / f"{stem}.png")
                Image.fromarray(cm.astype(np.uint16)).save(out_dir / "masks" / f"{stem}.png")
                ch, cw = cm.shape
                exp_images.append(ExpImage(id=cid, file_name=f"{stem}.png", width=cw, height=ch))
                extra[cid] = {"source_image": name, "offset": [x, y], "mask_file": f"{stem}.png"}
                exp_anns += anns

    write_coco(args.data / f"coco_crops_{args.split}.json", exp_images, exp_anns, extra,
               f"MinneApple {args.split} crops {args.size}px (ground truth)")
    n_trunc = sum(1 for a in exp_anns if getattr(a, "truncated", False))
    print(f"{args.split}: kept {kept} of {seen} crops, {len(exp_anns)} apples "
          f"({len(exp_anns) / max(kept, 1):.1f} per crop, {n_trunc} cut by a crop border)")


if __name__ == "__main__":
    main()
