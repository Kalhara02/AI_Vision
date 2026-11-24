"""Step 1.7: draw ground truth on random images into one grid PNG, to check it by eye.

Usage:
    python scripts/show_gt.py --coco data/minneapple/coco_crops_study.json \
                              --images data/minneapple/crops_study/images --out gt_check.png
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("gt_check.png"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coco = json.loads(args.coco.read_text())
    by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        by_image.setdefault(a["image_id"], []).append(a)
    images = random.Random(args.seed).sample(coco["images"], min(args.n, len(coco["images"])))

    rows = (len(images) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (args.cols * args.tile, rows * args.tile), "white")
    for k, im in enumerate(images):
        img = Image.open(args.images / im["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for a in by_image.get(im["id"], []):
            color = (255, 160, 0) if a.get("attributes", {}).get("truncated") else (0, 255, 255)
            for seg in a["segmentation"]:
                pts = list(zip(seg[0::2], seg[1::2]))
                draw.polygon(pts, outline=color, width=2)
            x, y, w, h = a["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 255), width=1)
        draw.text((6, 6), f"{im['file_name']}  n={len(by_image.get(im['id'], []))}", fill=(255, 255, 255))
        img.thumbnail((args.tile, args.tile))
        sheet.paste(img, ((k % args.cols) * args.tile, (k // args.cols) * args.tile))
    sheet.save(args.out)
    print(f"wrote {args.out}  (cyan = apple polygon, orange = cut by crop border, magenta = bbox)")


if __name__ == "__main__":
    main()
