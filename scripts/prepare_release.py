"""Prepare (and optionally upload) the public model and dataset for the submission.

Builds two folders under data/release/ that can be uploaded to Hugging Face as they are:

  data/release/model/    detector_v<k>.pt, its metadata, the training rounds, a model card
                         (README.md) and a stand-alone inference example
  data/release/dataset/  the COCO export from the app (images + annotations) and a dataset card

    # 1. build the folders (the project id is in the app's URL, e.g. #/projects/4)
    docker compose run --rm backend python scripts/prepare_release.py --project 4 --export data/export.zip

    # 2a. upload in the browser: create the repos on huggingface.co and drag the folder contents in
    # 2b. or upload from here, after logging in yourself (the token stays in the Docker volume):
    docker compose run --rm backend huggingface-cli login
    docker compose run --rm backend python scripts/prepare_release.py --project 4 --export data/export.zip \\
        --push-model <user>/orchardapple-detector --push-dataset <user>/orchardapple-instances-v1

The cards are templates with the numbers this script can read itself (training rounds, image and
object counts) plus the evaluation numbers from the report.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the backend package lives in backend/app in the repo and in /app/app inside the container
for p in (ROOT / "backend", ROOT):
    sys.path.insert(0, str(p))

LICENSE = "cc-by-nc-sa-3.0"  # MinneApple's licence; check it against the copy in your download
CITATION = """@article{haeni2020minneapple,
  title   = {MinneApple: A Benchmark Dataset for Apple Detection and Segmentation},
  author  = {H{\\"a}ni, Nicolai and Roy, Pravakar and Isler, Volkan},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {5}, number = {2}, pages = {852--858}, year = {2020},
  doi     = {10.1109/LRA.2020.2965061}
}"""

INFERENCE = '''"""Run the LabelLoop apple detector on one image (needs torch, torchvision, pillow)."""
import sys

import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import to_tensor

model = fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, box_detections_per_img=300)
model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
model.load_state_dict(torch.load("{weights}", map_location="cpu"))
model.eval()

img = Image.open(sys.argv[1]).convert("RGB")
with torch.no_grad():
    out = model([to_tensor(img)])[0]
keep = out["scores"] >= 0.5
for (x1, y1, x2, y2), s in zip(out["boxes"][keep].tolist(), out["scores"][keep].tolist()):
    print(f"apple {{s:.2f}}  x={{x1:.0f}} y={{y1:.0f}} w={{x2 - x1:.0f}} h={{y2 - y1:.0f}}")
'''


def model_card(meta: dict, rounds: list[dict], weights: str, repo: str | None) -> str:
    rows = "\n".join(
        f"| {r['round']} | {r['trigger']} | {r['n_finished']} | {r['n_train']} / {r.get('n_holdout', 0)} | "
        f"{r.get('ap50_previous', float('nan')):.2f} | {r.get('ap50_new', float('nan')):.2f} | "
        f"{'deployed' if r['accepted'] else 'kept old model'} |"
        for r in rounds)
    return f"""---
license: {LICENSE}
library_name: pytorch
pipeline_tag: object-detection
tags: [object-detection, faster-rcnn, torchvision, agriculture, apples, active-learning]
datasets: [{repo or '<user>/orchardapple-instances-v1'}]
---

# OrchardApple detector (Faster R-CNN ResNet-50 FPN v2)

This is the detector that suggests apples inside LabelLoop, the annotation tool I built for the AI
Vision course. It starts from torchvision's COCO weights and is fine-tuned inside the app, in rounds,
on the images the annotator has finished. Each new model is only deployed if it scores at least as
well as the current one on held-out finished images.

- Architecture: `fasterrcnn_resnet50_fpn_v2`, new 2-class box head (background, apple)
- Initialisation: COCO weights from torchvision (`FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1`)
- Training: my own loop (SGD, lr 0.005, momentum 0.9, weight decay 1e-4, step decay at 70%,
  horizontal flip and brightness/contrast jitter, batch 2, 3 trainable backbone stages)
- Data: 512 px crops of MinneApple orchard images, labelled in LabelLoop
- File: `{weights}` (version {meta.get('version', '?')}, {meta.get('epochs', '?')} epochs per round)

## Training rounds inside the app

AP50 of the old and the new model on the same held-out images of that round.

| Round | Trigger | Finished images | Train / held-out | AP50 old | AP50 new | Result |
|---|---|---|---|---|---|---|
{rows}

## Evaluation on the MinneApple test crops

The same training recipe, run offline on 532 held-out MinneApple test crops (pycocotools):

| Training crops | AP50 | mAP@[.5:.95] | AP_small |
|---|---|---|---|
| 0 (COCO model, "apple" class) | 0.43 | 0.20 | 0.20 |
| 25 | 0.68 | 0.31 | 0.30 |
| 100 | 0.76 | 0.37 | 0.36 |
| 200 | 0.77 | 0.38 | 0.37 |

This file is the model the app trained on my own labels and deployed after its guard checks (see the
rounds above). Its AP50 on the held-out images of the last rounds was 0.88 (14 images).

## How to use

```bash
python inference.py my_orchard_photo.jpg
```

## Limitations

- Trained on MinneApple only: mostly red apples, one orchard region, daylight. Yellow-green apples
  are missed far more often (see the failure analysis in the report).
- MinneApple labels only the front tree row and no fallen apples, so the model learned the same
  policy. Apples in background rows count as false positives in the evaluation.
- Small and heavily occluded apples remain the main source of misses.

## Citation

Dataset:

```bibtex
{CITATION}
```

Code: LabelLoop, <https://github.com/<user>/<repo>>
"""


def dataset_card(coco: dict, model_repo: str | None) -> str:
    n_img, n_ann = len(coco.get("images", [])), len(coco.get("annotations", []))
    return f"""---
license: {LICENSE}
task_categories: [object-detection, image-segmentation]
tags: [agriculture, apples, coco, instance-segmentation]
pretty_name: OrchardApple-Instances-v1
size_categories: [n<1K]
---

# OrchardApple-Instances-v1

{n_img} orchard image crops (512 x 512) with **{n_ann} apple instances** as polygons and boxes,
in COCO format. I labelled them with LabelLoop in assisted mode: the detector
({model_repo or '<user>/orchardapple-detector'}) proposed apples, SAM 2.1 turned clicks into
outlines, and I checked and corrected every object before marking the image done.

## Files

- `annotations.json`: COCO instances (category 1 = apple; `segmentation` polygons, `bbox`, `area`)
- `images/`: the crops, file names as in `annotations.json`

## Source and labelling policy

The images are crops of the MinneApple detection training set (Häni, Roy and Isler, 2020).
Following MinneApple, only apples on the front tree row are labelled; fallen apples on the ground and
apples in background rows are not.

## Quality check against the original MinneApple masks

One-to-one (Hungarian) matching on mask IoU, a match being IoU >= 0.5:

| Metric | Value |
|---|---|
| Recall vs. MinneApple | 0.70 |
| Precision vs. MinneApple | 0.61 |
| Mean mask IoU of matched apples | 0.73 |
| Labeling time | 61 s per crop, 0.90 clicks per apple |

Most disagreement comes from where the "back row" starts, from small apples whose two outlines overlap
less than 50%, and from real misses. The crops were labeled in fixed file-name order; crops skipped
before labeling, and why: TODO.

## Licence and citation

The images come from MinneApple and keep its licence ({LICENSE}). Please cite the original dataset:

```bibtex
{CITATION}
```
"""


def push(folder: Path, repo: str, kind: str):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type=kind, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo, repo_type=kind,
                      commit_message="LabelLoop release")
    print(f"uploaded {folder} -> https://huggingface.co/{'datasets/' if kind == 'dataset' else ''}{repo}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", type=int, help="project whose latest detector version is released")
    ap.add_argument("--version", type=int, help="detector version (default: latest)")
    ap.add_argument("--export", type=Path, help="export .zip from the app (COCO with images)")
    ap.add_argument("--out", type=Path, default=Path("data/release"))
    ap.add_argument("--push-model", metavar="USER/REPO")
    ap.add_argument("--push-dataset", metavar="USER/REPO")
    args = ap.parse_args()

    if args.project is not None:
        from app.config import MODEL_DIR
        src = Path(MODEL_DIR) / f"project_{args.project}"
        versions = sorted(int(p.stem.split("_v")[1]) for p in src.glob("detector_v*.pt"))
        if not versions:
            sys.exit(f"no trained detector in {src}; train at least one round in that project first")
        v = args.version or versions[-1]
        out = args.out / "model"
        out.mkdir(parents=True, exist_ok=True)
        weights = f"orchardapple_frcnn_v{v}.pt"
        shutil.copy2(src / f"detector_v{v}.pt", out / weights)
        meta_f = src / f"detector_v{v}.json"
        meta = json.loads(meta_f.read_text()) if meta_f.exists() else {"version": v}
        (out / "training_meta.json").write_text(json.dumps(meta, indent=1))
        rounds_f = src / "rounds.json"
        rounds = json.loads(rounds_f.read_text()) if rounds_f.exists() else []
        (out / "training_rounds.json").write_text(json.dumps(rounds, indent=1))
        (out / "inference.py").write_text(INFERENCE.format(weights=weights))
        (out / "README.md").write_text(model_card(meta, rounds, weights, args.push_dataset))
        print(f"model v{v} -> {out}  ({(out / weights).stat().st_size / 1e6:.0f} MB)")
        if args.push_model:
            push(out, args.push_model, "model")

    if args.export is not None:
        out = args.out / "dataset"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        with zipfile.ZipFile(args.export) as z:
            z.extractall(out)
        ann = next(iter(sorted(out.rglob("*.json"))), None)
        if ann is None:
            sys.exit("no COCO .json in the export; export as COCO with images")
        coco = json.loads(ann.read_text())
        (out / "README.md").write_text(dataset_card(coco, args.push_model))
        n_files = sum(1 for p in out.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        print(f"dataset: {len(coco['images'])} images ({n_files} image files), "
              f"{len(coco['annotations'])} objects -> {out}")
        if n_files < len(coco["images"]):
            print("WARNING: fewer image files than images in the json; export again with images included")
        if args.push_dataset:
            push(out, args.push_dataset, "dataset")


if __name__ == "__main__":
    main()
