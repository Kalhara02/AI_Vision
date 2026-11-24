# Data preparation (MinneApple)

**Easiest:** extract the dataset into `data/raw/` and run `make data` (or the five
`docker compose run` lines from the runbook). Everything then runs inside the backend
container and you need no Python on your machine. The host instructions below are optional.

Install once on your machine (outside Docker):

```bash
pip install -r scripts/requirements.txt
```

Download `detection.tar.gz` from the UMN Data Repository
(https://conservancy.umn.edu/handle/11299/206575), extract it, and point the scripts at the
training images and masks (the folder names below are what I expect; check yours):

```bash
IMG=~/datasets/minneapple/detection/train/images
MSK=~/datasets/minneapple/detection/train/masks

# 1. fixed split + COCO ground truth per split (seed 42, writes data/minneapple/splits.json)
python scripts/minneapple_to_coco.py --images $IMG --masks $MSK --out data/minneapple

# 2. labeling-sized crops (512 px, 8-15 apples each) for the app and the user study
python scripts/make_crops.py --images $IMG --masks $MSK --data data/minneapple --split study
python scripts/make_crops.py --images $IMG --masks $MSK --data data/minneapple --split pool

# 3. check it by eye
python scripts/show_gt.py --coco data/minneapple/coco_crops_study.json \
                          --images data/minneapple/crops_study/images --out gt_check.png
```

Notes

- `splits.json` is created once and reused. Delete it only if you really want a new split,
  because every number in the report depends on it.
- Each COCO annotation keeps `attributes.instance_id`, `pixel_area` and `truncated`.
  Evaluation uses the pixel-exact masks in `crops_*/masks/` (16-bit PNG, 0 = background),
  not the simplified polygons.
- Polygons follow pixel centres, so polygon area is a little smaller than the pixel count
  (e.g. 126 vs 149 for a small apple). That's why evaluation uses masks.
- `data/` is git-ignored. Never commit the dataset images.
