# LabelLoop project notes

Running log of decisions and results. Everything here goes into the report.

## Day 1 (2026-11-24): Data preparation

Dataset: MinneApple detection set (Häni, Roy, Isler, IEEE RA-L 2020).
Only the 670 training images have public ground truth masks; the 331 test images do not, so we use the 670.

### Split (seed 42, data/minneapple/splits.json)

| Split | Images | Apples | Apples per image | Used for |
| --- | --- | --- | --- | --- |
| pool | 400 | 16,493 | 41.2 | training rounds in the app, detector learning curve |
| val | 70 | 2,998 | 42.8 | reserved for tuning |
| test | 100 | 4,164 | 41.6 | all reported model numbers |
| study | 100 | 4,520 | 45.2 | user study and the submitted dataset |

Mask to polygon conversion dropped only 7 tiny instances (under 20 px) and 1 degenerate one in total.
13 apples consist of more than one visible part (split by leaves or branches); we keep the largest part.

### Crops (512 x 512 px)

| Crop set | Kept | Apples | Apples per crop | Cut by crop border | Filter |
| --- | --- | --- | --- | --- | --- |
| study | 136 of 600 | 1,572 | 11.6 | 234 | 8 to 15 apples |
| pool | 2,137 of 2,400 | 30,943 | 14.5 | 3,593 | at least 1 apple |
| test | 532 of 600 | 7,501 | 14.1 | 861 | at least 1 apple |

Study crops: 40 for the user study, the remaining 96 for the submitted dataset.

### Observations

- MinneApple only labels apples **on the tree**. Fallen apples on the ground are unlabeled.
  Labeling rule for me and all participants: label apples on the tree, ignore apples on the ground.
  Expected effect: the pretrained COCO detector will also detect ground apples, which count as
  false positives. Fine-tuning should learn this task definition.
- Ground truth check image (data/gt_check.png): polygons align with the apples.

### Environment

- Laptop: <CPU model>, <RAM> GB RAM, no GPU. Docker Desktop memory: <x> GB.
- First full build: about 4 minutes.

## Decisions

- Detector backbone: - Detector backbone: Faster R-CNN ResNet-50 FPN v2. CPU training: about 13 s per step (batch 2),
  acceptable for overnight evaluation. Kept over MobileNet because MinneApple apples are mostly small.
- In-app training rounds: 6 epochs (RETRAIN_EPOCHS) to keep a round around 10 minutes on CPU.
  Offline learning curve: 10 epochs.
- Smoke test (20 test crops): pretrained AP50 0.569, after 10 crops x 2 epochs AP50 0.654.
- SAM 2.1 tiny smoke test (30 apples, 26 small): mIoU@1 0.66, @5 0.81, box prompt 0.78, NoC@85 10.5.
  Small apples: 1 px boundary error is already about 0.81 IoU for a 20 px apple.