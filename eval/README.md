# Evaluation

Everything runs inside the backend container (PyTorch, SAM 2, pycocotools are installed there).
`./data` and `./eval` are mounted into it, results land in `eval/results/`.
Prepare the data first with `make data ...` (see `scripts/README.md`).

| Command | What it measures | Output |
| --- | --- | --- |
| `make eval-quick` | smoke test, a few minutes | checks that models load and scripts run |
| `make eval-detector` | box AP on test crops: pretrained COCO "apple" vs. our fine-tuning on 25/50/100/200 crops | `detector_curve.csv/.png` |
| `make eval-sam` | SAM 2.1 plain vs. focus mode: NoC@80/85/90, mIoU after k clicks, box-prompt IoU, ms per click, by apple size | `sam_noc_summary.json`, `sam_miou_curve.png` |
| `make eval-failures` | worst SAM cases, detector misses and false alarms | `failures_sam.png`, `failures_detector.png` |
| `make eval-sim` | simulated annotator: seconds per crop for manual, SAM only, detector+SAM (pretrained vs. fine-tuned); action costs calibrated from your logs if `data/logs.json` exists | `simulated_effort.json/.png` |
| `make eval-al` | active learning: uncertainty vs. random sampling over 4 rounds | `active_learning.csv/.png` |
| `make eval-effort` | manual vs. assisted time, clicks, quality vs. ground truth, suggestion acceptance | `effort_summary.md`, `effort_time.png` |
| `make eval-v5` | the model the app trained (project 5) on the MinneApple test crops | `app_model_p5_v5.json` |
| `make data-coco` + `make eval-generalize` | second domain (6 COCO classes): zero-shot OWLv2 vs. the learning loop after 25/50/100 images, and SAM NoC | `generalize_coco_sub.json` |

## User study (effort)

1. In the app, create a project "Study" with class `apple`. Upload crops from
   `data/minneapple/crops_study/images` (use different crops for manual and for assisted).
2. On the project page set the **Participant ID** (P1, P2, ...) and the **mode** before each block.
   Counterbalance: P1 manual first, P2 assisted first, and so on.
3. Every crop must end with **Done** (Ctrl+Enter); only finished crops are measured.
4. Download the logs: `http://localhost:8000/api/projects/<id>/logs` -> save as `data/logs.json`.
5. Export the project as COCO (images not needed), unzip `annotations.json` -> `data/study_export.json`.
6. `make eval-effort`

Notes: active time ignores gaps longer than 60 s (`--idle`). Quality uses one-to-one
matching on mask IoU against the MinneApple instance masks; a match needs IoU >= 0.5.

Order on a full run: `make eval-detector`, `make eval-sam`, then `make eval-sim` and `make eval-failures`
(they reuse the first two's outputs). After the user study: `make eval-effort`, and `make eval-sim` again so
the simulation uses your measured action times. `make eval-al` is independent and the slowest.

The app shows the same effort numbers live (project page, Insights), computed by the same code (`backend/app/analytics.py`).
