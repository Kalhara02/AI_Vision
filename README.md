# LabelLoop

An AI-assisted image annotation platform, built for the AI Vision course project.
Users upload images, label them with boxes and polygons, get help from computer vision
models while doing it, and export the result in standard formats.

Two models help while labeling, both running locally in our own code:

- **SAM 2.1** (click-to-segment): click an apple, get its outline; right-click to remove
  background; drag a box for a box prompt. Also turns detector boxes into polygons in one batch.
- **Faster R-CNN** (pre-labels): suggestions appear when an image opens. Round 0 uses the
  COCO-pretrained "apple" class. Every 20 finished images (or on **Retrain now**) it is fine-tuned
  on the user's corrected labels with our own training loop, so suggestions improve as the user works.
- **Model guard**: each round trains on ~80% of the finished images and compares new vs. current model
  on the other ~20% (AP50, our own implementation). A worse model is never deployed.
- **Active learning queue**: after each round all unfinished images are ranked by detector
  uncertainty; "Next" can jump to the most uncertain image.
- **Insights**: the project page shows time per image (manual vs. assisted), how many suggestions
  were kept per model version, and every training round, live from the interaction logs.

Demo dataset: MinneApple (orchard apples). See `scripts/README.md` and `eval/README.md`.

## Quick start

Requirements: Docker with Compose v2.

```bash
docker compose up --build
```

Then open **http://localhost:8080**. The first build downloads PyTorch (CPU) and SAM 2, so it takes a while;
model weights download on first use and are cached in the `appdata` volume.

NVIDIA GPU: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build`. The API docs are at http://localhost:8000/docs.

To wipe all data: `docker compose down -v`.

### Without Docker (development)

```bash
# backend (uses SQLite + ./data by default)
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload

# frontend (proxies /api to localhost:8000)
cd frontend && npm install && npm run dev      # http://localhost:5173
```

### Tests

```bash
make test         # all backend tests inside the container
make test-local   # exporter / geometry / ingestion tests on the host (only needs Pillow + NumPy)
```

## Usage

1. **Create a project** with a name and its classes (e.g. `car, truck, bus`).
2. **Upload images** by drag and drop. JPEG, PNG, WebP and TIFF are accepted.
3. **Annotate.** Pick a class with `1`-`9`, draw boxes (`B`) or polygons (`P`), edit with the select tool (`V`).
   Work is autosaved. `Ctrl+Enter` marks the image done and jumps to the next unfinished one.
4. **Export** from the project page as COCO, YOLO (detection or segmentation) or Pascal VOC, with or without images.

### Keyboard shortcuts

| Key | Action |
|---|---|
| `V` / `B` / `P` / `S` | select, box, polygon, smart (SAM) tool |
| Smart: click / right-click / drag | object point / background point / box prompt, `Enter` accepts |
| click / right-click a dashed box | accept / reject a detector suggestion |
| `A` | accept all suggestions |
| `R` | boxes to polygons with SAM (selected box, or all) |
| `1`-`9` | choose class (re-labels the selected object too) |
| `Enter` | close polygon |
| `Backspace` | remove last polygon point, otherwise delete selected |
| `Delete` | delete selected |
| `Esc` | cancel drawing / deselect |
| `Tab` / `Shift+Tab` | next / previous object |
| `Ctrl+Z` / `Ctrl+Shift+Z` | undo / redo |
| `Ctrl+S` | save now |
| `Ctrl+Enter` | mark done, go to next |
| `←` / `→` | previous / next image |
| `Space`+drag, middle drag | pan |
| wheel, `+` / `-`, `F` | zoom, fit |
| `H` | hide annotations |
| `?` | help overlay |

Polygon editing with the select tool: drag a vertex to move it, click a midpoint to insert a vertex,
right-click a vertex to delete it, drag the shape to move it.

## Export formats

| Format | Contents | Notes |
|---|---|---|
| `coco` | `annotations.json`, `images/` | boxes and polygons; category ids start at 1 |
| `yolo` | `labels/*.txt`, `data.yaml`, `images/` | polygons exported as their bounding box |
| `yolo-seg` | `labels/*.txt`, `data.yaml`, `images/` | boxes exported as 4-point polygons |
| `voc` | `Annotations/*.xml`, `JPEGImages/` | 1-based inclusive integer corners |

File names inside the archive are prefixed with the image id (`000042_name.jpg`), so two uploads
with the same name never collide. Every COCO annotation carries `attributes.source`
(`manual`, `sam` or `detector`), which records how much of the dataset came from the AI.

## Architecture

```
browser ── nginx (frontend container) ──/api──▶ FastAPI backend ──▶ PostgreSQL
           React + TypeScript + Konva          SQLAlchemy           (metadata, annotations, logs)
                                                    │
                                                    └──▶ /data volume (images)
```

- **Frontend** (`frontend/`): React + TypeScript, canvas drawing with Konva. All geometry is kept in
  image pixel coordinates; the stage applies zoom and pan, so shapes never drift. Undo/redo is a
  snapshot history on the client. Saving replaces an image's annotation set in one request,
  which keeps client and server consistent without a sync protocol.
- **Backend** (`backend/app/`): FastAPI with SQLAlchemy. Upload normalises images: EXIF rotation is
  baked in (otherwise boxes drawn in the browser would not match what a model sees), and 16-bit
  TIFFs are stretched to 8 bit so browsers can show them.
- **Exporters** (`backend/app/export/`): plain functions without framework dependencies, unit tested,
  and reused later by the evaluation scripts.
- **Interaction logging**: every action (pointer down, shape created, vertex moved, class changed,
  undo, shortcut, image open/done, ...) is logged with a timestamp and the session mode
  (`manual` or `assisted`). This is the raw data for measuring effort saved.
  Download it per project from the project page or `GET /api/projects/{id}/logs`.

### Data model

`Project` ─< `LabelClass`, `Project` ─< `Image` ─< `Annotation(kind, bbox, points, source, score)`,
plus `InteractionEvent(session, mode, event, payload, t)`.

## What is ours and what is used

Following the course rules, all application logic, the pipeline and the model work are written for
this project. Libraries are used only as building blocks; no existing annotation tool is used,
forked or wrapped, and no model is called through a hosted API.

| Built for this project | Used as a building block |
|---|---|
| Annotation UI: drawing, editing, vertex tools, undo/redo, shortcuts, autosave | React, Konva / react-konva (generic 2D canvas) |
| REST API, data model, image ingestion | FastAPI, SQLAlchemy, Pydantic, PostgreSQL, Pillow, NumPy |
| COCO / YOLO / VOC exporters | Python standard library |
| Interaction logger and effort metrics | |
| SAM integration: encoder cache, click/box prompting, mask to polygon, batch box refinement | SAM 2.1 (`sam2` package, Meta), pretrained `sam2.1-hiera-tiny` |
| Detector service, cold start via COCO "apple", fine-tuning loop (data, augmentation, optimiser, schedule, versioning) | torchvision Faster R-CNN ResNet-50 FPN v2, COCO weights, PyTorch |
| Training rounds: auto trigger, hold-out guard, AP50 metric, uncertainty ranking (`app/ml/metrics.py`), next-image queue | NumPy |
| Effort analytics shared by the app and the report (`app/analytics.py`) | NumPy |
| Evaluation: learning curve, NoC click simulation, simulated annotator, active learning vs. random, effort + quality from logs, failure gallery | pycocotools (mAP), SciPy (Hungarian matching, Wilcoxon), Matplotlib |
| Data preparation (split, mask to COCO, crops) | OpenCV, NumPy |

## Project structure

```
backend/
  app/
    main.py            FastAPI app
    models.py          database schema
    schemas.py         request/response validation
    storage.py         image ingestion (EXIF, 16-bit)
    geometry.py        shared geometry (area, IoU, clipping)
    routers/           projects, images + annotations, logs, export
    export/            format writers and zip packaging
  tests/               exporter/geometry/storage tests, end-to-end API test
frontend/
  src/components/      ProjectList, ProjectPage, AnnotatePage, AnnotationCanvas
  src/lib/             API client, logger, undo history, router, geometry
eval/                  evaluation scripts (phase 4)
docker-compose.yml
```

## Design decisions and limitations

- **SAM decoding runs on the server.** The encoder runs once per image (cached), each click then only runs the
  light decoder. A browser-side ONNX decoder would remove the network round trip, but SAM 2 has no official
  ONNX export; we measured click latency instead (logged per click) and list it as future work.
- **Faster R-CNN instead of YOLO**: torchvision lets us write and show the whole training loop ourselves.
- **Full replacement on save**: the browser owns undo/redo; one request replaces an image's annotations.
- **Limitations**: one class and one domain; CPU training is slow (GPU compose file provided); the user study
  is small; our MinneApple split differs from the official benchmark's test set.

## References

- MinneApple: N. Häni, P. Roy, V. Isler. MinneApple: A Benchmark Dataset for Apple Detection and Segmentation. IEEE RA-L, 2020.
- SAM 2: N. Ravi et al. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714, 2024.
- Faster R-CNN: S. Ren, K. He, R. Girshick, J. Sun. NeurIPS 2015. torchvision v2 weights: Y. Li et al., 2021.
- COCO: T.-Y. Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014.
- Active learning: B. Settles. Active Learning Literature Survey. University of Wisconsin-Madison, TR 1648, 2009.
- NoC protocol: K. Sofiiuk, I. Petrov, A. Konushin. Reviving Iterative Training with Mask Guidance for Interactive Segmentation (RITM). ICIP 2022.
