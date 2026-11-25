"""AI assistance endpoints: SAM click-to-segment, detector proposals, retraining.

Model code is imported lazily, so the rest of the API (and its tests) works
without PyTorch installed.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, storage
from ..config import AI_ENABLED, AUTO_RETRAIN_EVERY, RETRAIN_EPOCHS
from ..db import get_db
from ..geometry import clip_points

router = APIRouter(prefix="/api", tags=["ai"])


class SamPredictIn(BaseModel):
    points: list[list[float]] = Field(default_factory=list, max_length=50)  # [x, y, 1|0]
    box: list[float] | None = None  # [x, y, w, h]
    focus: bool = False  # legacy switch; `mode` wins when given
    mode: Literal["plain", "focus", "hybrid"] | None = None  # see app/ml/sam.py


class SamBoxesIn(BaseModel):
    boxes: list[list[float]] = Field(max_length=500)


def _require_ai():
    if not AI_ENABLED:
        raise HTTPException(503, "AI assistance is disabled (AI_ENABLED=0)")


def _image(db: Session, image_id: int) -> models.Image:
    image = db.get(models.Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    return image


def _sam():
    _require_ai()
    from ..ml.sam import SamUnavailable, sam_service
    return sam_service, SamUnavailable


def _detector():
    _require_ai()
    from ..ml.detector import detector_service
    return detector_service


@router.get("/ai/status")
def ai_status():
    if not AI_ENABLED:
        return {"enabled": False}
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return {"enabled": False, "reason": "PyTorch not installed"}
    from ..ml.sam import sam_service
    return {"enabled": True, "device": dev, "sam_model": sam_service.model_id, "sam_loaded": sam_service.loaded}


# ---------------------------------------------------------------- SAM
@router.post("/images/{image_id}/sam/prepare")
def sam_prepare(image_id: int, db: Session = Depends(get_db)):
    """Run the image encoder now (called when an image opens) so the first click is fast."""
    sam, Unavailable = _sam()
    image = _image(db, image_id)
    try:
        ms = sam.prepare(str(image_id), storage.image_path(image.stored_name))
    except Unavailable as exc:
        raise HTTPException(503, str(exc))
    return {"encode_ms": round(ms, 1)}


@router.post("/images/{image_id}/sam/predict")
def sam_predict(image_id: int, body: SamPredictIn, db: Session = Depends(get_db)):
    if not body.points and body.box is None:
        raise HTTPException(422, "give at least one point or a box")
    sam, Unavailable = _sam()
    image = _image(db, image_id)
    try:
        res = sam.predict(str(image_id), storage.image_path(image.stored_name), body.points, body.box,
                          focus=body.focus, mode=body.mode)
    except Unavailable as exc:
        raise HTTPException(503, str(exc))
    if res["points"]:
        res["points"] = clip_points(res["points"], image.width, image.height)
    return res


@router.post("/images/{image_id}/sam/boxes")
def sam_boxes(image_id: int, body: SamBoxesIn, db: Session = Depends(get_db)):
    sam, Unavailable = _sam()
    image = _image(db, image_id)
    try:
        res = sam.boxes_to_polygons(str(image_id), storage.image_path(image.stored_name), body.boxes)
    except Unavailable as exc:
        raise HTTPException(503, str(exc))
    res["polygons"] = [clip_points(p, image.width, image.height) if p else None for p in res["polygons"]]
    return res


# ---------------------------------------------------------------- detector
@router.get("/images/{image_id}/proposals")
def proposals(image_id: int, threshold: float = Query(0.5, ge=0.0, le=1.0), db: Session = Depends(get_db)):
    det = _detector()
    image = _image(db, image_id)
    return det.predict(image.project_id, storage.image_path(image.stored_name), threshold)


@router.get("/projects/{project_id}/model")
def model_status(project_id: int):
    return _detector().status(project_id)


def _training_inputs(db: Session, project_id: int):
    from ..ml.detector import Sample

    images = db.scalars(select(models.Image).where(models.Image.project_id == project_id)
                        .order_by(models.Image.id)).all()
    samples = [Sample(storage.image_path(im.stored_name), [a.bbox for a in im.annotations])
               for im in images if im.status == "done" and im.annotations]
    to_rank = [(im.id, storage.image_path(im.stored_name)) for im in images if im.status != "done"]
    return samples, to_rank


def maybe_auto_retrain(db: Session, project_id: int) -> bool:
    """Called whenever an image is marked done: start a round every AUTO_RETRAIN_EVERY new images."""
    if not AI_ENABLED or AUTO_RETRAIN_EVERY <= 0:
        return False
    try:
        from ..ml.detector import TrainConfig, detector_service as det
    except ImportError:
        return False
    if det.is_busy(project_id):
        return False
    samples, to_rank = _training_inputs(db, project_id)
    if len(samples) < 5 or len(samples) - det.images_seen(project_id) < AUTO_RETRAIN_EVERY:
        return False
    return det.start_training(project_id, samples, TrainConfig(epochs=RETRAIN_EPOCHS), "auto", to_rank)


@router.post("/projects/{project_id}/model/train", status_code=202)
def train(project_id: int, epochs: int = Query(RETRAIN_EPOCHS, ge=1, le=100), db: Session = Depends(get_db)):
    """Start a training round now on every finished image. Runs in the background; poll GET .../model."""
    det = _detector()
    from ..ml.detector import TrainConfig

    samples, to_rank = _training_inputs(db, project_id)
    if len(samples) < 5:
        raise HTTPException(409, f"need at least 5 finished images with annotations, have {len(samples)}")
    if not det.start_training(project_id, samples, TrainConfig(epochs=epochs), "manual", to_rank):
        raise HTTPException(409, "the model is busy (training or ranking)")
    return {"started": True, "n_images": len(samples), "auto_every": AUTO_RETRAIN_EVERY}


@router.post("/projects/{project_id}/model/rank", status_code=202)
def rank(project_id: int, db: Session = Depends(get_db)):
    """Score every unfinished image by detector uncertainty (active learning queue)."""
    det = _detector()
    _, to_rank = _training_inputs(db, project_id)
    if not det.start_ranking(project_id, to_rank):
        raise HTTPException(409, "nothing to rank, or the model is busy")
    return {"started": True, "n_images": len(to_rank)}


@router.get("/projects/{project_id}/next")
def next_image(project_id: int, current: int | None = None, order: str = Query("upload", pattern="^(upload|uncertainty)$"),
               db: Session = Depends(get_db)):
    """Next unfinished image: by upload order, or the most uncertain one (falls back to upload order)."""
    open_ids = [i for i, in db.execute(
        select(models.Image.id).where(models.Image.project_id == project_id, models.Image.status != "done")
        .order_by(models.Image.id)).all() if i != current]
    if not open_ids:
        return {"image_id": None, "order": order, "score": None}
    if order == "uncertainty" and AI_ENABLED:
        from ..ml.detector import detector_service
        scores = detector_service.scores(project_id)["scores"]
        ranked = [i for i in open_ids if str(i) in scores]
        if ranked:
            best = max(ranked, key=lambda i: scores[str(i)])
            return {"image_id": best, "order": "uncertainty", "score": scores[str(best)]}
    after = [i for i in open_ids if current is None or i > current]
    return {"image_id": (after or open_ids)[0], "order": "upload", "score": None}
