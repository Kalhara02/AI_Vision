"""Interaction logging. These events are the raw data for the effort evaluation
(time per image, clicks per object, manual vs. assisted)."""
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import models, schemas
from ..db import get_db

router = APIRouter(prefix="/api", tags=["logs"])


@router.post("/logs", status_code=202)
def ingest(batch: schemas.EventBatch, db: Session = Depends(get_db)):
    db.add_all([
        models.InteractionEvent(
            session_id=e.session_id, project_id=e.project_id, image_id=e.image_id,
            mode=e.mode, event=e.event, payload=e.payload, t_client=e.t,
        )
        for e in batch.events
    ])
    db.commit()
    return {"accepted": len(batch.events)}


@router.get("/projects/{project_id}/logs")
def export_logs(project_id: int, db: Session = Depends(get_db)):
    rows = db.scalars(
        select(models.InteractionEvent)
        .where(models.InteractionEvent.project_id == project_id)
        .order_by(models.InteractionEvent.t_client)
    ).all()
    return [
        {"session_id": r.session_id, "image_id": r.image_id, "mode": r.mode, "event": r.event,
         "t": r.t_client, "payload": r.payload}
        for r in rows
    ]


@router.get("/projects/{project_id}/insights")
def insights(project_id: int, idle: float = 60.0, db: Session = Depends(get_db)):
    """Evidence inside the app: effort per mode, suggestion acceptance per model version, model rounds."""
    from ..analytics import acceptance, per_image_effort, summarize_by_mode
    from ..config import AI_ENABLED

    events = [{"session_id": r.session_id, "image_id": r.image_id, "mode": r.mode, "event": r.event,
               "t": r.t_client, "payload": r.payload or {}}
              for r in db.scalars(select(models.InteractionEvent)
                                  .where(models.InteractionEvent.project_id == project_id)).all()]
    rows = per_image_effort(events, idle)
    modes = summarize_by_mode(rows)
    saved = None
    m = {x["mode"]: x for x in modes}
    if "manual" in m and "assisted" in m and m["manual"]["active_s_mean"] > 0:
        saved = round(100 * (1 - m["assisted"]["active_s_mean"] / m["manual"]["active_s_mean"]), 1)
    rounds = []
    if AI_ENABLED:
        try:
            from ..ml.detector import detector_service
            rounds = detector_service.status(project_id)["rounds"]
        except ImportError:
            pass
    return {"modes": modes, "time_saved_percent": saved, "acceptance": acceptance(events), "rounds": rounds,
            "n_events": len(events)}
