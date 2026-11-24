from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .. import models, schemas, storage
from ..db import get_db
from ..geometry import bbox_from_points, clip_bbox, clip_points

router = APIRouter(prefix="/api", tags=["images"])


def _get_image(db: Session, image_id: int) -> models.Image:
    image = db.get(models.Image, image_id)
    if image is None:
        raise HTTPException(404, "image not found")
    return image


def _image_out(image: models.Image, count: int) -> schemas.ImageOut:
    out = schemas.ImageOut.model_validate(image)
    out.annotation_count = count
    return out


@router.post("/projects/{project_id}/images", response_model=schemas.UploadResult, status_code=201)
async def upload_images(project_id: int, files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    if db.get(models.Project, project_id) is None:
        raise HTTPException(404, "project not found")
    created, errors = [], []
    for f in files:
        data = await f.read()
        try:
            stored, w, h = storage.save_image(data)
        except storage.InvalidImage as exc:
            errors.append(schemas.UploadError(filename=f.filename or "?", error=str(exc)))
            continue
        image = models.Image(project_id=project_id, filename=(f.filename or stored)[:255],
                             stored_name=stored, width=w, height=h, status="unlabeled")
        db.add(image)
        db.flush()
        created.append(_image_out(image, 0))
    db.commit()
    return schemas.UploadResult(created=created, errors=errors)


@router.get("/projects/{project_id}/images", response_model=list[schemas.ImageOut])
def list_images(project_id: int, db: Session = Depends(get_db)):
    counts = dict(db.execute(
        select(models.Annotation.image_id, func.count())
        .join(models.Image).where(models.Image.project_id == project_id)
        .group_by(models.Annotation.image_id)
    ).all())
    images = db.scalars(select(models.Image).where(models.Image.project_id == project_id).order_by(models.Image.id)).all()
    return [_image_out(im, counts.get(im.id, 0)) for im in images]


@router.get("/images/{image_id}/file")
def image_file(image_id: int, db: Session = Depends(get_db)):
    image = _get_image(db, image_id)
    path = storage.image_path(image.stored_name)
    if not path.exists():
        raise HTTPException(404, "image file missing on disk")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})


@router.patch("/images/{image_id}", response_model=schemas.ImageOut)
def update_image(image_id: int, body: schemas.ImagePatch, db: Session = Depends(get_db)):
    image = _get_image(db, image_id)
    image.status = body.status
    db.commit()
    return _image_out(image, len(image.annotations))


@router.delete("/images/{image_id}", status_code=204)
def delete_image(image_id: int, db: Session = Depends(get_db)):
    image = _get_image(db, image_id)
    stored = image.stored_name
    db.delete(image)
    db.commit()
    storage.delete_file(stored)


# ---------- annotations ----------
@router.get("/images/{image_id}/annotations", response_model=schemas.AnnotationSetOut)
def get_annotations(image_id: int, db: Session = Depends(get_db)):
    image = _get_image(db, image_id)
    return schemas.AnnotationSetOut(
        image=_image_out(image, len(image.annotations)),
        annotations=[schemas.AnnotationOut.model_validate(a) for a in image.annotations],
    )


@router.put("/images/{image_id}/annotations", response_model=schemas.AnnotationSetOut)
def replace_annotations(image_id: int, body: schemas.AnnotationSetIn, db: Session = Depends(get_db)):
    """Replace all annotations of an image in one transaction.

    The browser owns the editing state (including undo/redo), so a full
    replacement keeps client and server trivially consistent.
    """
    image = _get_image(db, image_id)
    valid_classes = set(db.scalars(
        select(models.LabelClass.id).where(models.LabelClass.project_id == image.project_id)
    ).all())

    new_rows = []
    for i, a in enumerate(body.annotations):
        if a.class_id not in valid_classes:
            raise HTTPException(422, f"annotation {i}: class {a.class_id} does not belong to this project")
        if a.kind == "polygon":
            pts = clip_points(a.points, image.width, image.height)
            bbox = bbox_from_points(pts)
        else:
            pts = None
            bbox = clip_bbox(a.bbox, image.width, image.height)
        if bbox[2] <= 0 or bbox[3] <= 0:
            continue  # fully outside the image after clipping
        new_rows.append(models.Annotation(class_id=a.class_id, kind=a.kind, bbox=bbox, points=pts,
                                          source=a.source, score=a.score))

    newly_done = body.status == "done" and image.status != "done"
    image.annotations.clear()
    db.flush()
    image.annotations.extend(new_rows)
    if body.status is not None:
        image.status = body.status
    elif new_rows and image.status == "unlabeled":
        image.status = "in_progress"
    db.commit()
    db.refresh(image)
    if newly_done:
        from .ai import maybe_auto_retrain  # human-in-the-loop: every N finished images starts a round
        maybe_auto_retrain(db, image.project_id)
    return schemas.AnnotationSetOut(
        image=_image_out(image, len(image.annotations)),
        annotations=[schemas.AnnotationOut.model_validate(a) for a in image.annotations],
    )
