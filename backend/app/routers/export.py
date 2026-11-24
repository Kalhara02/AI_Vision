import os
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from .. import models, storage
from ..db import get_db
from ..export.formats import ExpAnnotation, ExpImage
from ..export.package import FORMATS, build_zip, safe_name

router = APIRouter(prefix="/api", tags=["export"])


@router.get("/projects/{project_id}/export")
def export_project(
    project_id: int,
    format: str = Query("coco", description=f"one of {', '.join(FORMATS)}"),
    include_images: bool = True,
    only_done: bool = False,
    db: Session = Depends(get_db),
):
    if format not in FORMATS:
        raise HTTPException(422, f"format must be one of {FORMATS}")
    project = db.get(models.Project, project_id)
    if project is None:
        raise HTTPException(404, "project not found")

    class_index = {c.id: i for i, c in enumerate(project.classes)}
    categories = [c.name for c in project.classes]

    q = select(models.Image).where(models.Image.project_id == project_id).order_by(models.Image.id)
    if only_done:
        q = q.where(models.Image.status == "done")
    images = db.scalars(q).all()

    exp_images, exp_anns, files = [], [], {}
    for im in images:
        ext = Path(im.stored_name).suffix
        exp_images.append(ExpImage(id=im.id, file_name=safe_name(im.id, im.filename, ext),
                                   width=im.width, height=im.height))
        files[im.id] = storage.image_path(im.stored_name)
        for a in im.annotations:
            exp_anns.append(ExpAnnotation(image_id=im.id, category=class_index[a.class_id], kind=a.kind,
                                          bbox=a.bbox, points=a.points, score=a.score, source=a.source))

    path = build_zip(format, exp_images, exp_anns, categories, files, include_images, project.name)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", project.name).strip("_") or "project"
    return FileResponse(
        path,
        media_type="application/zip",
        filename=f"{slug}_{format}.zip",
        background=BackgroundTask(os.unlink, path),
    )
