from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import models, schemas, storage
from ..db import get_db

router = APIRouter(prefix="/api", tags=["projects"])

# Distinguishable colours, roughly colour-blind safe (Okabe-Ito + extras)
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
           "#8DD3C7", "#FB8072", "#80B1D3", "#B3DE69", "#FCCDE5"]


def _next_color(project: models.Project) -> str:
    used = {c.color.lower() for c in project.classes}
    for c in PALETTE:
        if c.lower() not in used:
            return c
    return PALETTE[len(project.classes) % len(PALETTE)]


def _get_project(db: Session, project_id: int) -> models.Project:
    project = db.get(models.Project, project_id)
    if project is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "project not found")
    return project


def _project_out(db: Session, p: models.Project) -> schemas.ProjectOut:
    total = db.scalar(select(func.count()).select_from(models.Image).where(models.Image.project_id == p.id)) or 0
    done = db.scalar(
        select(func.count()).select_from(models.Image)
        .where(models.Image.project_id == p.id, models.Image.status == "done")
    ) or 0
    out = schemas.ProjectOut.model_validate(p)
    out.image_count, out.done_count = total, done
    return out


@router.get("/projects", response_model=list[schemas.ProjectOut])
def list_projects(db: Session = Depends(get_db)):
    projects = db.scalars(select(models.Project).order_by(models.Project.id.desc())).all()
    return [_project_out(db, p) for p in projects]


@router.post("/projects", response_model=schemas.ProjectOut, status_code=201)
def create_project(body: schemas.ProjectIn, db: Session = Depends(get_db)):
    names = [c.name.strip() for c in body.classes]
    if len(names) != len(set(names)):
        raise HTTPException(422, "class names must be unique")
    project = models.Project(name=body.name.strip(), description=body.description)
    db.add(project)
    for c in body.classes:
        project.classes.append(models.LabelClass(name=c.name.strip(), color=c.color or _next_color(project)))
    db.commit()
    db.refresh(project)
    return _project_out(db, project)


@router.get("/projects/{project_id}", response_model=schemas.ProjectOut)
def get_project(project_id: int, db: Session = Depends(get_db)):
    return _project_out(db, _get_project(db, project_id))


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = _get_project(db, project_id)
    files = [im.stored_name for im in project.images]
    # annotations reference classes too, so remove them first to keep FK order explicit
    image_ids = select(models.Image.id).where(models.Image.project_id == project_id)
    db.execute(delete(models.Annotation).where(models.Annotation.image_id.in_(image_ids)))
    db.expire_all()
    db.delete(project)
    db.commit()
    for f in files:
        storage.delete_file(f)


# ---------- label classes ----------
@router.post("/projects/{project_id}/classes", response_model=schemas.ClassOut, status_code=201)
def add_class(project_id: int, body: schemas.ClassIn, db: Session = Depends(get_db)):
    project = _get_project(db, project_id)
    cls = models.LabelClass(name=body.name.strip(), color=body.color or _next_color(project))
    project.classes.append(cls)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, f"class '{body.name}' already exists")
    db.refresh(cls)
    return cls


@router.patch("/classes/{class_id}", response_model=schemas.ClassOut)
def update_class(class_id: int, body: schemas.ClassPatch, db: Session = Depends(get_db)):
    cls = db.get(models.LabelClass, class_id)
    if cls is None:
        raise HTTPException(404, "class not found")
    if body.name is not None:
        cls.name = body.name.strip()
    if body.color is not None:
        cls.color = body.color
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, f"class '{body.name}' already exists")
    db.refresh(cls)
    return cls


@router.delete("/classes/{class_id}", status_code=204)
def delete_class(class_id: int, db: Session = Depends(get_db)):
    cls = db.get(models.LabelClass, class_id)
    if cls is None:
        raise HTTPException(404, "class not found")
    in_use = db.scalar(select(func.count()).select_from(models.Annotation).where(models.Annotation.class_id == class_id))
    if in_use:
        raise HTTPException(409, f"class is used by {in_use} annotation(s); reassign or delete them first")
    db.delete(cls)
    db.commit()
