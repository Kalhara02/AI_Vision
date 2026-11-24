"""Database schema.

Annotations are stored in image pixel coordinates. A bounding box is always
stored (also for polygons, where it is derived from the points), so every
exporter and every metric can rely on it.
"""
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    classes: Mapped[list["LabelClass"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="LabelClass.id"
    )
    images: Mapped[list["Image"]] = relationship(
        back_populates="project", cascade="all, delete-orphan", order_by="Image.id"
    )


class LabelClass(Base):
    __tablename__ = "label_classes"
    __table_args__ = (UniqueConstraint("project_id", "name", name="uq_class_name_per_project"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(100))
    color: Mapped[str] = mapped_column(String(9))

    project: Mapped[Project] = relationship(back_populates="classes")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))  # original upload name
    stored_name: Mapped[str] = mapped_column(String(255), unique=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    # unlabeled -> in_progress -> done
    status: Mapped[str] = mapped_column(String(20), default="unlabeled", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    project: Mapped[Project] = relationship(back_populates="images")
    annotations: Mapped[list["Annotation"]] = relationship(
        back_populates="image", cascade="all, delete-orphan", order_by="Annotation.id"
    )


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"), index=True)
    class_id: Mapped[int] = mapped_column(ForeignKey("label_classes.id"), index=True)
    kind: Mapped[str] = mapped_column(String(16))  # "bbox" | "polygon"
    bbox: Mapped[list] = mapped_column(JSON)  # [x, y, w, h] in pixels
    points: Mapped[list | None] = mapped_column(JSON, nullable=True)  # [[x, y], ...] for polygons
    # where the annotation came from; lets us measure how much the AI contributed
    source: Mapped[str] = mapped_column(String(16), default="manual")  # manual | sam | detector
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    image: Mapped[Image] = relationship(back_populates="annotations")


class InteractionEvent(Base):
    """One user action in the annotation UI. Used to measure annotation effort."""

    __tablename__ = "interaction_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    project_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    image_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    mode: Mapped[str] = mapped_column(String(16))  # manual | assisted
    event: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    t_client: Mapped[float] = mapped_column(Float)  # ms since epoch, browser clock
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
