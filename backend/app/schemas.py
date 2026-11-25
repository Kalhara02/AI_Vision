from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .geometry import bbox_from_points

ImageStatus = Literal["unlabeled", "in_progress", "done"]
Kind = Literal["bbox", "polygon"]
Source = Literal["manual", "sam", "detector"]


# ---------- classes ----------
class ClassIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")


class ClassPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")


class ClassOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    color: str


# ---------- projects ----------
class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    classes: list[ClassIn] = []


class ProjectOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: str
    created_at: datetime
    classes: list[ClassOut]
    image_count: int = 0
    done_count: int = 0
    annotation_count: int = 0
    cover_image_id: int | None = None


# ---------- images ----------
class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    project_id: int
    filename: str
    width: int
    height: int
    status: ImageStatus
    annotation_count: int = 0


class ImagePatch(BaseModel):
    status: ImageStatus


class UploadError(BaseModel):
    filename: str
    error: str


class UploadResult(BaseModel):
    created: list[ImageOut]
    errors: list[UploadError]


# ---------- annotations ----------
class AnnotationIn(BaseModel):
    class_id: int
    kind: Kind
    bbox: list[float] | None = None
    points: list[list[float]] | None = None
    source: Source = "manual"
    score: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _check_geometry(self):
        if self.kind == "polygon":
            if not self.points or len(self.points) < 3:
                raise ValueError("a polygon needs at least 3 points")
            if any(len(p) != 2 for p in self.points):
                raise ValueError("polygon points must be [x, y] pairs")
            self.bbox = bbox_from_points(self.points)
        else:
            if self.bbox is None or len(self.bbox) != 4:
                raise ValueError("a bbox must be [x, y, w, h]")
            if self.bbox[2] <= 0 or self.bbox[3] <= 0:
                raise ValueError("bbox width and height must be positive")
            self.points = None
        return self


class AnnotationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    class_id: int
    kind: Kind
    bbox: list[float]
    points: list[list[float]] | None
    source: Source
    score: float | None


class AnnotationSetIn(BaseModel):
    """Full replacement of an image's annotations (the client owns undo/redo state)."""

    annotations: list[AnnotationIn] = Field(max_length=5000)
    status: ImageStatus | None = None


class AnnotationSetOut(BaseModel):
    image: ImageOut
    annotations: list[AnnotationOut]


# ---------- interaction logs ----------
class EventIn(BaseModel):
    session_id: str = Field(max_length=64)
    mode: Literal["manual", "assisted"]
    event: str = Field(max_length=64)
    t: float  # ms since epoch
    project_id: int | None = None
    image_id: int | None = None
    payload: dict = {}


class EventBatch(BaseModel):
    events: list[EventIn] = Field(max_length=2000)
