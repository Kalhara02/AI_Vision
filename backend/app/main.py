from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import models  # noqa: F401  (registers tables)
from .db import Base, engine
from .routers import ai, export, images, logs, projects


@asynccontextmanager
async def lifespan(_: FastAPI):
    # The schema is small and owned by this app, so create_all is enough (no migrations).
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="LabelLoop API", version="0.1.0", lifespan=lifespan,
              description="AI-assisted image annotation platform (course project)")


@app.get("/api/health")
def health():
    return {"status": "ok"}


app.include_router(projects.router)
app.include_router(images.router)
app.include_router(logs.router)
app.include_router(export.router)
app.include_router(ai.router)
