"""Runtime configuration, read from environment variables.

Defaults let the backend run locally without Docker (SQLite + ./data).
"""
import os
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./labelloop.db")
DATA_DIR = Path(os.getenv("DATA_DIR", "./data")).resolve()
IMAGE_DIR = DATA_DIR / "images"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ---- AI models (all run locally; weights are downloaded once into the data volume) ----
AI_ENABLED = os.getenv("AI_ENABLED", "1") == "1"
SAM_MODEL_ID = os.getenv("SAM_MODEL_ID", "facebook/sam2.1-hiera-tiny")
DETECTOR_ARCH = os.getenv("DETECTOR_ARCH", "resnet50_v2")  # resnet50_v2 | mobilenet_v3
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
AUTO_RETRAIN_EVERY = int(os.getenv("AUTO_RETRAIN_EVERY", "20"))  # finished images per round, 0 = off
RETRAIN_EPOCHS = int(os.getenv("RETRAIN_EPOCHS", "10"))

# Zero-shot suggestions before any training: an open-vocabulary detector prompted with the
# project's own class names (OWLv2, Minderer et al. 2023). Its raw scores are low (true objects
# often 0.1 to 0.4), so they are rescaled for the UI: raw ZERO_SHOT_MID shows as 0.50.
ZERO_SHOT_MODEL = os.getenv("ZERO_SHOT_MODEL", "google/owlv2-base-patch16-ensemble")
ZERO_SHOT_ENABLED = os.getenv("ZERO_SHOT_ENABLED", "1") == "1"
ZERO_SHOT_MID = float(os.getenv("ZERO_SHOT_MID", "0.15"))
