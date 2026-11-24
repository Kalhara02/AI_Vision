"""pytest loads this before any test module, so every test uses a throwaway
SQLite database and data folder (never the real Postgres instance)."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="labelloop_pytest_")
os.environ["DATA_DIR"] = _tmp
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"

os.environ.setdefault("AI_ENABLED", "0")  # API tests never load PyTorch models
