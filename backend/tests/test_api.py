"""End-to-end API test: create project, upload, annotate, export.

Runs against a throwaway SQLite database. Needs the backend requirements installed
(it is skipped otherwise), e.g.  docker compose run --rm backend python -m pytest tests
"""
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

_tmp = tempfile.mkdtemp(prefix="labelloop_api_")
os.environ["DATA_DIR"] = _tmp
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/test.db"
os.environ.setdefault("AI_ENABLED", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from fastapi.testclient import TestClient
    from PIL import Image

    from app.main import app
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"backend dependencies missing: {exc}")


def _png(w=64, h=48) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (0, 120, 200)).save(buf, format="PNG")
    return buf.getvalue()


class ApiFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = TestClient(app)
        cls.client = cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def test_full_flow(self):
        c = self.client
        r = c.post("/api/projects", json={"name": "demo", "classes": [{"name": "car"}, {"name": "tree"}]})
        self.assertEqual(r.status_code, 201, r.text)
        project = r.json()
        car, tree = project["classes"]

        r = c.post(f"/api/projects/{project['id']}/images",
                   files=[("files", ("a.png", _png(), "image/png")), ("files", ("bad.png", b"xx", "image/png"))])
        self.assertEqual(r.status_code, 201, r.text)
        body = r.json()
        self.assertEqual(len(body["created"]), 1)
        self.assertEqual(len(body["errors"]), 1)
        image = body["created"][0]

        anns = [
            {"class_id": car["id"], "kind": "bbox", "bbox": [5, 5, 20, 10]},
            {"class_id": tree["id"], "kind": "polygon", "points": [[30, 5], [60, 5], [60, 100]], "source": "sam"},
        ]
        r = c.put(f"/api/images/{image['id']}/annotations", json={"annotations": anns})
        self.assertEqual(r.status_code, 200, r.text)
        saved = r.json()
        self.assertEqual(saved["image"]["status"], "in_progress")
        poly = saved["annotations"][1]
        self.assertEqual(poly["points"][2], [60, 48])  # clipped to image height
        self.assertEqual(poly["bbox"], [30, 5, 30, 43])

        # class from another project is rejected
        r = c.put(f"/api/images/{image['id']}/annotations",
                  json={"annotations": [{"class_id": 9999, "kind": "bbox", "bbox": [0, 0, 1, 1]}]})
        self.assertEqual(r.status_code, 422)

        # class in use cannot be deleted
        self.assertEqual(c.delete(f"/api/classes/{car['id']}").status_code, 409)

        r = c.put(f"/api/images/{image['id']}/annotations", json={"annotations": anns, "status": "done"})
        self.assertEqual(r.json()["image"]["status"], "done")

        r = c.post("/api/logs", json={"events": [
            {"session_id": "s1", "mode": "manual", "event": "box_created", "t": 1.0,
             "project_id": project["id"], "image_id": image["id"], "payload": {"w": 20}}]})
        self.assertEqual(r.status_code, 202)
        self.assertEqual(len(c.get(f"/api/projects/{project['id']}/logs").json()), 1)
        ins = c.get(f"/api/projects/{project['id']}/insights").json()
        self.assertEqual(ins["n_events"], 1)
        self.assertIn("modes", ins)
        nxt = c.get(f"/api/projects/{project['id']}/next", params={"current": image["id"]}).json()
        self.assertIsNone(nxt["image_id"])  # the only image is done

        for fmt in ("coco", "yolo", "yolo-seg", "voc"):
            r = c.get(f"/api/projects/{project['id']}/export", params={"format": fmt})
            self.assertEqual(r.status_code, 200, fmt)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                self.assertTrue(any(n.endswith(".png") for n in z.namelist()), fmt)

        p = c.get(f"/api/projects/{project['id']}").json()
        self.assertEqual((p["image_count"], p["done_count"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
