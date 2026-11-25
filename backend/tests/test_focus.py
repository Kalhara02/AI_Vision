"""Tests for the focus-window geometry of SAM focus mode (no model needed)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from app.ml.sam import FOCUS_MAX_SIDE, FOCUS_MIN_SIDE, focus_window
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"dependencies missing: {exc}")


class FocusWindowTest(unittest.TestCase):
    def test_small_object_gets_min_window(self):
        x0, y0, x1, y1 = focus_window((100, 100, 110, 110), [(105, 105)], 512, 512)
        self.assertGreaterEqual(x1 - x0, FOCUS_MIN_SIDE)
        self.assertTrue(x0 <= 100 and x1 >= 110 and y0 <= 100 and y1 >= 110)

    def test_context_scales_with_object(self):
        x0, y0, x1, y1 = focus_window((200, 200, 260, 260), [], 1000, 1000)
        self.assertGreaterEqual(x1 - x0, 180)  # 3 x 60
        self.assertLessEqual(x1 - x0, FOCUS_MAX_SIDE + 32)

    def test_clamped_to_image(self):
        x0, y0, x1, y1 = focus_window((0, 0, 20, 20), [(5, 5)], 300, 400)
        self.assertEqual((x0, y0), (0, 0))
        self.assertLessEqual(x1, 400)
        self.assertLessEqual(y1, 300)

    def test_contains_far_click(self):
        x0, y0, x1, y1 = focus_window((100, 100, 120, 120), [(110, 110), (230, 110)], 512, 512)
        self.assertGreaterEqual(x1, 230 + 16)

    def test_grid_aligned(self):
        x0, y0, x1, y1 = focus_window((101, 133, 121, 150), [], 512, 512)
        self.assertEqual((x0 % 16, y0 % 16), (0, 0))


if __name__ == "__main__":
    unittest.main()


import numpy as np  # noqa: E402

from app.ml.sam import SamService  # noqa: E402


class FakePredictor:
    """Stands in for SAM: returns a disc of radius 10 (input pixels) around the first click or box centre."""

    def __init__(self):
        self.encoded = []

    def set_image(self, img):
        self.encoded.append(img.shape[:2])
        self._features, self._orig_hw, self._is_image_set = {"n": len(self.encoded)}, [img.shape[:2]], True

    def predict(self, point_coords=None, point_labels=None, box=None, mask_input=None, multimask_output=True):
        h, w = self._orig_hw[0]
        if box is not None:
            b = np.asarray(box).reshape(-1)
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        else:
            cx, cy = point_coords[0]
        yy, xx = np.mgrid[:h, :w]
        m = (((xx - cx) ** 2 + (yy - cy) ** 2) <= 100).astype(np.float32)
        n = 3 if multimask_output else 1
        return np.stack([m] * n), np.linspace(0.5, 0.9, n), np.zeros((n, 256, 256), np.float32)


class SegmentTest(unittest.TestCase):
    def setUp(self):
        self.s = SamService("fake")
        self.s._predictor = self.fake = FakePredictor()
        self.img = np.zeros((512, 512, 3), np.uint8)

    def centroid(self, mask):
        ys, xs = np.nonzero(mask)
        return xs.mean(), ys.mean()

    def test_plain_click(self):
        r = self.s.segment("a", self.img, [[300, 200, 1]])
        self.assertIsNone(r["window"])
        cx, cy = self.centroid(r["mask"])
        self.assertAlmostEqual(cx, 300, delta=0.5)
        self.assertAlmostEqual(cy, 200, delta=0.5)

    def test_focus_click_maps_back_to_full_image(self):
        r = self.s.segment("a", self.img, [[300, 200, 1]], focus=True)
        x0, y0, x1, y1 = r["window"]
        self.assertTrue(x0 <= 300 < x1 and y0 <= 200 < y1)
        cx, cy = self.centroid(r["mask"])
        self.assertAlmostEqual(cx, 300, delta=0.5)
        self.assertAlmostEqual(cy, 200, delta=0.5)
        self.assertEqual(r["mask"].shape, (512, 512))

    def test_focus_box(self):
        r = self.s.segment("a", self.img, [], box=[100, 120, 20, 20], focus=True)
        cx, cy = self.centroid(r["mask"])
        self.assertAlmostEqual(cx, 110, delta=0.5)
        self.assertAlmostEqual(cy, 130, delta=0.5)

    def test_second_click_reuses_window_encoding(self):
        r1 = self.s.segment("a", self.img, [[300, 200, 1]], focus=True)
        n = len(self.fake.encoded)
        r2 = self.s.segment("a", self.img, [[300, 200, 1], [305, 203, 0]], focus=True, prev=r1)
        self.assertEqual(r2["window"], r1["window"])
        self.assertEqual(len(self.fake.encoded), n)  # nothing re-encoded

    def test_large_object_falls_back_to_plain(self):
        class Big(FakePredictor):
            def predict(self, **kw):
                h, w = self._orig_hw[0]
                m = np.ones((h, w), np.float32)  # "object" fills the image
                n = 3 if kw.get("multimask_output") else 1
                return np.stack([m] * n), np.linspace(0.5, 0.9, n), np.zeros((n, 256, 256), np.float32)
        self.s._predictor = Big()
        r = self.s.segment("a", self.img, [[300, 200, 1]], focus=True)
        self.assertIsNone(r["window"])
