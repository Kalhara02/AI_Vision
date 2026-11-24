"""Tests for mask <-> polygon conversion used by SAM and the evaluation."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import cv2  # noqa: F401
    from app.ml.masks import largest_component, mask_iou, mask_to_polygon, polygon_to_mask
except ImportError as exc:  # pragma: no cover
    raise unittest.SkipTest(f"opencv missing: {exc}")


def disc(h, w, cx, cy, r):
    yy, xx = np.mgrid[:h, :w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


class MaskTest(unittest.TestCase):
    def test_roundtrip_circle(self):
        m = disc(100, 100, 50, 50, 20)
        poly = mask_to_polygon(m)
        self.assertIsNotNone(poly)
        self.assertGreater(mask_iou(m, polygon_to_mask(poly, 100, 100)), 0.9)

    def test_keeps_largest_component(self):
        m = disc(100, 100, 30, 30, 15) | disc(100, 100, 80, 80, 3)
        self.assertFalse(largest_component(m)[80, 80])
        self.assertTrue(largest_component(m)[30, 30])

    def test_empty_mask(self):
        self.assertIsNone(mask_to_polygon(np.zeros((20, 20), bool)))

    def test_iou(self):
        a = np.zeros((10, 10), bool); a[:, :5] = True
        b = np.zeros((10, 10), bool); b[:, 5:] = True
        self.assertEqual(mask_iou(a, b), 0.0)
        self.assertEqual(mask_iou(a, a), 1.0)


if __name__ == "__main__":
    unittest.main()
