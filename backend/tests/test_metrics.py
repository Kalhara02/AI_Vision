"""Tests for our own detection metrics and uncertainty score."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.metrics import ap_at_iou, box_iou_xywh, image_uncertainty  # noqa: E402

G = [[0, 0, 10, 10], [20, 20, 10, 10]]


class MetricsTest(unittest.TestCase):
    def test_perfect_detections(self):
        dets = [[{"bbox": g, "score": 0.9} for g in G]]
        self.assertAlmostEqual(ap_at_iou(dets, [G]), 1.0)

    def test_half_recall(self):
        self.assertAlmostEqual(ap_at_iou([[{"bbox": G[0], "score": 0.9}]], [G]), 0.5)

    def test_false_positive_ranked_first_lowers_ap(self):
        dets = [[{"bbox": [50, 50, 5, 5], "score": 0.99}] + [{"bbox": g, "score": 0.9} for g in G]]
        ap = ap_at_iou(dets, [G])
        self.assertAlmostEqual(ap, 2 / 3, places=5)  # precision 0.5 at r=0.5, 2/3 at r=1, interpolated

    def test_duplicate_is_false_positive(self):
        dets = [[{"bbox": G[0], "score": 0.9}, {"bbox": G[0], "score": 0.8}]]
        self.assertAlmostEqual(ap_at_iou(dets, [[G[0]]]), 1.0)  # duplicate after full recall costs nothing
        self.assertAlmostEqual(ap_at_iou(dets, [G]), 0.5)

    def test_no_detections(self):
        self.assertEqual(ap_at_iou([[]], [G]), 0.0)

    def test_iou_matrix(self):
        m = box_iou_xywh([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 0, 10, 10]])
        self.assertAlmostEqual(m[0, 0], 1.0)
        self.assertAlmostEqual(m[0, 1], 50 / 150)

    def test_uncertainty_prefers_unsure_boxes(self):
        self.assertGreater(image_uncertainty([0.5, 0.5]), image_uncertainty([0.99, 0.98, 0.97]))
        self.assertEqual(image_uncertainty([0.01]), 0.0)


if __name__ == "__main__":
    unittest.main()
