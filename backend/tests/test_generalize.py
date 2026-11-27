"""Label-set independence: class-aware mAP, label mapping, zero-shot decoding. No PyTorch needed."""
from pathlib import Path

import numpy as np
import pytest

from app.ml.detector import MIN_TRAINED_INSTANCES, DetectorService, Sample, model_labels
from app.ml.metrics import map_at_iou
from app.ml.zeroshot import calibrate, decode, nms, prompt_for


class TestClassAwareMap:
    def test_wrong_class_never_matches(self):
        gt = [[([0, 0, 10, 10], 7)]]
        assert map_at_iou([[{"bbox": [0, 0, 10, 10], "score": 0.9, "class_id": 8}]], gt) == 0.0
        assert map_at_iou([[{"bbox": [0, 0, 10, 10], "score": 0.9, "class_id": 7}]], gt) == pytest.approx(1.0)

    def test_mean_over_classes(self):
        gt = [[([0, 0, 10, 10], 1), ([20, 20, 10, 10], 2)]]
        dets = [[{"bbox": [0, 0, 10, 10], "score": 0.9, "class_id": 1}]]  # class 2 missed entirely
        assert map_at_iou(dets, gt) == pytest.approx(0.5)

    def test_no_ground_truth(self):
        assert map_at_iou([[]], [[]]) == 0.0


class TestLabelMapping:
    def test_project_ids_to_contiguous_labels(self):
        s = Sample(Path("a.png"), [[0, 0, 1, 1]] * 3, [42, 17, 42])
        assert model_labels(s, [17, 42]) == [2, 1, 2]

    def test_single_class_data(self):
        s = Sample(Path("a.png"), [[0, 0, 1, 1]] * 2)
        assert model_labels(s, None) == [1, 1]

    def test_weak_classes_keep_zero_shot(self):
        meta = {"boxes_per_class": {"1": MIN_TRAINED_INSTANCES, "2": MIN_TRAINED_INSTANCES - 1}}
        assert DetectorService._weak(meta, [1, 2, 3]) == {2, 3}

    def test_old_single_class_models_are_trusted(self):
        assert DetectorService._weak({}, [5]) == set()


class TestZeroShotDecoding:
    def test_prompt(self):
        assert prompt_for("traffic_light") == "a photo of a traffic light"
        assert prompt_for(" Tumor ") == "a photo of a tumor"

    def test_calibration_keeps_order_and_range(self):
        assert calibrate(0.15, 0.15) == pytest.approx(0.5)
        assert calibrate(0.6, 0.15) == 1.0
        assert calibrate(0.05, 0.15) < calibrate(0.1, 0.15)

    def test_owlv2_boxes_use_the_padded_square(self):
        # a box in the middle of the square: on a 200 x 100 image OWLv2 normalises by 200 both ways
        logits = np.array([[5.0, -5.0]])
        boxes = np.array([[0.5, 0.25, 0.1, 0.1]])
        d = decode(logits, boxes, 200, 100, padded_square=True)[0]
        assert d["query"] == 0
        assert d["bbox"] == pytest.approx([90, 40, 20, 20])

    def test_owlvit_boxes_use_the_image(self):
        d = decode(np.array([[-5.0, 5.0]]), np.array([[0.5, 0.5, 0.1, 0.2]]), 200, 100, padded_square=False)[0]
        assert d["query"] == 1
        assert d["bbox"] == pytest.approx([90, 40, 20, 20])

    def test_json_safe_types(self):
        # float32 model outputs must come back as plain floats, or the API cannot encode them
        d = decode(np.array([[5.0]], np.float32), np.array([[0.5, 0.5, 0.1, 0.1]], np.float32), 100, 100, True)[0]
        assert all(type(v) is float for v in d["bbox"]) and type(d["raw"]) is float and type(d["query"]) is int

    def test_low_scores_dropped(self):
        assert decode(np.array([[-10.0]]), np.array([[0.5, 0.5, 0.1, 0.1]]), 100, 100, True) == []

    def test_nms_per_class(self):
        a = {"bbox": [0, 0, 10, 10], "query": 0, "raw": 0.9}
        b = {"bbox": [1, 1, 10, 10], "query": 0, "raw": 0.5}  # same object, same class: removed
        c = {"bbox": [1, 1, 10, 10], "query": 1, "raw": 0.5}  # other class: kept
        assert nms([b, a, c]) == [a, c]
