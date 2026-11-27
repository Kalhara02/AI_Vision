"""Detection metrics and uncertainty, written for this project (NumPy only, unit tested).

ap_at_iou: average precision at one IoU threshold, all-point interpolation
(the PASCAL VOC 2010+ definition). Used by the in-app model guard; the offline
evaluation additionally reports COCO mAP via pycocotools.

image_uncertainty: sum of binary entropies of the detector's scores on one
image. Boxes near 0.5 contribute most, confident boxes (near 0 or 1) almost
nothing, so images with many "unsure" apples are ranked first (entropy-based
uncertainty sampling, Settles 2009).
"""
from __future__ import annotations

import math

import numpy as np


def box_iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (N, 4) and b (M, 4), both [x, y, w, h]."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ax1, ay1 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax1[:, None], bx1[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay1[:, None], by1[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def ap_at_iou(detections: list[list[dict]], ground_truth: list[list[list[float]]], iou_thr: float = 0.5) -> float:
    """detections[i] = [{"bbox": [x,y,w,h], "score": s}, ...] for image i; ground_truth[i] = list of boxes."""
    n_gt = sum(len(g) for g in ground_truth)
    if n_gt == 0:
        return 0.0
    flat = [(d["score"], i, d["bbox"]) for i, ds in enumerate(detections) for d in ds]
    flat.sort(key=lambda t: -t[0])
    used = [np.zeros(len(g), bool) for g in ground_truth]
    tp = np.zeros(len(flat))
    for k, (_, i, box) in enumerate(flat):
        g = ground_truth[i]
        if not len(g):
            continue
        ious = box_iou_xywh(np.array([box]), np.array(g))[0]
        ious[used[i]] = -1  # each ground-truth box can be matched once
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            tp[k] = 1
            used[i][j] = True
    if not len(flat):
        return 0.0
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(flat) + 1)
    # all-point interpolation: make precision monotonically decreasing, integrate over recall
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def binary_entropy(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def image_uncertainty(scores: list[float], min_score: float = 0.05) -> float:
    return float(sum(binary_entropy(s) for s in scores if s >= min_score))


def map_at_iou(detections: list[list[dict]], ground_truth: list[list[tuple[list[float], int]]],
               iou_thr: float = 0.5) -> float:
    """Class-aware mean AP: detections[i] = [{"bbox", "score", "class_id"}], ground_truth[i] = [(box, class_id)].

    AP is computed per class that has ground truth (ap_at_iou on that class only) and averaged, the
    PASCAL VOC / COCO convention. A detection of the wrong class never matches.
    """
    classes = sorted({c for g in ground_truth for _, c in g})
    if not classes:
        return 0.0
    aps = []
    for c in classes:
        dets = [[d for d in ds if d.get("class_id") == c] for ds in detections]
        gts = [[b for b, k in g if k == c] for g in ground_truth]
        aps.append(ap_at_iou(dets, gts, iou_thr))
    return float(np.mean(aps))
