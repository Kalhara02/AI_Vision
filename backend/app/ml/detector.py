"""Suggestions for any user-defined classes: zero-shot first, then our own fine-tuning loop.

Round 0 (cold start): nothing is labeled yet, so suggestions come from OWLv2 prompted with the
project's class names (app/ml/zeroshot.py). This works for any image domain and any label set.

Every training round fine-tunes a Faster R-CNN (COCO-pretrained backbone and RPN, a new box head
with one output per project class + background) on the images the user has finished, with the
training loop written here (data, augmentation, optimiser, schedule, saving). A guard compares the
new model with what the user currently gets on held-out finished images (class-aware mAP50) and
deploys it only if it is not worse. Round 1 is therefore judged against zero-shot.

Classes the trained model has not seen enough of (fewer than MIN_TRAINED_INSTANCES boxes, or added
after the last round) keep getting zero-shot suggestions, so a new label is never left without help.

References: Ren et al. 2015 (Faster R-CNN); Li et al. 2021 (Faster R-CNN v2 recipe in torchvision);
COCO (Lin et al. 2014) pretrained weights; Minderer et al. 2023 (OWLv2).
"""
from __future__ import annotations

import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import DETECTOR_ARCH, MODEL_DIR

MIN_TRAINED_INSTANCES = 10  # below this many training boxes a class also keeps zero-shot suggestions
LOW = 0.05  # lowest score kept for ranking, AP and the threshold slider


# ---------------------------------------------------------------- model building
def _weights_and_builder(arch: str):
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        FasterRCNN_ResNet50_FPN_V2_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
        fasterrcnn_resnet50_fpn_v2,
    )
    if arch == "mobilenet_v3":
        return FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1, fasterrcnn_mobilenet_v3_large_fpn
    return FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, fasterrcnn_resnet50_fpn_v2


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_pretrained(arch: str = DETECTOR_ARCH, coco_class: str = "apple"):
    """The plain COCO detector + the index of one COCO class. Only used by the offline MinneApple
    evaluation (eval/detector_curve.py, n = 0); the app's cold start is zero-shot."""
    weights, builder = _weights_and_builder(arch)
    model = builder(weights=weights, box_score_thresh=LOW)
    return model.eval(), weights.meta["categories"].index(coco_class)


def build_finetune(arch: str = DETECTOR_ARCH, num_classes: int = 2):
    """COCO-pretrained backbone + RPN, new box head with `num_classes` outputs (background included)."""
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights, builder = _weights_and_builder(arch)
    model = builder(weights=weights, trainable_backbone_layers=3, box_score_thresh=LOW,
                    box_detections_per_img=300)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _to_tensor(img: np.ndarray):
    import torch
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)


# ---------------------------------------------------------------- inference
def detect(model, image: np.ndarray, label: int | None, threshold: float) -> list[dict]:
    """Run a Faster R-CNN on an RGB uint8 image. label=None keeps every class (each result then
    carries its model label 1..K); a number keeps only that label (the single-class evaluation)."""
    import torch

    dev = next(model.parameters()).device
    with torch.inference_mode():
        out = model([_to_tensor(image).to(dev)])[0]
    res = []
    for b, s, l in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()):
        if (label is None or int(l) == label) and float(s) >= threshold:
            x0, y0, x1, y1 = (float(v) for v in b)
            res.append({"bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(s), "label": int(l)})
    return res


# ---------------------------------------------------------------- training (our own loop)
@dataclass
class Sample:
    path: Path
    boxes: list[list[float]]  # [x, y, w, h]
    labels: list[int] | None = None  # project class id per box; None = single-class data


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 2
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 1e-4
    lr_drop_at: float = 0.7  # fraction of training after which lr is divided by 10
    seed: int = 0


def _augment(img: np.ndarray, boxes: np.ndarray, rng: random.Random):
    """Horizontal flip + brightness/contrast jitter (domain-agnostic: no colour or shape assumptions)."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
        if len(boxes):
            boxes = boxes.copy()
            x0 = w - boxes[:, 2]
            x1 = w - boxes[:, 0]
            boxes[:, 0], boxes[:, 2] = x0, x1
    alpha = rng.uniform(0.75, 1.25)  # contrast
    beta = rng.uniform(-25, 25)  # brightness
    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return img, boxes


def model_labels(sample: Sample, class_ids: list[int] | None) -> list[int]:
    """Project class ids -> contiguous model labels 1..K (0 is background). None = single class."""
    if class_ids is None or sample.labels is None:
        return [1] * len(sample.boxes)
    return [class_ids.index(c) + 1 for c in sample.labels]


def train_detector(samples: list[Sample], cfg: TrainConfig = TrainConfig(), arch: str = DETECTOR_ARCH,
                   progress=None, class_ids: list[int] | None = None) -> tuple[object, dict]:
    """Fine-tune Faster R-CNN on `samples`. class_ids fixes the label order (None = one class)."""
    import torch

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    dev = device()
    num_classes = (len(class_ids) if class_ids else 1) + 1
    model = build_finetune(arch, num_classes).to(dev)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, (len(samples) + cfg.batch_size - 1) // cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(total * cfg.lr_drop_at)], gamma=0.1)

    images = [np.array(Image.open(s.path).convert("RGB")) for s in samples]
    labels = [np.array(model_labels(s, class_ids), np.int64) for s in samples]
    t0, step, history = time.time(), 0, []
    for epoch in range(cfg.epochs):
        order = list(range(len(samples)))
        rng.shuffle(order)
        epoch_loss = 0.0
        for i in range(0, len(order), cfg.batch_size):
            batch_imgs, targets = [], []
            for j in order[i:i + cfg.batch_size]:
                b = np.array([[x, y, x + w, y + h] for x, y, w, h in samples[j].boxes], np.float32).reshape(-1, 4)
                img, b = _augment(images[j], b, rng)
                ok = ((b[:, 2] - b[:, 0]) > 1) & ((b[:, 3] - b[:, 1]) > 1)  # torchvision rejects empty boxes
                batch_imgs.append(_to_tensor(img).to(dev))
                targets.append({"boxes": torch.from_numpy(b[ok]).to(dev),
                                "labels": torch.from_numpy(labels[j][ok]).to(dev)})
            losses = model(batch_imgs, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss became {loss.item()} at step {step}; lower the learning rate")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            step += 1
            epoch_loss += float(loss.item())
            if progress:
                progress(step / total, f"epoch {epoch + 1}/{cfg.epochs}")
        history.append(round(epoch_loss / steps_per_epoch, 4))
    model.eval()
    counts: dict[str, int] = {}
    for s in samples:
        for c in (s.labels if s.labels is not None else [0] * len(s.boxes)):
            counts[str(c)] = counts.get(str(c), 0) + 1
    meta = {"arch": arch, "device": str(dev), "n_images": len(samples),
            "n_boxes": sum(len(s.boxes) for s in samples), "epochs": cfg.epochs,
            "class_ids": class_ids, "boxes_per_class": counts,
            "loss_per_epoch": history, "train_seconds": round(time.time() - t0, 1)}
    return model, meta


def evaluate_ap50(model, label: int, samples: list[Sample]) -> float:
    """Single-class AP50 with our own metric (used by the offline single-class evaluation)."""
    from .metrics import ap_at_iou

    dets = [detect(model, np.array(Image.open(s.path).convert("RGB")), label, LOW) for s in samples]
    return ap_at_iou(dets, [s.boxes for s in samples], 0.5)


def split_holdout(samples: list[Sample], every: int = 5) -> tuple[list[Sample], list[Sample]]:
    """Deterministic ~20% hold-out (by file name), so the same images judge every round."""
    import zlib

    if len(samples) < 10:
        return samples, []
    hold = [s for s in samples if zlib.crc32(s.path.name.encode()) % every == 0]
    train = [s for s in samples if zlib.crc32(s.path.name.encode()) % every != 0]
    if not hold or len(train) < 5:
        return samples, []
    return train, hold


# ---------------------------------------------------------------- per-project service
@dataclass
class TrainState:
    state: str = "idle"  # idle | training | scoring | error
    progress: float = 0.0
    message: str = ""
    started: float | None = None


@dataclass
class _Loaded:
    version: int
    model: object | None  # None = no trained model yet (zero-shot only)
    class_ids: list[int] = field(default_factory=list)  # model label k -> class_ids[k - 1]
    weak: set[int] = field(default_factory=set)  # trained classes that also keep zero-shot help


Classes = list[tuple[int, str]]  # (class id, name) of a project, in id order


class DetectorService:
    """What the user is suggested, per project: trained model where it knows a class, zero-shot
    elsewhere. Trains, guards and ranks in a background thread.

    Round flow: train on ~80% of finished images -> class-aware mAP50 of new and current
    suggestions on the same ~20% hold-out -> keep the new model only if it is not worse ->
    score all unfinished images by uncertainty for the active-learning queue.
    """

    def __init__(self):
        self._models: dict[int, _Loaded] = {}
        self._lock = threading.Lock()  # guards inference + model swap
        self._train: dict[int, TrainState] = {}

    @staticmethod
    def project_dir(pid: int) -> Path:
        d = MODEL_DIR / f"project_{pid}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _rounds(self, pid: int) -> list[dict]:
        f = self.project_dir(pid) / "rounds.json"
        return json.loads(f.read_text()) if f.exists() else []

    def _add_round(self, pid: int, info: dict) -> None:
        rounds = self._rounds(pid) + [info]
        (self.project_dir(pid) / "rounds.json").write_text(json.dumps(rounds, indent=1))

    def versions(self, pid: int) -> list[dict]:
        out = []
        for f in sorted(self.project_dir(pid).glob("detector_v*.json"), key=lambda p: int(p.stem.split("_v")[1])):
            out.append(json.loads(f.read_text()))
        return out

    def latest_version(self, pid: int) -> int:
        v = self.versions(pid)
        return v[-1]["version"] if v else 0

    def images_seen(self, pid: int) -> int:
        """Finished images available at the last training round (accepted or not)."""
        r = self._rounds(pid)
        return r[-1]["n_finished"] if r else 0

    def is_busy(self, pid: int) -> bool:
        st = self._train.get(pid)
        return bool(st and st.state in ("training", "scoring"))

    @staticmethod
    def _weak(meta: dict, class_ids: list[int]) -> set[int]:
        counts = meta.get("boxes_per_class") or {}
        if not counts:  # models saved before multi-class support: trust them for their class
            return set()
        return {c for c in class_ids if counts.get(str(c), 0) < MIN_TRAINED_INSTANCES}

    def _get(self, pid: int, classes: Classes) -> _Loaded:
        import torch

        want = self.latest_version(pid)
        cur = self._models.get(pid)
        if cur is not None and cur.version == want:
            return cur
        if want == 0:
            loaded = _Loaded(0, None)
        else:
            meta = json.loads((self.project_dir(pid) / f"detector_v{want}.json").read_text())
            # single-class models saved before multi-class support belong to the project's first class
            class_ids = meta.get("class_ids") or [classes[0][0]]
            model = build_finetune(num_classes=len(class_ids) + 1)
            model.load_state_dict(torch.load(self.project_dir(pid) / f"detector_v{want}.pt", map_location="cpu"))
            loaded = _Loaded(want, model.to(device()).eval(), class_ids, self._weak(meta, class_ids))
        self._models[pid] = loaded
        return loaded

    @staticmethod
    def _suggest(m: _Loaded, key: str, image: np.ndarray, classes: Classes) -> tuple[list[dict], list[str]]:
        """All suggestions at score >= LOW as [{bbox, score, class_id, source}] + the zero-shot classes."""
        dets: list[dict] = []
        valid = {cid for cid, _ in classes}
        if m.model is not None:
            for d in detect(m.model, image, None, LOW):
                cid = m.class_ids[d["label"] - 1] if 0 < d["label"] <= len(m.class_ids) else None
                if cid in valid:
                    dets.append({"bbox": d["bbox"], "score": d["score"], "class_id": cid, "source": "trained"})
        zs = [(cid, name) for cid, name in classes if cid not in m.class_ids or cid in m.weak]
        if zs:
            from .zeroshot import ZeroShotUnavailable, calibrate, zero_shot_service
            try:
                raw, _ = zero_shot_service.detect(key, image, [n for _, n in zs])
            except ZeroShotUnavailable:
                raw = []
            for d in raw:
                s = calibrate(d["raw"])
                if s >= LOW:
                    dets.append({"bbox": d["bbox"], "score": s, "class_id": zs[d["query"]][0], "source": "zero-shot"})
        return dets, [n for _, n in zs]

    def predict(self, pid: int, path: Path, threshold: float, classes: Classes) -> dict:
        if not classes:
            return {"version": 0, "proposals": [], "zero_shot_classes": [], "ms": 0.0}
        image = np.array(Image.open(path).convert("RGB"))
        t0 = time.perf_counter()
        with self._lock:
            m = self._get(pid, classes)
            dets, zs = self._suggest(m, str(path), image, classes)
        props = [d for d in dets if d["score"] >= threshold]
        return {"version": m.version, "proposals": props, "zero_shot_classes": zs,
                "ms": round((time.perf_counter() - t0) * 1000, 1)}

    # ---------- active learning ranking
    def scores(self, pid: int) -> dict:
        f = self.project_dir(pid) / "uncertainty.json"
        return json.loads(f.read_text()) if f.exists() else {"version": None, "scores": {}}

    def _score(self, pid: int, items: list[tuple[int, Path]], classes: Classes, st: TrainState) -> None:
        from .metrics import image_uncertainty

        st.state, st.progress = "scoring", 0.0
        out = {}
        for k, (image_id, path) in enumerate(items):
            image = np.array(Image.open(path).convert("RGB"))
            with self._lock:
                dets, _ = self._suggest(self._get(pid, classes), str(path), image, classes)
            out[str(image_id)] = round(image_uncertainty([d["score"] for d in dets]), 4)
            st.progress, st.message = (k + 1) / len(items), f"ranking images {k + 1}/{len(items)}"
        (self.project_dir(pid) / "uncertainty.json").write_text(
            json.dumps({"version": self.latest_version(pid), "scores": out}))

    def start_ranking(self, pid: int, items: list[tuple[int, Path]], classes: Classes) -> bool:
        if self.is_busy(pid) or not items or not classes:
            return False
        st = self._train[pid] = TrainState(state="scoring", message="ranking", started=time.time())

        def run():
            try:
                self._score(pid, items, classes, st)
                st.state, st.progress, st.message = "idle", 1.0, "ranking ready"
            except Exception as exc:
                st.state, st.message = "error", f"{type(exc).__name__}: {exc}"

        threading.Thread(target=run, daemon=True).start()
        return True

    # ---------- training rounds
    def status(self, pid: int) -> dict:
        from ..config import ZERO_SHOT_MODEL

        st = self._train.get(pid, TrainState())
        sc = self.scores(pid)
        return {"version": self.latest_version(pid), "state": st.state, "progress": round(st.progress, 3),
                "message": st.message, "history": self.versions(pid), "rounds": self._rounds(pid),
                "ranked_version": sc["version"], "n_ranked": len(sc["scores"]), "zero_shot_model": ZERO_SHOT_MODEL}

    def start_training(self, pid: int, samples: list[Sample], cfg: TrainConfig, trigger: str,
                       classes: Classes, to_rank: list[tuple[int, Path]] | None = None) -> bool:
        if self.is_busy(pid) or not classes:
            return False
        self._train[pid] = TrainState(state="training", message="starting", started=time.time())
        threading.Thread(target=self._run, args=(pid, samples, cfg, trigger, classes, to_rank or []),
                         daemon=True).start()
        return True

    def _map50(self, m: _Loaded, hold: list[Sample], classes: Classes) -> float:
        from .metrics import map_at_iou

        dets = [self._suggest(m, str(s.path), np.array(Image.open(s.path).convert("RGB")), classes)[0] for s in hold]
        gts = [list(zip(s.boxes, s.labels or [classes[0][0]] * len(s.boxes))) for s in hold]
        return map_at_iou(dets, gts, 0.5)

    def _run(self, pid: int, samples: list[Sample], cfg: TrainConfig, trigger: str, classes: Classes, to_rank):
        import torch

        st = self._train[pid]

        def progress(frac: float, msg: str):
            st.progress, st.message = frac, msg

        try:
            class_ids = [cid for cid, _ in classes]
            train_s, hold_s = split_holdout(samples)
            model, meta = train_detector(train_s, cfg, progress=progress, class_ids=class_ids)
            info = {"round": len(self._rounds(pid)) + 1, "trigger": trigger, "n_finished": len(samples),
                    "n_train": len(train_s), "n_holdout": len(hold_s), "n_classes": len(class_ids),
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "train_seconds": meta["train_seconds"]}
            new = _Loaded(-1, model, class_ids, self._weak(meta, class_ids))
            accept = True
            if hold_s:
                st.message = "comparing with the current suggestions on held-out images"
                with self._lock:
                    cur = self._get(pid, classes)
                    ap_old = self._map50(cur, hold_s, classes)
                    ap_new = self._map50(new, hold_s, classes)
                info.update({"previous_version": cur.version, "ap50_previous": round(ap_old, 4),
                             "ap50_new": round(ap_new, 4)})
                accept = ap_new >= ap_old - 0.005  # guard: never replace suggestions with worse ones
            info["accepted"] = accept
            if accept:
                version = self.latest_version(pid) + 1
                d = self.project_dir(pid)
                torch.save(model.state_dict(), d / f"detector_v{version}.pt")
                meta.update({"version": version, "created": info["created"],
                             "ap50_holdout": info.get("ap50_new"), "n_holdout": len(hold_s),
                             "class_names": [n for _, n in classes]})
                (d / f"detector_v{version}.json").write_text(json.dumps(meta, indent=1))
                new.version = version
                with self._lock:
                    self._models[pid] = new
                info["version"] = version
                msg = f"v{version} ready"
            else:
                prev = f"v{info['previous_version']}" if info["previous_version"] else "zero-shot"
                msg = (f"kept {prev}: new model scored mAP50 {info['ap50_new']:.2f} "
                       f"vs {info['ap50_previous']:.2f} on held-out images")
            self._add_round(pid, info)
            if accept and to_rank:
                self._score(pid, to_rank, classes, st)
            st.state, st.progress, st.message = "idle", 1.0, msg
        except Exception as exc:  # surface the reason in the UI instead of a silent thread death
            st.state, st.message = "error", f"{type(exc).__name__}: {exc}"


detector_service = DetectorService()
