"""Apple detector: pretrained Faster R-CNN (torchvision) + our own fine-tuning loop.

Round 0 (cold start): the COCO-pretrained detector already knows the class
"apple"; we keep only those detections. Every "Retrain" fine-tunes a 2-class
head (background, apple) on the images the user has finished, with the
training loop written here (data, augmentation, optimiser, schedule, saving).

References: Ren et al. 2015 (Faster R-CNN); Li et al. 2021 (Faster R-CNN v2
recipe in torchvision); COCO (Lin et al. 2014) pretrained weights.
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


def build_pretrained(arch: str = DETECTOR_ARCH):
    """COCO-pretrained detector + the index of COCO's 'apple' class."""
    weights, builder = _weights_and_builder(arch)
    model = builder(weights=weights, box_score_thresh=0.05)
    apple_idx = weights.meta["categories"].index("apple")
    return model.eval(), apple_idx


def build_finetune(arch: str = DETECTOR_ARCH):
    """COCO-pretrained backbone + RPN, new 2-class box head (background, apple)."""
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    weights, builder = _weights_and_builder(arch)
    model = builder(weights=weights, trainable_backbone_layers=3, box_score_thresh=0.05,
                    box_detections_per_img=300)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def _to_tensor(img: np.ndarray):
    import torch
    return torch.from_numpy(img).permute(2, 0, 1).float().div(255.0)


# ---------------------------------------------------------------- inference
def detect(model, image: np.ndarray, label: int, threshold: float) -> list[dict]:
    """Run a detector on an RGB uint8 image; keep boxes of `label` above `threshold`."""
    import torch

    dev = next(model.parameters()).device
    with torch.inference_mode():
        out = model([_to_tensor(image).to(dev)])[0]
    res = []
    for b, s, l in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), out["labels"].cpu().numpy()):
        if int(l) == label and float(s) >= threshold:
            x0, y0, x1, y1 = (float(v) for v in b)
            res.append({"bbox": [x0, y0, x1 - x0, y1 - y0], "score": float(s)})
    return res


# ---------------------------------------------------------------- training (our own loop)
@dataclass
class Sample:
    path: Path
    boxes: list[list[float]]  # [x, y, w, h]


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
    """Horizontal flip + brightness/contrast jitter. Orchard light and leaf cover vary a lot."""
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


def train_detector(samples: list[Sample], cfg: TrainConfig = TrainConfig(), arch: str = DETECTOR_ARCH,
                   progress=None) -> tuple[object, dict]:
    """Fine-tune a 2-class Faster R-CNN on `samples`. Returns (model, metadata)."""
    import torch

    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    dev = device()
    model = build_finetune(arch).to(dev)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, (len(samples) + cfg.batch_size - 1) // cfg.batch_size)
    total = cfg.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(total * cfg.lr_drop_at)], gamma=0.1)

    images = [np.array(Image.open(s.path).convert("RGB")) for s in samples]
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
                b = b[((b[:, 2] - b[:, 0]) > 1) & ((b[:, 3] - b[:, 1]) > 1)]  # torchvision rejects empty boxes
                batch_imgs.append(_to_tensor(img).to(dev))
                targets.append({"boxes": torch.from_numpy(b).to(dev),
                                "labels": torch.ones(len(b), dtype=torch.int64, device=dev)})
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
    meta = {"arch": arch, "device": str(dev), "n_images": len(samples),
            "n_boxes": sum(len(s.boxes) for s in samples), "epochs": cfg.epochs,
            "loss_per_epoch": history, "train_seconds": round(time.time() - t0, 1)}
    return model, meta


# ---------------------------------------------------------------- per-project service
def evaluate_ap50(model, label: int, samples: list[Sample]) -> float:
    """AP at IoU 0.5 with our own metric (app/ml/metrics.py) on held-out samples."""
    from .metrics import ap_at_iou

    dets = [detect(model, np.array(Image.open(s.path).convert("RGB")), label, 0.05) for s in samples]
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


@dataclass
class TrainState:
    state: str = "idle"  # idle | training | scoring | error
    progress: float = 0.0
    message: str = ""
    started: float | None = None


@dataclass
class _Loaded:
    version: int
    model: object
    label: int


class DetectorService:
    """Newest accepted model per project in memory; trains, guards and ranks in a background thread.

    Round flow: train on ~80% of finished images -> AP50 of new and current model on
    the same ~20% hold-out -> keep the new model only if it is not worse -> score all
    unfinished images by uncertainty for the active-learning queue.
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

    def _get(self, pid: int) -> _Loaded:
        import torch

        want = self.latest_version(pid)
        cur = self._models.get(pid)
        if cur is not None and cur.version == want:
            return cur
        if want == 0:
            model, apple = build_pretrained()
            loaded = _Loaded(0, model.to(device()), apple)
        else:
            model = build_finetune()
            state = torch.load(self.project_dir(pid) / f"detector_v{want}.pt", map_location="cpu")
            model.load_state_dict(state)
            loaded = _Loaded(want, model.to(device()).eval(), 1)
        self._models[pid] = loaded
        return loaded

    def predict(self, pid: int, path: Path, threshold: float) -> dict:
        image = np.array(Image.open(path).convert("RGB"))
        with self._lock:
            m = self._get(pid)
            t0 = time.perf_counter()
            dets = detect(m.model, image, m.label, threshold)
        return {"version": m.version, "proposals": dets, "ms": round((time.perf_counter() - t0) * 1000, 1)}

    # ---------- active learning ranking
    def scores(self, pid: int) -> dict:
        f = self.project_dir(pid) / "uncertainty.json"
        return json.loads(f.read_text()) if f.exists() else {"version": None, "scores": {}}

    def _score(self, pid: int, items: list[tuple[int, Path]], st: TrainState) -> None:
        from .metrics import image_uncertainty

        st.state, st.progress = "scoring", 0.0
        out = {}
        for k, (image_id, path) in enumerate(items):
            image = np.array(Image.open(path).convert("RGB"))
            with self._lock:
                m = self._get(pid)
                dets = detect(m.model, image, m.label, 0.05)
            out[str(image_id)] = round(image_uncertainty([d["score"] for d in dets]), 4)
            st.progress, st.message = (k + 1) / len(items), f"ranking images {k + 1}/{len(items)}"
        (self.project_dir(pid) / "uncertainty.json").write_text(
            json.dumps({"version": self.latest_version(pid), "scores": out}))

    def start_ranking(self, pid: int, items: list[tuple[int, Path]]) -> bool:
        if self.is_busy(pid) or not items:
            return False
        st = self._train[pid] = TrainState(state="scoring", message="ranking", started=time.time())

        def run():
            try:
                self._score(pid, items, st)
                st.state, st.progress, st.message = "idle", 1.0, "ranking ready"
            except Exception as exc:
                st.state, st.message = "error", f"{type(exc).__name__}: {exc}"

        threading.Thread(target=run, daemon=True).start()
        return True

    # ---------- training rounds
    def status(self, pid: int) -> dict:
        st = self._train.get(pid, TrainState())
        sc = self.scores(pid)
        return {"version": self.latest_version(pid), "state": st.state, "progress": round(st.progress, 3),
                "message": st.message, "history": self.versions(pid), "rounds": self._rounds(pid),
                "ranked_version": sc["version"], "n_ranked": len(sc["scores"])}

    def start_training(self, pid: int, samples: list[Sample], cfg: TrainConfig, trigger: str,
                       to_rank: list[tuple[int, Path]] | None = None) -> bool:
        if self.is_busy(pid):
            return False
        self._train[pid] = TrainState(state="training", message="starting", started=time.time())
        threading.Thread(target=self._run, args=(pid, samples, cfg, trigger, to_rank or []), daemon=True).start()
        return True

    def _run(self, pid: int, samples: list[Sample], cfg: TrainConfig, trigger: str, to_rank):
        import torch

        st = self._train[pid]

        def progress(frac: float, msg: str):
            st.progress, st.message = frac, msg

        try:
            train_s, hold_s = split_holdout(samples)
            model, meta = train_detector(train_s, cfg, progress=progress)
            info = {"round": len(self._rounds(pid)) + 1, "trigger": trigger, "n_finished": len(samples),
                    "n_train": len(train_s), "n_holdout": len(hold_s),
                    "created": time.strftime("%Y-%m-%d %H:%M:%S"), "train_seconds": meta["train_seconds"]}
            accept = True
            if hold_s:
                st.message = "comparing with the current model on held-out images"
                with self._lock:
                    cur = self._get(pid)
                    ap_old = evaluate_ap50(cur.model, cur.label, hold_s)
                    ap_new = evaluate_ap50(model, 1, hold_s)
                info.update({"previous_version": cur.version, "ap50_previous": round(ap_old, 4),
                             "ap50_new": round(ap_new, 4)})
                accept = ap_new >= ap_old - 0.005  # guard: never replace a model with a worse one
            info["accepted"] = accept
            if accept:
                version = self.latest_version(pid) + 1
                d = self.project_dir(pid)
                torch.save(model.state_dict(), d / f"detector_v{version}.pt")
                meta.update({"version": version, "created": info["created"],
                             "ap50_holdout": info.get("ap50_new"), "n_holdout": len(hold_s)})
                (d / f"detector_v{version}.json").write_text(json.dumps(meta, indent=1))
                with self._lock:
                    self._models[pid] = _Loaded(version, model, 1)
                info["version"] = version
                msg = f"v{version} ready"
            else:
                msg = (f"kept v{info['previous_version']}: new model scored AP50 {info['ap50_new']:.2f} "
                       f"vs {info['ap50_previous']:.2f} on held-out images")
            self._add_round(pid, info)
            if accept and to_rank:
                self._score(pid, to_rank, st)
            st.state, st.progress, st.message = "idle", 1.0, msg
        except Exception as exc:  # surface the reason in the UI instead of a silent thread death
            st.state, st.message = "error", f"{type(exc).__name__}: {exc}"


detector_service = DetectorService()
