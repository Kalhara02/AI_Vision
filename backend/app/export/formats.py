"""Exporters for standard annotation formats.

Every exporter takes the same plain data (no database objects), so they are
easy to unit test and reuse from the evaluation scripts.

Coordinate conventions:
- Internal: pixel coordinates, origin top-left, boxes as [x, y, w, h] (floats).
- COCO: same as internal (bbox [x, y, w, h], polygons as flat [x1, y1, x2, y2, ...]).
- YOLO: class index + normalised centre/size in [0, 1]; YOLO-seg uses normalised polygon points.
- Pascal VOC: integer, 1-based, inclusive corner coordinates (xmin, ymin, xmax, ymax).
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone

from ..geometry import polygon_area


@dataclass
class ExpImage:
    id: int
    file_name: str  # name inside the export archive
    width: int
    height: int


@dataclass
class ExpAnnotation:
    image_id: int
    category: int  # 0-based index into the category list
    kind: str  # "bbox" | "polygon"
    bbox: list[float]
    points: list[list[float]] | None = None
    score: float | None = None
    source: str = "manual"


def _r(v: float, nd: int = 2) -> float:
    return round(float(v), nd)


# ---------------------------------------------------------------- COCO
def to_coco(images: list[ExpImage], anns: list[ExpAnnotation], categories: list[str],
            description: str = "") -> dict:
    out = {
        "info": {
            "description": description or "Exported from LabelLoop",
            "version": "1.0",
            "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "licenses": [],
        "categories": [{"id": i + 1, "name": n, "supercategory": "object"} for i, n in enumerate(categories)],
        "images": [{"id": im.id, "file_name": im.file_name, "width": im.width, "height": im.height}
                   for im in images],
        "annotations": [],
    }
    for k, a in enumerate(anns, start=1):
        if a.kind == "polygon" and a.points:
            seg = [[_r(c) for p in a.points for c in p]]
            area = polygon_area(a.points)
        else:
            seg = []
            area = a.bbox[2] * a.bbox[3]
        out["annotations"].append({
            "id": k,
            "image_id": a.image_id,
            "category_id": a.category + 1,
            "bbox": [_r(v) for v in a.bbox],
            "area": _r(area),
            "segmentation": seg,
            "iscrowd": 0,
            # non-standard but harmless: lets you analyse how much the AI contributed
            "attributes": {"source": a.source},
        })
    return out


# ---------------------------------------------------------------- YOLO
def to_yolo(images: list[ExpImage], anns: list[ExpAnnotation], segmentation: bool = False) -> dict[str, str]:
    """Returns {image file_name -> label file content}. Images without annotations get an
    empty label file, which YOLO treats as a background image."""
    by_image: dict[int, list[ExpAnnotation]] = {im.id: [] for im in images}
    for a in anns:
        by_image.setdefault(a.image_id, []).append(a)

    labels = {}
    for im in images:
        lines = []
        W, H = float(im.width), float(im.height)
        for a in by_image.get(im.id, []):
            if segmentation:
                if a.kind == "polygon" and a.points:
                    pts = a.points
                else:
                    x, y, w, h = a.bbox
                    pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                coords = " ".join(f"{p[0] / W:.6f} {p[1] / H:.6f}" for p in pts)
                lines.append(f"{a.category} {coords}")
            else:
                x, y, w, h = a.bbox
                lines.append(f"{a.category} {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        labels[im.file_name] = "\n".join(lines) + ("\n" if lines else "")
    return labels


def yolo_data_yaml(categories: list[str]) -> str:
    names = "\n".join(f"  {i}: {json.dumps(n)}" for i, n in enumerate(categories))
    return (
        "# YOLO dataset config. Adjust train/val splits as needed.\n"
        "path: .\n"
        "train: images\n"
        "val: images\n"
        f"nc: {len(categories)}\n"
        f"names:\n{names}\n"
    )


# ---------------------------------------------------------------- Pascal VOC
def to_voc(image: ExpImage, anns: list[ExpAnnotation], categories: list[str], depth: int = 3) -> str:
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = "JPEGImages"
    ET.SubElement(root, "filename").text = image.file_name
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image.width)
    ET.SubElement(size, "height").text = str(image.height)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(root, "segmented").text = "0"
    for a in anns:
        x, y, w, h = a.bbox
        xmin = min(max(int(round(x)) + 1, 1), image.width)
        ymin = min(max(int(round(y)) + 1, 1), image.height)
        xmax = min(max(int(round(x + w)), xmin), image.width)
        ymax = min(max(int(round(y + h)), ymin), image.height)
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = categories[a.category]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(xmin)
        ET.SubElement(bb, "ymin").text = str(ymin)
        ET.SubElement(bb, "xmax").text = str(xmax)
        ET.SubElement(bb, "ymax").text = str(ymax)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")
