"""Build a downloadable zip: annotation files in the chosen format plus (optionally) the images."""
from __future__ import annotations

import json
import re
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .formats import ExpAnnotation, ExpImage, to_coco, to_voc, to_yolo, yolo_data_yaml

FORMATS = ("coco", "yolo", "yolo-seg", "voc")


def safe_name(image_id: int, original: str, stored_ext: str) -> str:
    """Unique, filesystem-safe archive name. Prefixing the id avoids collisions
    when two uploads share a file name."""
    stem = PurePosixPath(original).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "image"
    return f"{image_id:06d}_{stem[:80]}{stored_ext}"


def build_zip(fmt: str, images: list[ExpImage], anns: list[ExpAnnotation], categories: list[str],
              image_files: dict[int, Path], include_images: bool, project_name: str) -> Path:
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt}")
    tmp = tempfile.NamedTemporaryFile(prefix="labelloop_", suffix=".zip", delete=False)
    tmp.close()
    path = Path(tmp.name)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        if fmt == "coco":
            coco = to_coco(images, anns, categories, description=project_name)
            z.writestr("annotations.json", json.dumps(coco, indent=1))
            img_dir = "images"
        elif fmt in ("yolo", "yolo-seg"):
            labels = to_yolo(images, anns, segmentation=(fmt == "yolo-seg"))
            for file_name, txt in labels.items():
                z.writestr(f"labels/{PurePosixPath(file_name).stem}.txt", txt)
            z.writestr("data.yaml", yolo_data_yaml(categories))
            z.writestr("classes.txt", "\n".join(categories) + "\n")
            img_dir = "images"
        else:  # voc
            by_image: dict[int, list[ExpAnnotation]] = {}
            for a in anns:
                by_image.setdefault(a.image_id, []).append(a)
            for im in images:
                xml = to_voc(im, by_image.get(im.id, []), categories)
                z.writestr(f"Annotations/{PurePosixPath(im.file_name).stem}.xml", xml)
            z.writestr("labels.txt", "\n".join(categories) + "\n")
            img_dir = "JPEGImages"

        if include_images:
            for im in images:
                src = image_files.get(im.id)
                if src is not None and src.exists():
                    # images are already compressed, so store them without recompressing
                    z.write(src, f"{img_dir}/{im.file_name}", compress_type=zipfile.ZIP_STORED)

        z.writestr("README.txt", (
            f"Dataset exported from LabelLoop\n"
            f"project: {project_name}\nformat: {fmt}\n"
            f"images: {len(images)}\nannotations: {len(anns)}\n"
            f"classes: {', '.join(categories)}\n"
        ))
    return path
