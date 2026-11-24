"""Unit tests for geometry, exporters and image ingestion.

Run with:  python -m pytest backend/tests   (or python -m unittest discover backend/tests)
"""
import io
import json
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="labelloop_test_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.export.formats import ExpAnnotation, ExpImage, to_coco, to_voc, to_yolo, yolo_data_yaml  # noqa: E402
from app.export.package import build_zip, safe_name  # noqa: E402
from app.geometry import bbox_from_points, bbox_iou, clip_bbox, polygon_area  # noqa: E402

CATS = ["car", "truck"]
IMG = ExpImage(id=7, file_name="000007_scene.jpg", width=200, height=100)
BOX = ExpAnnotation(image_id=7, category=0, kind="bbox", bbox=[10, 20, 50, 40])
POLY = ExpAnnotation(image_id=7, category=1, kind="polygon",
                     bbox=[100, 10, 40, 40], points=[[100, 10], [140, 10], [140, 50], [100, 50]])


class GeometryTest(unittest.TestCase):
    def test_polygon_area_square(self):
        self.assertAlmostEqual(polygon_area([[0, 0], [4, 0], [4, 4], [0, 4]]), 16)

    def test_polygon_area_orientation_independent(self):
        self.assertAlmostEqual(polygon_area([[0, 4], [4, 4], [4, 0], [0, 0]]), 16)

    def test_bbox_from_points(self):
        self.assertEqual(bbox_from_points([[3, 5], [10, 1], [7, 9]]), [3, 1, 7, 8])

    def test_clip_bbox(self):
        self.assertEqual(clip_bbox([-10, 90, 30, 30], 100, 100), [0, 90, 20, 10])

    def test_iou(self):
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [5, 0, 10, 10]), 50 / 150)
        self.assertEqual(bbox_iou([0, 0, 1, 1], [5, 5, 1, 1]), 0.0)


class CocoTest(unittest.TestCase):
    def test_structure(self):
        coco = to_coco([IMG], [BOX, POLY], CATS)
        self.assertEqual([c["id"] for c in coco["categories"]], [1, 2])
        self.assertEqual(coco["images"][0]["file_name"], IMG.file_name)
        a_box, a_poly = coco["annotations"]
        self.assertEqual(a_box["category_id"], 1)
        self.assertEqual(a_box["area"], 2000)
        self.assertEqual(a_box["segmentation"], [])
        self.assertEqual(a_poly["segmentation"], [[100, 10, 140, 10, 140, 50, 100, 50]])
        self.assertEqual(a_poly["area"], 1600)
        self.assertEqual(len({a["id"] for a in coco["annotations"]}), 2)

    def test_loads_with_pycocotools_if_available(self):
        try:
            from pycocotools.coco import COCO
        except ImportError:
            self.skipTest("pycocotools not installed")
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(to_coco([IMG], [BOX, POLY], CATS), f)
        api = COCO(f.name)
        self.assertEqual(len(api.getAnnIds()), 2)


class YoloTest(unittest.TestCase):
    def test_detection_line(self):
        labels = to_yolo([IMG], [BOX])
        cls, cx, cy, w, h = labels[IMG.file_name].split()
        self.assertEqual(cls, "0")
        self.assertAlmostEqual(float(cx), 35 / 200)
        self.assertAlmostEqual(float(cy), 40 / 100)
        self.assertAlmostEqual(float(w), 50 / 200)
        self.assertAlmostEqual(float(h), 40 / 100)

    def test_segmentation_lines(self):
        labels = to_yolo([IMG], [BOX, POLY], segmentation=True)
        box_line, poly_line = labels[IMG.file_name].strip().split("\n")
        self.assertEqual(len(box_line.split()), 1 + 8)  # box becomes a 4-point polygon
        vals = [float(v) for v in poly_line.split()[1:]]
        self.assertTrue(all(0 <= v <= 1 for v in vals))
        self.assertAlmostEqual(vals[0], 0.5)

    def test_empty_image_gets_empty_file(self):
        self.assertEqual(to_yolo([IMG], []), {IMG.file_name: ""})

    def test_yaml_quotes_names(self):
        y = yolo_data_yaml(["car", "pick-up: truck"])
        self.assertIn('1: "pick-up: truck"', y)


class VocTest(unittest.TestCase):
    def test_one_based_inclusive(self):
        root = ET.fromstring(to_voc(IMG, [BOX], CATS))
        bb = root.find("object/bndbox")
        self.assertEqual([int(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")], [11, 21, 60, 60])
        self.assertEqual(root.find("object/name").text, "car")

    def test_clamped_to_image(self):
        edge = ExpAnnotation(image_id=7, category=0, kind="bbox", bbox=[190, 90, 10, 10])
        bb = ET.fromstring(to_voc(IMG, [edge], CATS)).find("object/bndbox")
        self.assertEqual(int(bb.find("xmax").text), 200)
        self.assertEqual(int(bb.find("ymax").text), 100)


class PackageTest(unittest.TestCase):
    def test_safe_name(self):
        self.assertEqual(safe_name(3, "my photo (1).JPG", ".jpg"), "000003_my_photo_1.jpg")
        self.assertEqual(safe_name(3, "../../etc/passwd", ".png"), "000003_passwd.png")

    def test_zip_contents_per_format(self):
        tmp = Path(tempfile.mkdtemp())
        img_file = tmp / "x.jpg"
        img_file.write_bytes(b"fake")
        expected = {
            "coco": {"annotations.json", f"images/{IMG.file_name}"},
            "yolo": {"data.yaml", "labels/000007_scene.txt", f"images/{IMG.file_name}"},
            "yolo-seg": {"data.yaml", "labels/000007_scene.txt"},
            "voc": {"Annotations/000007_scene.xml", f"JPEGImages/{IMG.file_name}"},
        }
        for fmt, must_have in expected.items():
            p = build_zip(fmt, [IMG], [BOX, POLY], CATS, {7: img_file}, True, "test")
            with zipfile.ZipFile(p) as z:
                names = set(z.namelist())
            os.unlink(p)
            self.assertTrue(must_have <= names, f"{fmt}: missing {must_have - names}")


class StorageTest(unittest.TestCase):
    def setUp(self):
        from app import storage
        self.storage = storage

    def _jpeg(self, size=(40, 20), orientation=None):
        from PIL import Image
        im = Image.new("RGB", size, (200, 10, 10))
        buf = io.BytesIO()
        if orientation:
            exif = im.getexif()
            exif[0x0112] = orientation
            im.save(buf, format="JPEG", exif=exif.tobytes())
        else:
            im.save(buf, format="JPEG")
        return buf.getvalue()

    def test_plain_jpeg_kept(self):
        name, w, h = self.storage.save_image(self._jpeg())
        self.assertEqual((w, h), (40, 20))
        self.assertTrue(name.endswith(".jpg"))

    def test_exif_rotation_is_baked_in(self):
        # orientation 6 = rotate 90 degrees: browser shows 20x40, so we must store 20x40
        name, w, h = self.storage.save_image(self._jpeg(orientation=6))
        self.assertEqual((w, h), (20, 40))

    def test_16bit_tiff_becomes_png(self):
        import numpy as np
        from PIL import Image
        arr = (np.arange(64 * 32, dtype=np.uint16).reshape(32, 64) * 20)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="TIFF")
        name, w, h = self.storage.save_image(buf.getvalue())
        self.assertTrue(name.endswith(".png"))
        self.assertEqual((w, h), (64, 32))

    def test_garbage_rejected(self):
        with self.assertRaises(self.storage.InvalidImage):
            self.storage.save_image(b"definitely not an image")


if __name__ == "__main__":
    unittest.main()
