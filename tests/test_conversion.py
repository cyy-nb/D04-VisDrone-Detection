"""Small synthetic fixtures; no VisDrone download or GPU required."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from script.convert_visdrone import convert_split
from script.yolo2coco import convert


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        (self.source / "images").mkdir(parents=True)
        (self.source / "annotations").mkdir()
        Image.new("RGB", (100, 80)).save(self.source / "images" / "sample.JPG")
        self.annotation = self.source / "annotations" / "sample.txt"
        self.output = self.root / "output"

    def run_conversion(self, text):
        self.annotation.write_text(text, encoding="utf-8")
        convert_split("val", "source", self.root, self.output)
        return (self.output / "labels/val/sample.txt").read_text().splitlines()

    def test_clipping_classes_and_coco_round_trip(self):
        lines = self.run_conversion(
            "-10,-5,30,25,1,1,0,0\n"
            "0,0,10,10,1,11,0,0\n"
            "0,0,10,10,0,2,0,0\n"
            "120,0,10,10,1,3,0,0\n"
            "0,0,-10,10,1,3,0,0\n"
            "nan,0,10,10,1,3,0,0\n"
        )
        self.assertEqual(lines, ["0 0.100000 0.125000 0.200000 0.250000"])
        target = self.root / "coco.json"
        convert(self.output / "images/val", self.output / "labels/val", target)
        data = json.loads(target.read_text())
        self.assertEqual([item["id"] for item in data["categories"]], list(range(1, 11)))
        self.assertEqual(data["annotations"][0]["category_id"], 1)
        self.assertEqual(data["annotations"][0]["bbox"], [0, 0, 20, 20])
        self.assertEqual(data["annotations"][0]["area"], 400)

    def test_malformed_annotation_fails_with_context(self):
        with self.assertRaisesRegex(ValueError, "expected at least 8 columns"):
            self.run_conversion("0,0,10,10,1\n")

    def test_missing_annotation_fails_before_output(self):
        with self.assertRaises(FileNotFoundError):
            convert_split("val", "source", self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_empty_annotation_is_valid_background(self):
        self.assertEqual(self.run_conversion(""), [])

    def test_coco_rejects_unknown_class(self):
        self.run_conversion("")
        (self.output / "labels/val/sample.txt").write_text("10 0.5 0.5 0.2 0.2")
        with self.assertRaises(ValueError):
            convert(self.output / "images/val", self.output / "labels/val", self.root / "bad.json")

    def test_coco_missing_label_does_not_write_output(self):
        target = self.root / "missing.json"
        with self.assertRaises(FileNotFoundError):
            convert(self.source / "images", self.source / "annotations", target)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
