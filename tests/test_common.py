"""Tests for shared path, label and JSON helpers."""

import json
import tempfile
import unittest
from pathlib import Path

from script.d04_common import atomic_write_text, image_files, project_path, read_yolo_label, write_json


class CommonUtilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_project_path_honors_explicit_root(self):
        self.assertEqual(
            project_path("datasets", "VisDrone", root=self.root),
            self.root / "datasets" / "VisDrone",
        )

    def test_image_files_are_sorted_case_insensitively(self):
        (self.root / "B.JPG").write_bytes(b"jpg")
        (self.root / "a.png").write_bytes(b"png")
        (self.root / "ignore.txt").write_text("x")
        self.assertEqual([path.name for path in image_files(self.root)], ["a.png", "B.JPG"])

    def test_read_yolo_label_returns_pixel_boxes(self):
        label = self.root / "sample.txt"
        label.write_text("0 0.5 0.5 0.2 0.25\n", encoding="utf-8")
        objects = read_yolo_label(label, 100, 80)
        self.assertEqual(objects[0]["class_id"], 0)
        self.assertEqual(objects[0]["bbox"], [40.0, 30.0, 20.0, 20.0])
        self.assertEqual(objects[0]["area"], 400.0)

    def test_read_yolo_label_rejects_malformed_or_invalid_rows(self):
        label = self.root / "bad.txt"
        for text in (
            "0 0.5 0.5 0.2\n",
            "10 0.5 0.5 0.2 0.2\n",
            "0 0.5 0.5 -0.2 0.2\n",
            "0 nan 0.5 0.2 0.2\n",
        ):
            with self.subTest(text=text):
                label.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_yolo_label(label, 100, 80)

    def test_atomic_json_write_creates_parent_and_round_trips(self):
        target = self.root / "nested" / "result.json"
        write_json(target, {"value": [1, 2, 3]})
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["value"], [1, 2, 3])
        self.assertFalse(any(path.suffix == ".tmp" for path in target.parent.iterdir()))

    def test_atomic_text_write_does_not_leave_temp_files(self):
        target = self.root / "result.txt"
        atomic_write_text(target, "ok\n")
        self.assertEqual(target.read_text(encoding="utf-8"), "ok\n")
        self.assertEqual(list(self.root.iterdir()), [target])


if __name__ == "__main__":
    unittest.main()
