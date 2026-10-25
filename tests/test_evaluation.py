"""Tests for shared evaluation helpers and ROI filtering."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from script.D04_filter_roi_small32 import filter_predictions
from script.D04_small_object_eval_core import build_coco_ground_truth, render_report


class EvaluationCoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.images = self.root / "images"
        self.labels = self.root / "labels"
        self.images.mkdir()
        self.labels.mkdir()
        Image.new("RGB", (100, 80)).save(self.images / "sample.jpg")
        (self.labels / "sample.txt").write_text(
            "0 0.5 0.5 0.2 0.25\n9 0.2 0.2 0.1 0.1\n",
            encoding="utf-8",
        )

    def test_build_coco_ground_truth_uses_one_based_category_ids(self):
        coco = build_coco_ground_truth(
            [self.images / "sample.jpg"],
            self.labels,
            description="test",
        )
        self.assertEqual([category["id"] for category in coco["categories"]], list(range(1, 11)))
        self.assertEqual([item["category_id"] for item in coco["annotations"]], [1, 10])
        self.assertEqual(coco["annotations"][0]["area"], 400.0)

    def test_build_coco_ground_truth_rejects_missing_label(self):
        (self.labels / "sample.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            build_coco_ground_truth(
                [self.images / "sample.jpg"],
                self.labels,
                description="test",
            )

    def test_render_report_contains_standard_metrics(self):
        report = render_report(
            "Example",
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            image_count=2,
            prediction_count=3,
        )
        self.assertIn("mAP50-95: 0.100000", report)
        self.assertIn("COCO predictions: 3", report)


class RoiFilterTests(unittest.TestCase):
    def test_filter_predictions_keeps_only_small_boxes(self):
        payload = {
            "images": [
                {"predictions": [{"bbox": [0, 0, 10, 10]}, {"bbox": [0, 0, 40, 40]}]},
                {"predictions": [{"bbox": [0, 0, 31, 31]}]},
            ]
        }
        stats = {
            "total_before": 0,
            "total_after": 0,
            "filtered_count": 0,
            "invalid_boxes": 0,
        }
        filtered = filter_predictions(payload, stats, small_max=32 * 32)
        self.assertEqual(len(filtered["images"][0]["predictions"]), 1)
        self.assertEqual(len(filtered["images"][1]["predictions"]), 1)
        self.assertEqual(stats["total_before"], 3)
        self.assertEqual(stats["total_after"], 2)
        self.assertEqual(stats["filtered_count"], 1)


if __name__ == "__main__":
    unittest.main()
