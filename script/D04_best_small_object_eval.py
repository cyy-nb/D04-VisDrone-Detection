"""Evaluate the detection-embedded D04 model with COCO AP metrics."""

from __future__ import annotations

try:
    from .d04_common import project_path
    from .D04_small_object_eval_core import main as evaluate_main
except ImportError:
    from d04_common import project_path
    from D04_small_object_eval_core import main as evaluate_main

MODEL_PATH = project_path(
    "experiments", "detection_embedded", "dee_yolov8s_v1", "weights", "best.pt"
)
IMAGE_DIR = project_path("datasets", "VisDroneYOLO", "images", "val")
LABEL_DIR = project_path("datasets", "VisDroneYOLO", "labels", "val")
OUTPUT_DIR = project_path(
    "experiments", "detection_embedded", "dee_yolov8s_v1_best_eval", "ap_eval"
)


def main() -> None:
    evaluate_main(
        default_model=MODEL_PATH,
        default_image_dir=IMAGE_DIR,
        default_label_dir=LABEL_DIR,
        default_output_dir=OUTPUT_DIR,
        title="D04 Detection-Embedded AP Evaluation",
        description="D04 Detection-Embedded Evaluation",
        result_name="d04_best_ap_results.txt",
    )


if __name__ == "__main__":
    main()
