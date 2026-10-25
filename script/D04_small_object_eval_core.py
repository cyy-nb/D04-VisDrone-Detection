"""Reusable COCO-style evaluation for the D04 detector variants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

try:  # Supports both ``python -m script...`` and direct script execution.
    from .d04_common import CLASS_NAMES, atomic_write_text, image_files, read_yolo_label, write_json
except ImportError:
    from d04_common import CLASS_NAMES, atomic_write_text, image_files, read_yolo_label, write_json

try:
    from .d04_registry import ensure_d04_registered
except ImportError:
    from d04_registry import ensure_d04_registered


def build_coco_ground_truth(
    images: Sequence[Path],
    label_dir: str | Path,
    *,
    description: str,
) -> dict[str, Any]:
    """Build COCO ground truth from validated YOLO labels."""

    labels = Path(label_dir)
    if not images:
        raise ValueError("No images were supplied for evaluation")
    if not labels.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels}")

    coco: dict[str, Any] = {
        "info": {"description": description},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": index + 1, "name": name, "supercategory": "object"}
            for index, name in enumerate(CLASS_NAMES)
        ],
    }

    annotation_id = 1
    for image_id, image_path in enumerate(images, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        label_path = labels / f"{image_path.stem}.txt"
        objects = read_yolo_label(label_path, width, height)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )
        for obj in objects:
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": obj["class_id"] + 1,
                    "bbox": obj["bbox"],
                    "area": obj["area"],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    if not coco["annotations"]:
        raise ValueError("The selected validation split has no annotated objects")
    return coco


def collect_predictions(
    model: Any,
    images: Sequence[Path],
    *,
    imgsz: int = 640,
    confidence: float = 0.001,
    max_det: int = 317,
    device: str | int = "0",
) -> list[dict[str, Any]]:
    """Run a detector and convert Ultralytics boxes to COCO prediction records."""

    predictions: list[dict[str, Any]] = []
    for image_id, image_path in enumerate(images, start=1):
        results = model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=confidence,
            max_det=max_det,
            device=device,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        for box, class_id, score in zip(boxes, classes, scores):
            x1, y1, x2, y2 = (float(value) for value in box)
            box_width = x2 - x1
            box_height = y2 - y1
            category_id = int(class_id) + 1
            if not math.isfinite(score) or not 1 <= category_id <= len(CLASS_NAMES):
                continue
            if not all(math.isfinite(value) for value in (x1, y1, box_width, box_height)):
                continue
            if box_width <= 0 or box_height <= 0:
                continue
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, box_width, box_height],
                    "score": float(score),
                }
            )
    return predictions


def evaluate_coco(
    ground_truth: str | Path,
    predictions: str | Path,
    *,
    max_det: int = 317,
) -> list[float]:
    """Evaluate COCO JSON and return the twelve standard summary metrics."""

    gt_path = Path(ground_truth)
    pred_path = Path(predictions)
    if not gt_path.is_file():
        raise FileNotFoundError(f"Ground-truth JSON not found: {gt_path}")
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction JSON not found: {pred_path}")

    with pred_path.open("r", encoding="utf-8") as handle:
        prediction_records = json.load(handle)
    if not prediction_records:
        print("WARNING: no predictions were produced; COCO metrics are marked -1.")
        return [-1.0] * 12

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required for COCO evaluation. "
            "Install it with: python -m pip install -r requirements-train.txt"
        ) from exc

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = coco_gt.getImgIds()
    evaluator.params.maxDets = [1, 10, max(10, max_det)]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return [float(value) for value in evaluator.stats]


def render_report(title: str, stats: Sequence[float], *, image_count: int, prediction_count: int) -> str:
    """Render a stable plain-text report for experiment logs."""

    if len(stats) < 6:
        raise ValueError("Expected at least six COCO summary metrics")
    names = ("mAP50-95", "mAP50", "mAP75", "AP_s", "AP_m", "AP_l")
    metrics = "\n".join(f"{name}: {stats[index]:.6f}" for index, name in enumerate(names))
    return (
        f"{title}\n"
        f"{'=' * len(title)}\n\n"
        f"Images: {image_count}\n"
        f"COCO predictions: {prediction_count}\n\n"
        f"{metrics}\n"
    )


def run_evaluation(
    *,
    model_path: str | Path,
    image_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    title: str,
    description: str,
    result_name: str,
    imgsz: int = 640,
    confidence: float = 0.001,
    max_det: int = 317,
    device: str | int = "0",
) -> dict[str, Path]:
    """Run prediction and COCO evaluation and persist all reusable artifacts."""

    images = image_files(image_dir)
    if not images:
        raise ValueError(f"No supported images found in: {image_dir}")
    if not Path(label_dir).is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ground_truth = build_coco_ground_truth(images, label_dir, description=description)
    gt_path = write_json(output / "ground_truth.json", ground_truth)

    ensure_d04_registered()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for model evaluation. "
            "Install it with: python -m pip install -r requirements-train.txt"
        ) from exc

    print(f"Model: {model_path}")
    print(f"Images: {len(images)}")
    model = YOLO(str(model_path))
    prediction_records = collect_predictions(
        model,
        images,
        imgsz=imgsz,
        confidence=confidence,
        max_det=max_det,
        device=device,
    )
    prediction_path = write_json(output / "predictions.json", prediction_records)
    stats = evaluate_coco(gt_path, prediction_path, max_det=max_det)
    report = render_report(
        title,
        stats,
        image_count=len(images),
        prediction_count=len(prediction_records),
    )
    result_path = atomic_write_text(output / result_name, report)
    return {"ground_truth": gt_path, "predictions": prediction_path, "results": result_path}


def build_parser(
    *,
    default_model: str | Path,
    default_image_dir: str | Path,
    default_label_dir: str | Path,
    default_output_dir: str | Path,
    description: str,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", type=Path, default=Path(default_model), help="YOLO weights path")
    parser.add_argument("--image-dir", type=Path, default=Path(default_image_dir), help="validation image directory")
    parser.add_argument("--label-dir", type=Path, default=Path(default_label_dir), help="YOLO label directory")
    parser.add_argument("--output-dir", type=Path, default=Path(default_output_dir), help="artifact output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--conf", type=float, default=0.001, help="minimum confidence")
    parser.add_argument("--max-det", type=int, default=317, help="maximum detections per image")
    parser.add_argument("--device", default="0", help='Ultralytics device, for example "0" or "cpu"')
    return parser


def main(
    *,
    default_model: str | Path,
    default_image_dir: str | Path,
    default_label_dir: str | Path,
    default_output_dir: str | Path,
    title: str,
    description: str,
    result_name: str,
    argv: Sequence[str] | None = None,
) -> None:
    args = build_parser(
        default_model=default_model,
        default_image_dir=default_image_dir,
        default_label_dir=default_label_dir,
        default_output_dir=default_output_dir,
        description=description,
    ).parse_args(argv)
    artifacts = run_evaluation(
        model_path=args.model,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        title=title,
        description=description,
        result_name=result_name,
        imgsz=args.imgsz,
        confidence=args.conf,
        max_det=args.max_det,
        device=args.device,
    )
    print("\nArtifacts:")
    for name, path in artifacts.items():
        print(f"  {name}: {path}")
