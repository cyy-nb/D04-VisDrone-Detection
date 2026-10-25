"""Generate small-object ROI crops and metadata from a baseline detector."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from PIL import Image
from ultralytics import YOLO

try:
    from .d04_common import image_files, project_path, write_json
    from .d04_registry import ensure_d04_registered
except ImportError:
    from d04_common import image_files, project_path, write_json
    from d04_registry import ensure_d04_registered

MODEL_PATH = project_path("experiments", "baseline_640", "weights", "best.pt")
IMAGE_DIR = project_path("datasets", "VisDroneYOLO", "images", "val")
OUTPUT_DIR = project_path("experiments", "roi_sr_x2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--small-max", type=float, default=32**2)
    parser.add_argument("--padding-scale", type=float, default=2.0)
    parser.add_argument("--min-size", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.model.is_file():
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    images = image_files(args.image_dir)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise ValueError(f"No images found in: {args.image_dir}")

    roi_dir = args.output_dir / "roi_crops"
    roi_dir.mkdir(parents=True, exist_ok=True)
    ensure_d04_registered()
    model = YOLO(str(args.model))

    metadata: dict[str, list[dict]] = {}
    roi_count = 0
    print(f"Images: {len(images)} | confidence: {args.conf} | small max area: {args.small_max}")

    for index, image_path in enumerate(images, start=1):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        results = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )
        boxes = results[0].boxes
        records: list[dict] = []
        if boxes is not None:
            for box_id, box in enumerate(boxes):
                x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].cpu().numpy())
                box_width = x2 - x1
                box_height = y2 - y1
                if box_width <= 0 or box_height <= 0 or box_width * box_height >= args.small_max:
                    continue

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                expanded_width = box_width * args.padding_scale
                expanded_height = box_height * args.padding_scale
                rx1 = max(0, int(center_x - expanded_width / 2))
                ry1 = max(0, int(center_y - expanded_height / 2))
                rx2 = min(width, int(center_x + expanded_width / 2))
                ry2 = min(height, int(center_y + expanded_height / 2))
                if rx2 - rx1 < args.min_size or ry2 - ry1 < args.min_size:
                    continue

                roi_name = f"{image_path.stem}_roi_{box_id}.png"
                image.crop((rx1, ry1, rx2, ry2)).save(roi_dir / roi_name)
                records.append(
                    {
                        "roi_file": roi_name,
                        "x1": rx1,
                        "y1": ry1,
                        "x2": rx2,
                        "y2": ry2,
                        "class": int(box.cls[0].item()),
                        "confidence": float(box.conf[0].item()),
                        "original_box": [x1, y1, x2, y2],
                    }
                )
                roi_count += 1
        metadata[image_path.name] = records
        if index % 50 == 0 or index == len(images):
            print(f"{index}/{len(images)} images processed | ROIs: {roi_count}")

    metadata_path = write_json(args.output_dir / "roi_metadata.json", metadata)
    print(f"ROI folder: {roi_dir}\nMetadata: {metadata_path}")


if __name__ == "__main__":
    main()
