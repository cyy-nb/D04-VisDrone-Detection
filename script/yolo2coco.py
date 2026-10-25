"""Convert validated YOLO labels into COCO detection annotations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from PIL import Image

try:
    from .d04_common import CLASS_NAMES, image_files, project_path, read_yolo_label, write_json
except ImportError:
    from d04_common import CLASS_NAMES, image_files, project_path, read_yolo_label, write_json

# Default paths remain compatible with the original project layout.
IMG_DIR = project_path("datasets", "VisDroneYOLO", "images", "val")
LABEL_DIR = project_path("datasets", "VisDroneYOLO", "labels", "val")
OUTPUT_JSON = project_path("datasets", "VisDroneYOLO", "annotations", "val_coco.json")


def convert(
    img_dir: str | Path = IMG_DIR,
    label_dir: str | Path = LABEL_DIR,
    output_json: str | Path = OUTPUT_JSON,
) -> dict:
    """Convert a YOLO split to COCO JSON and return the generated document."""

    images = image_files(img_dir)
    labels = Path(label_dir)
    if not labels.is_dir():
        raise FileNotFoundError(f"Label directory not found: {labels}")
    if not images:
        raise ValueError(f"No supported images found: {img_dir}")

    coco = {
        "info": {"description": "D04 VisDrone YOLO to COCO conversion"},
        "licenses": [],
        "images": [],
        "annotations": [],
        # COCO category IDs are one-based throughout this repository.
        "categories": [
            {"id": index + 1, "name": name}
            for index, name in enumerate(CLASS_NAMES)
        ],
    }

    annotation_id = 1
    for image_id, image_path in enumerate(images, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for obj in read_yolo_label(labels / f"{image_path.stem}.txt", width, height):
            x, y, box_width, box_height = obj["bbox"]
            x1 = min(max(x, 0.0), width)
            y1 = min(max(y, 0.0), height)
            x2 = min(max(x + box_width, 0.0), width)
            y2 = min(max(y + box_height, 0.0), height)
            box_width = x2 - x1
            box_height = y2 - y1
            if box_width <= 0 or box_height <= 0:
                continue
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": obj["class_id"] + 1,
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    write_json(output_json, coco)
    print(f"COCO annotations written: {output_json}")
    return coco


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=IMG_DIR)
    parser.add_argument("--label-dir", type=Path, default=LABEL_DIR)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    convert(args.image_dir, args.label_dir, args.output_json)


if __name__ == "__main__":
    main()
