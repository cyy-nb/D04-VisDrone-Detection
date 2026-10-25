"""Convert VisDrone DET annotations into ten-class YOLO labels."""

from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Sequence

from PIL import Image

try:
    from .d04_common import atomic_write_text, project_path
except ImportError:
    from d04_common import atomic_write_text, project_path

# Default paths remain compatible with the original project layout.  Set
# D04_HOME to relocate the whole tree without editing source files.
DATA_ROOT = project_path("datasets", "VisDrone")
OUTPUT_ROOT = project_path("datasets", "VisDroneYOLO")

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def _parse_integer(value: str, *, field: str, path: Path, line_number: int) -> int:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: invalid {field} value {value!r}") from exc
    if not math.isfinite(parsed) or not parsed.is_integer():
        raise ValueError(f"{path}:{line_number}: {field} must be an integer")
    return int(parsed)


def convert_split(
    split_name: str,
    source_name: str,
    data_root: str | Path = DATA_ROOT,
    output_root: str | Path = OUTPUT_ROOT,
) -> int:
    """Convert one VisDrone split and return the number of processed images."""

    source_dir = Path(data_root) / source_name
    image_dir = source_dir / "images"
    annotation_dir = source_dir / "annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Missing image or annotation directory: {source_dir}")

    files = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() == ".jpg"),
        key=lambda path: path.name.lower(),
    )
    if not files:
        raise ValueError(f"No JPG images found: {image_dir}")

    missing = [path.name for path in files if not (annotation_dir / f"{path.stem}.txt").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing annotation {annotation_dir / (Path(missing[0]).stem + '.txt')} "
            f"({len(missing)} image(s) affected)"
        )

    output_image_dir = Path(output_root) / "images" / split_name
    output_label_dir = Path(output_root) / "labels" / split_name
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {split_name}: {len(files)} images")
    for index, image_path in enumerate(files, start=1):
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid image dimensions: {image_path}")

        yolo_lines: list[str] = []
        with annotation_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = [part.strip() for part in line.split(",")]
                if len(parts) < 8:
                    raise ValueError(f"{annotation_path}:{line_number}: expected at least 8 columns")

                try:
                    x, y, width, height = (float(value) for value in parts[:4])
                except ValueError as exc:
                    raise ValueError(f"{annotation_path}:{line_number}: invalid bbox coordinate") from exc
                score = _parse_integer(parts[4], field="score", path=annotation_path, line_number=line_number)
                category = _parse_integer(
                    parts[5], field="category", path=annotation_path, line_number=line_number
                )

                # score=0 marks ignore regions; VisDrone category 11 is "others".
                if score == 0 or category < 1 or category > 10:
                    continue
                if not all(math.isfinite(value) for value in (x, y, width, height)):
                    continue
                if width <= 0 or height <= 0:
                    continue

                x2 = min(x + width, image_width)
                y2 = min(y + height, image_height)
                x = max(x, 0.0)
                y = max(y, 0.0)
                width = x2 - x
                height = y2 - y
                if width <= 0 or height <= 0:
                    continue

                x_center = (x + width / 2) / image_width
                y_center = (y + height / 2) / image_height
                width /= image_width
                height /= image_height
                if not 0 <= x_center <= 1 or not 0 <= y_center <= 1:
                    continue
                if not 0 < width <= 1 or not 0 < height <= 1:
                    continue

                yolo_lines.append(
                    f"{category - 1} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

        atomic_write_text(output_label_dir / f"{image_path.stem}.txt", "\n".join(yolo_lines))
        output_image_path = output_image_dir / image_path.name
        if not output_image_path.exists():
            shutil.copy2(image_path, output_image_path)

        if index % 500 == 0:
            print(f"Processed {index}/{len(files)}")

    print(f"Finished {split_name}: {len(files)} images")
    return len(files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT, help="VisDrone source root")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="YOLO output root")
    parser.add_argument("--splits", nargs="+", choices=tuple(SPLITS), default=list(SPLITS))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    for split_name in args.splits:
        convert_split(split_name, SPLITS[split_name], args.data_root, args.output_root)
    print("\nVisDrone -> YOLO conversion complete.")


if __name__ == "__main__":
    main()
