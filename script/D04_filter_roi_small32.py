"""Filter fused ROI-SR predictions to small bounding boxes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

try:
    from .d04_common import project_path, write_json
except ImportError:
    from d04_common import project_path, write_json

INPUT_FILE = project_path(
    "experiments", "roi_sr_x2", "roi_fused_predictions.json"
)
OUTPUT_FILE = project_path(
    "experiments", "roi_sr_x2", "roi_fused_predictions_small32.json"
)


def filter_predictions(obj: Any, stats: dict[str, int], *, small_max: float) -> Any:
    """Recursively keep prediction dictionaries whose bbox area is below threshold."""

    if isinstance(obj, list):
        filtered: list[Any] = []
        for item in obj:
            if isinstance(item, dict) and "bbox" in item:
                stats["total_before"] += 1
                bbox = item.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    stats["invalid_boxes"] += 1
                    continue
                try:
                    x1, y1, x2, y2 = (float(value) for value in bbox)
                except (TypeError, ValueError):
                    stats["invalid_boxes"] += 1
                    continue
                if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
                    stats["invalid_boxes"] += 1
                    continue
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if area < small_max:
                    filtered.append(item)
                    stats["total_after"] += 1
                else:
                    stats["filtered_count"] += 1
            else:
                filtered.append(filter_predictions(item, stats, small_max=small_max))
        return filtered
    if isinstance(obj, dict):
        return {
            key: value if key == "bbox" else filter_predictions(value, stats, small_max=small_max)
            for key, value in obj.items()
        }
    return obj


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--max-area", type=float, default=32 * 32)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    with args.input.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    stats = {
        "total_before": 0,
        "total_after": 0,
        "filtered_count": 0,
        "invalid_boxes": 0,
    }
    filtered = filter_predictions(data, stats, small_max=args.max_area)
    write_json(args.output, filtered)

    ratio = stats["total_after"] / stats["total_before"] * 100 if stats["total_before"] else 0.0
    print(f"Input: {args.input}\nOutput: {args.output}")
    print(f"Original boxes: {stats['total_before']}")
    print(f"Small boxes (< {args.max_area}): {stats['total_after']}")
    print(f"Filtered boxes: {stats['filtered_count']}")
    print(f"Invalid boxes: {stats['invalid_boxes']}")
    print(f"Small-object ratio: {ratio:.2f}%")


if __name__ == "__main__":
    main()
