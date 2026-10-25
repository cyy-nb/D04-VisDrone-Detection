"""Run Ultralytics validation and save COCO-compatible prediction JSON."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ultralytics import YOLO

try:
    from .d04_common import project_path
    from .d04_registry import ensure_d04_registered
except ImportError:
    from d04_common import project_path
    from d04_registry import ensure_d04_registered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=project_path("experiments", "baseline_640", "weights", "best.pt"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=project_path("datasets", "VisDroneYOLO", "visdrone.yaml"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.model.is_file():
        raise FileNotFoundError(f"Model weights not found: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    ensure_d04_registered()
    result = YOLO(str(args.model)).val(
        data=str(args.data),
        imgsz=args.imgsz,
        device=args.device,
        save_json=True,
        save_plots=True,
    )
    print(f"Validation results: {result.save_dir}")


if __name__ == "__main__":
    main()
