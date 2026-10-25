# -*- coding: utf-8 -*-
"""Benchmark baseline and detection-embedded D04 inference latency."""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from script.d04_common import atomic_write_text, image_files, project_path
from script.d04_registry import ensure_d04_registered


def sync() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def load_images(image_dir: Path, limit: int | None = None) -> list[tuple[str, np.ndarray]]:
    paths = image_files(image_dir)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise RuntimeError(f"No validation images found: {image_dir}")

    images: list[tuple[str, np.ndarray]] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"OpenCV could not read image: {path}")
        images.append((path.name, image))

    print(f"Loaded validation images: {len(images)}")
    return images


def make_tensor(image: np.ndarray, model: torch.nn.Module, imgsz: int) -> torch.Tensor:
    """Prepare one normalized NCHW tensor outside the pure-forward timer."""

    resized = cv2.resize(image, (imgsz, imgsz))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    parameter = next(model.parameters())
    tensor = torch.from_numpy(array).unsqueeze(0).to(device=parameter.device, dtype=parameter.dtype)
    return tensor.float().div_(255.0)


def summarize(times: Sequence[float]) -> dict[str, float]:
    if not times:
        raise ValueError("At least one timing sample is required")
    average = statistics.mean(times)
    return {
        "avg": average,
        "median": statistics.median(times),
        "p95": float(np.percentile(times, 95)),
        "fps": 1000.0 / average,
    }


def pure_forward(
    yolo: YOLO,
    images: Sequence[tuple[str, np.ndarray]],
    *,
    imgsz: int,
    warmup: int,
) -> dict[str, float]:
    model = yolo.model.eval()
    with torch.inference_mode():
        for index in range(warmup):
            model(make_tensor(images[index % len(images)][1], model, imgsz))
        sync()

        times: list[float] = []
        for _, image in images:
            tensor = make_tensor(image, model, imgsz)
            sync()
            start = time.perf_counter()
            model(tensor)
            sync()
            times.append((time.perf_counter() - start) * 1000)
    return summarize(times)


def end_to_end(
    yolo: YOLO,
    images: Sequence[tuple[str, np.ndarray]],
    *,
    imgsz: int,
    device: str,
    warmup: int,
    confidence: float,
    iou: float,
) -> dict[str, float]:
    with torch.inference_mode():
        for index in range(warmup):
            yolo.predict(
                source=images[index % len(images)][1],
                imgsz=imgsz,
                device=device,
                conf=confidence,
                iou=iou,
                verbose=False,
                save=False,
            )
        sync()

        times: list[float] = []
        for _, image in images:
            sync()
            start = time.perf_counter()
            yolo.predict(
                source=image,
                imgsz=imgsz,
                device=device,
                conf=confidence,
                iou=iou,
                verbose=False,
                save=False,
            )
            sync()
            times.append((time.perf_counter() - start) * 1000)
    return summarize(times)


def format_stats(name: str, result: dict[str, float]) -> str:
    return (
        f"{name}\n"
        f"  Average latency : {result['avg']:.3f} ms/image\n"
        f"  Median latency  : {result['median']:.3f} ms/image\n"
        f"  P95 latency     : {result['p95']:.3f} ms/image\n"
        f"  FPS             : {result['fps']:.3f}\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=project_path("experiments", "baseline_640", "weights", "best.pt"))
    parser.add_argument(
        "--d04",
        type=Path,
        default=project_path(
            "experiments", "detection_embedded", "dee_yolov8s_v1", "weights", "best.pt"
        ),
    )
    parser.add_argument(
        "--image-dir", type=Path, default=project_path("datasets", "VisDroneYOLO", "images", "val")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project_path("experiments", "latency_comparison")
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help='CUDA device such as "0", or "cpu"')
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--limit", type=int, default=None, help="benchmark only the first N images")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ensure_d04_registered()
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    models = {"Baseline": args.baseline, "D04_Detection_Embedded": args.d04}

    for name, path in models.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} weights not found: {path}")
    images = load_images(args.image_dir, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is unavailable; benchmark will run on CPU.")

    results: dict[str, dict[str, object]] = {}
    for name, path in models.items():
        print(f"\nBenchmarking: {name}\nWeights: {path}")
        yolo = YOLO(str(path))
        pure = pure_forward(yolo, images, imgsz=args.imgsz, warmup=args.warmup)
        print(format_stats("Pure model forward", pure))
        e2e = end_to_end(
            yolo,
            images,
            imgsz=args.imgsz,
            device=device,
            warmup=args.warmup,
            confidence=args.conf,
            iou=args.iou,
        )
        print(format_stats("End-to-end prediction", e2e))
        results[name] = {"pure": pure, "e2e": e2e, "path": str(path)}
        del yolo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    baseline = results["Baseline"]
    d04 = results["D04_Detection_Embedded"]
    pure_ratio = d04["pure"]["avg"] / baseline["pure"]["avg"]
    e2e_ratio = d04["e2e"]["avg"] / baseline["e2e"]["avg"]

    report = [
        "D04 BASELINE VS DETECTION-EMBEDDED LATENCY",
        "=" * 65,
        f"Time: {datetime.now().isoformat(timespec='seconds')}",
        f"Device: {device}",
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"PyTorch: {torch.__version__}",
        f"Input size: {args.imgsz}x{args.imgsz}",
        f"Validation images: {len(images)}",
        f"Warm-up runs per method/model: {args.warmup}",
        f"Baseline weights: {baseline['path']}",
        f"D04 weights: {d04['path']}",
        "",
    ]
    for name in ("Baseline", "D04_Detection_Embedded"):
        entry = results[name]
        report.append(f"MODEL: {name}")
        report.append(format_stats("Pure model forward", entry["pure"]))
        report.append(format_stats("End-to-end prediction", entry["e2e"]))
        report.append("-" * 65)
    report += [
        "LATENCY RATIOS (D04 / BASELINE)",
        f"Pure model latency ratio: {pure_ratio:.4f}x",
        f"End-to-end latency ratio: {e2e_ratio:.4f}x",
        f"Constraint <= 3x: {'PASS' if e2e_ratio <= 3 else 'FAIL'}",
        "",
        "Notes:",
        "1. Pure forward excludes image decoding, resizing and NMS.",
        "2. End-to-end uses sequential single-image predict() calls.",
        "3. FPS = 1000 / average latency in milliseconds.",
        "4. Results are inference benchmarks, not training time.",
        "5. This test does not include ROI-SR cropping or SR processing.",
    ]

    text = "\n".join(report) + "\n"
    output_file = atomic_write_text(args.output_dir / "D04_latency_fps_results.txt", text)
    print("\n" + text)
    print(f"Results saved: {output_file}")


if __name__ == "__main__":
    main()
