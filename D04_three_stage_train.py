"""D04 three-stage training for detection-embedded enhancement.

Stages:
  1. Freeze ``D04EnhanceStem``; train the detector.
  2. Freeze detector layers; train ``D04EnhanceStem`` only.
  3. Unfreeze all layers; joint fine-tuning.

The script never overwrites an existing stage output directory.  Paths default
to the historical ``D04_HOME`` layout and can be overridden from the CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch
from ultralytics import YOLO

from script.d04_common import project_path
from script.d04_registry import ensure_d04_registered

ROOT = project_path()
DATA = ROOT / "datasets" / "VisDroneYOLO" / "visdrone.yaml"
INITIAL_WEIGHTS = (
    ROOT / "experiments" / "detection_embedded" / "dee_yolov8s_v1" / "weights" / "best.pt"
)
PROJECT = ROOT / "experiments" / "detection_embedded" / "three_stage"

IMGSZ = 640
BATCH = 4
WORKERS = 2
DEVICE = "0"

STAGES = [
    {
        "name": "stage1_detector_frozen_enhance_v2",
        "epochs": 15,
        "lr0": 0.001,
        "freeze_mode": "enhance",
    },
    {
        "name": "stage2_enhance_frozen_detector_v2",
        "epochs": 10,
        "lr0": 0.0005,
        "freeze_mode": "detector",
    },
    {
        "name": "stage3_joint_finetune_v2",
        "epochs": 20,
        "lr0": 0.0001,
        "freeze_mode": "none",
    },
]


def configure_freezing(trainer, mode: str) -> None:
    """Set gradient flags after the trainer has built the model."""

    model = trainer.model
    layers = list(model.model)
    if mode == "enhance":
        layers[0].requires_grad_(False)
    elif mode == "detector":
        for layer in layers[1:]:
            layer.requires_grad_(False)
    elif mode != "none":
        raise ValueError(f"Unknown freeze mode: {mode}")

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"[D04] Freeze mode={mode}; trainable parameters={trainable:,}/{total:,}", flush=True)
    if trainable == 0:
        raise RuntimeError("No trainable parameters remain.")


def enforce_frozen_eval(trainer, mode: str) -> None:
    """Keep frozen BatchNorm statistics in eval mode before each batch."""

    layers = list(trainer.model.model)
    if mode == "enhance":
        layers[0].eval()
    elif mode == "detector":
        for layer in layers[1:]:
            layer.eval()


def train_stage(
    weights_path: str | Path,
    stage: dict,
    *,
    data: Path,
    project: Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
) -> Path:
    """Train one stage and return the resulting ``best.pt`` path."""

    weights = Path(weights_path)
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if not weights.is_file():
        raise FileNotFoundError(f"Starting weights not found: {weights}")

    output_dir = project / stage["name"]
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists:\n{output_dir}\n"
            "Rename/archive it or choose a new stage name before rerunning."
        )

    print("\n" + "=" * 72, flush=True)
    print(f"[D04] Starting {stage['name']}", flush=True)
    print(f"[D04] Initial weights: {weights}", flush=True)
    print(f"[D04] Output directory: {output_dir}", flush=True)
    print("=" * 72, flush=True)

    model = YOLO(str(weights))
    model.model.requires_grad_(True)
    mode = stage["freeze_mode"]
    model.add_callback(
        "on_train_start",
        lambda trainer, _mode=mode: configure_freezing(trainer, _mode),
    )
    model.add_callback(
        "on_train_batch_start",
        lambda trainer, _mode=mode: enforce_frozen_eval(trainer, _mode),
    )
    model.train(
        data=str(data),
        epochs=stage["epochs"],
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device,
        project=str(project),
        name=stage["name"],
        exist_ok=False,
        pretrained=True,
        resume=False,
        optimizer="AdamW",
        lr0=stage["lr0"],
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        patience=10,
        amp=torch.cuda.is_available() and device != "cpu",
        cache=False,
        seed=0,
        deterministic=True,
        val=True,
        plots=True,
        save=True,
        save_period=-1,
        close_mosaic=5,
        mosaic=1.0,
    )

    best = output_dir / "weights" / "best.pt"
    last = output_dir / "weights" / "last.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training ended but best.pt was not found: {best}")
    print(f"[D04] Stage best weights: {best}", flush=True)
    print(f"[D04] Stage last weights: {last if last.is_file() else 'not found'}", flush=True)
    return best


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--weights", type=Path, default=INITIAL_WEIGHTS)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--allow-cpu", action="store_true", help="permit training without CUDA")
    parser.add_argument("--smoke-test", action="store_true", help="run one joint-finetuning epoch only")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print("[D04] PyTorch:", torch.__version__)
    print("[D04] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[D04] GPU:", torch.cuda.get_device_name(0))
    elif not args.allow_cpu:
        raise RuntimeError("CUDA is unavailable. Use --allow-cpu only for a deliberate CPU smoke test.")
    else:
        args.device = "cpu"

    ensure_d04_registered()
    current_weights = args.weights
    if args.smoke_test:
        smoke_stage = dict(STAGES[2])
        smoke_stage["name"] = "smoke_test_stage3_v1"
        smoke_stage["epochs"] = 1
        print(
            "[D04] SMOKE TEST: Stage 3 only, 1 epoch; full stages will NOT start.",
            flush=True,
        )
        current_weights = train_stage(
            current_weights,
            smoke_stage,
            data=args.data,
            project=args.project,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
        )
        print("[D04] Smoke-test weights:", current_weights, flush=True)
        return

    for stage in STAGES:
        current_weights = train_stage(
            current_weights,
            stage,
            data=args.data,
            project=args.project,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
        )

    print("\n[D04] All three stages completed.", flush=True)
    print("[D04] Final weights:", current_weights, flush=True)
    print("[D04] Original experiment directory was not modified.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[D04] Interrupted by user. Check the active stage's weights/last.pt.", flush=True)
        sys.exit(130)
