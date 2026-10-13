"""
D04 three-stage training for detection-embedded enhancement.
Designed for the existing Windows project at F:\\D04.

Stages:
  1. Freeze D04EnhanceStem; train the detector.
  2. Freeze detector layers; train D04EnhanceStem only.
  3. Unfreeze all layers; joint fine-tuning.

This script never overwrites the original dee_yolov8s_v1 experiment.
"""
from pathlib import Path
import sys
import torch
from ultralytics import YOLO

ROOT = Path(r"F:\D04")
DATA = ROOT / "datasets" / "VisDroneYOLO" / "visdrone.yaml"
INITIAL_WEIGHTS = (
    ROOT / "experiments" / "detection_embedded" /
    "dee_yolov8s_v1" / "weights" / "best.pt"
)
PROJECT = ROOT / "experiments" / "detection_embedded" / "three_stage"

# Conservative settings for GTX 1660 Ti (6 GB class GPU).
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


def configure_freezing(trainer, mode):
    """Set gradient flags after the trainer has built the model."""
    model = trainer.model
    layers = list(model.model)

    

    if mode == "enhance":
        # Stage 1: freeze the enhancement stem.
        layers[0].requires_grad_(False)
    elif mode == "detector":
        # Stage 2: freeze all detector layers, leaving only the stem trainable.
        for layer in layers[1:]:
            layer.requires_grad_(False)
    elif mode != "none":
        raise ValueError(f"Unknown freeze mode: {mode}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"[D04] Freeze mode={mode}; trainable parameters={trainable:,}/{total:,}",
        flush=True,
    )
    if trainable == 0:
        raise RuntimeError("No trainable parameters remain.")


def enforce_frozen_eval(trainer, mode):
    """
    Called immediately before each training batch.
    Ultralytics switches the complete model to train mode each epoch, so
    re-apply eval() to frozen modules here to keep BatchNorm running stats fixed.
    """
    layers = list(trainer.model.model)
    if mode == "enhance":
        layers[0].eval()
    elif mode == "detector":
        for layer in layers[1:]:
            layer.eval()


def train_stage(weights_path, stage):
    if not DATA.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA}")
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Starting weights not found: {weights_path}")

    output_dir = PROJECT / stage["name"]
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists:\n{output_dir}\n"
            "To avoid overwriting previous results, rename/archive that folder "
            "or choose a new stage name before rerunning."
        )

    print("\n" + "=" * 72, flush=True)
    print(f"[D04] Starting {stage['name']}", flush=True)
    print(f"[D04] Initial weights: {weights_path}", flush=True)
    print(f"[D04] Output directory: {output_dir}", flush=True)
    print("=" * 72, flush=True)

    model = YOLO(str(weights_path))
    model.model.requires_grad_(True)

    # Important: pass the loaded model's weights into the newly constructed
    # training model. Do not set pretrained=False here, as that would discard
    # the loaded checkpoint weights in the normal Ultralytics training path.
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
        data=str(DATA),
        epochs=stage["epochs"],
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=str(PROJECT),
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
        amp=True,
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
        raise FileNotFoundError(
            f"Training ended but best.pt was not found where expected: {best}"
        )
    print(f"[D04] Stage best weights: {best}", flush=True)
    print(f"[D04] Stage last weights: {last if last.is_file() else 'not found'}", flush=True)
    return best


def main():
    smoke_test = "--smoke-test" in sys.argv
    print("[D04] PyTorch:", torch.__version__)
    print("[D04] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[D04] GPU:", torch.cuda.get_device_name(0))
    else:
        raise RuntimeError(
            "CUDA is not available. Activate the correct venv and verify PyTorch/CUDA."
        )

    # Fail early if the custom layer is not registered in this Python environment.
    from ultralytics.nn import tasks
    if not hasattr(tasks, "D04EnhanceStem"):
        raise RuntimeError(
            "D04EnhanceStem is not registered in ultralytics.nn.tasks in this process. "
            "Run the same environment check that successfully loaded best.pt, then "
            "register/import the custom module before training."
        )

    current_weights = INITIAL_WEIGHTS
    if smoke_test:
        # A separate one-epoch run checks the real data pipeline and callbacks
        # before committing to the full multi-stage run.
        smoke_stage = dict(STAGES[2])
        smoke_stage["name"] = "smoke_test_stage3_v1"
        smoke_stage["epochs"] = 1

        current_weights = (
            PROJECT / "smoke_test_stage2_v1" / "weights" / "best.pt"
        )

        print(
            "[D04] SMOKE TEST: Stage 3 only, 1 epoch; "
            "full stages will NOT start.",
            flush=True,
        )

        current_weights = train_stage(current_weights, smoke_stage)

        print("[D04] Stage 3 smoke test finished.", flush=True)
        print("[D04] Smoke-test weights:", current_weights, flush=True)
        return

    for stage in STAGES:
        current_weights = train_stage(current_weights, stage)

    print("\n[D04] All three stages completed.", flush=True)
    print("[D04] Final weights:", current_weights, flush=True)
    print("[D04] Original experiment directory was not modified.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[D04] Interrupted by user. Check the active stage's weights/last.pt.", flush=True)
        sys.exit(130)
