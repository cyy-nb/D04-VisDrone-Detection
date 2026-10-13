
# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import time
import gc
import statistics

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ==================== 统一配置 ====================
ROOT = Path(r"F:\D04")

MODELS = {
    "Baseline": ROOT / "experiments" / "baseline_640" /
                "weights" / "best.pt",

    "D04_Detection_Embedded": ROOT / "experiments" /
                "detection_embedded" / "dee_yolov8s_v1" /
                "weights" / "best.pt",
}

IMAGE_DIR = ROOT / "datasets" / "VisDroneYOLO" / "images" / "val"
OUTPUT_DIR = ROOT / "experiments" / "latency_comparison"

IMG_SIZE = 640
DEVICE = "0" if torch.cuda.is_available() else "cpu"
WARMUP = 20
CONF = 0.25
IOU = 0.7


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_images():
    # 只读取 jpg，避免 Windows 下大小写通配符重复计数
    paths = sorted(IMAGE_DIR.glob("*.jpg"))
    if not paths:
        paths = sorted(IMAGE_DIR.glob("*.JPG"))

    if not paths:
        raise RuntimeError(f"没有找到验证图片：{IMAGE_DIR}")

    images = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is not None:
            images.append((p.name, im))

    print(f"成功读取验证图片：{len(images)} 张")
    if len(images) != 548:
        print("警告：图片数量不是预期的 548 张，请检查验证集。")
    return images


def make_tensor(im, model):
    # 固定输入为 640×640，预处理在纯模型计时之外
    im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    arr = np.ascontiguousarray(im.transpose(2, 0, 1))

    p = next(model.parameters())
    x = torch.from_numpy(arr).unsqueeze(0)
    x = x.to(device=p.device, dtype=p.dtype)
    x = x.float().div_(255.0)
    return x


def stats(times):
    avg = statistics.mean(times)
    return {
        "avg": avg,
        "median": statistics.median(times),
        "p95": float(np.percentile(times, 95)),
        "fps": 1000.0 / avg,
    }


def pure_forward(yolo, images):
    model = yolo.model.eval()
    tensors = [make_tensor(im, model) for _, im in images]

    with torch.inference_mode():
        # 预热
        for i in range(WARMUP):
            model(tensors[i % len(tensors)])
        sync()

        times = []
        for x in tensors:
            sync()
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t0) * 1000)

    return stats(times)


def end_to_end(yolo, images):
    # 计时包含 Ultralytics 的预处理、模型推理、后处理
    with torch.inference_mode():
        for i in range(WARMUP):
            yolo.predict(
                source=images[i % len(images)][1],
                imgsz=IMG_SIZE,
                device=DEVICE,
                conf=CONF,
                iou=IOU,
                verbose=False,
                save=False
            )
        sync()

        times = []
        for _, im in images:
            sync()
            t0 = time.perf_counter()
            yolo.predict(
                source=im,
                imgsz=IMG_SIZE,
                device=DEVICE,
                conf=CONF,
                iou=IOU,
                verbose=False,
                save=False
            )
            sync()
            times.append((time.perf_counter() - t0) * 1000)

    return stats(times)


def format_stats(name, result):
    return (
        f"{name}\n"
        f"  Average latency : {result['avg']:.3f} ms/image\n"
        f"  Median latency  : {result['median']:.3f} ms/image\n"
        f"  P95 latency     : {result['p95']:.3f} ms/image\n"
        f"  FPS             : {result['fps']:.3f}\n"
    )


def main():
    if not torch.cuda.is_available():
        print("警告：CUDA 不可用，本次将在 CPU 上测试。")

    for name, path in MODELS.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} 权重不存在：{path}")

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"验证集目录不存在：{IMAGE_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = load_images()

    print(f"设备：{DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU：{torch.cuda.get_device_name(0)}")
    print(f"输入尺寸：{IMG_SIZE}")
    print(f"每种测试的预热次数：{WARMUP}")

    results = {}

    for name, path in MODELS.items():
        print(f"\n正在测试：{name}")
        print(f"权重：{path}")

        yolo = YOLO(str(path))

        pure = pure_forward(yolo, images)
        print(format_stats("纯模型推理", pure))

        e2e = end_to_end(yolo, images)
        print(format_stats("端到端推理", e2e))

        results[name] = {
            "pure": pure,
            "e2e": e2e,
            "path": str(path),
        }

        del yolo
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base = results["Baseline"]
    d04 = results["D04_Detection_Embedded"]

    pure_ratio = d04["pure"]["avg"] / base["pure"]["avg"]
    e2e_ratio = d04["e2e"]["avg"] / base["e2e"]["avg"]

    report = [
        "D04 BASELINE VS DETECTION-EMBEDDED LATENCY",
        "=" * 65,
        f"Time: {datetime.now().isoformat(timespec='seconds')}",
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}",
        f"PyTorch: {torch.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
        f"Input size: {IMG_SIZE}x{IMG_SIZE}",
        f"Validation images: {len(images)}",
        f"Warm-up runs per method/model: {WARMUP}",
        f"Baseline weights: {base['path']}",
        f"D04 weights: {d04['path']}",
        "",
    ]

    for name in ("Baseline", "D04_Detection_Embedded"):
        r = results[name]
        report.append(f"MODEL: {name}")
        report.append(format_stats("Pure model forward", r["pure"]))
        report.append(format_stats("End-to-end prediction", r["e2e"]))
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
    output_file = OUTPUT_DIR / "D04_latency_fps_results.txt"
    output_file.write_text(text, encoding="utf-8")

    print("\n" + text)
    print(f"\n结果已保存：{output_file}")


if __name__ == "__main__":
    main()