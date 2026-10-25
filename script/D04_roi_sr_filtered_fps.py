from ultralytics import YOLO
from pathlib import Path
import time
import json
import torch

try:
    from .d04_common import project_path
except ImportError:
    from d04_common import project_path


# ============================================================
# D04 ROI-SR Filtered FPS Evaluation
# ============================================================
#
# 流程：
#
# 原图
#   ↓
# 第一次 YOLO
#   ↓
# 已生成的 ROI-SR 图片
#   ↓
# ROI YOLO
#   ↓
# 只统计 < 32×32 的检测框
#
# 注意：
# 这里不重新运行 Real-ESRGAN。
# 使用已经生成好的 ROI-SR 图片。
#
# ============================================================


# ============================================================
# 基础配置
# ============================================================

MODEL_PATH = project_path('experiments', 'baseline_640', 'weights', 'best.pt')

IMG_SIZE = 640

WARMUP = 20

DEVICE = 0

CONF = 0.001

SMALL_MAX = 32 ** 2


# ============================================================
# 数据路径
# ============================================================

ORIGINAL_IMAGE_DIR = Path(
    project_path('datasets', 'VisDroneYOLO', 'images', 'val')
)

ROI_ROOT = Path(
    project_path('experiments', 'roi_sr_x2')
)

ROI_SR_DIR = ROI_ROOT / "roi_sr"

ROI_METADATA = ROI_ROOT / "roi_metadata.json"


# ============================================================
# 输出
# ============================================================

RESULT_FILE = (
    ROI_ROOT /
    "roi_sr_filtered_fps_results.txt"
)


# ============================================================
# 加载模型
# ============================================================

print("=" * 70)
print("D04 ROI-SR Filtered FPS Evaluation")
print("=" * 70)
print()

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")
print()


# ============================================================
# 获取原始图片
# ============================================================

images = sorted(
    ORIGINAL_IMAGE_DIR.glob("*.jpg")
)

if len(images) == 0:
    raise RuntimeError(
        f"No JPG images found in:\n{ORIGINAL_IMAGE_DIR}"
    )

print(f"Images: {len(images)}")
print(f"Image size: {IMG_SIZE}")
print(f"Small object threshold: < {SMALL_MAX} pixels")
print()


# ============================================================
# 加载 ROI metadata
# ============================================================

if not ROI_METADATA.exists():
    raise FileNotFoundError(
        f"ROI metadata not found:\n{ROI_METADATA}"
    )

with open(
    ROI_METADATA,
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

print(
    f"Metadata images: {len(metadata)}"
)
print()


# ============================================================
# 建立 ROI-SR 图片映射
# ============================================================

print("Building ROI-SR image map...")

sr_map = {}

for p in ROI_SR_DIR.glob("*_out.png"):

    key = p.stem[:-4]

    sr_map[key] = p

print(
    f"ROI-SR images found: {len(sr_map)}"
)

print()


# ============================================================
# GPU Warm-up
# ============================================================

print(
    f"Warming up GPU ({WARMUP} images)..."
)

for image_path in images[:WARMUP]:

    model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF,
        device=DEVICE,
        verbose=False
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()

print("Warm-up finished.")
print()


# ============================================================
# 正式测试
# ============================================================

times = []

total_original_boxes = 0

total_roi_boxes = 0

total_filtered_small_boxes = 0

total_missing_roi = 0


print("Starting FPS measurement...")
print()


for i, image_path in enumerate(images):

    # --------------------------------------------------------
    # GPU同步
    # --------------------------------------------------------

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()


    # ========================================================
    # 第一阶段：原图 YOLO
    # ========================================================

    original_result = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF,
        device=DEVICE,
        verbose=False
    )

    original_boxes = original_result[0].boxes

    if original_boxes is not None:
        total_original_boxes += len(
            original_boxes
        )


    # ========================================================
    # 获取该图片对应的 ROI
    # ========================================================

    records = metadata.get(
        image_path.name,
        []
    )


    # ========================================================
    # 第二阶段：ROI-SR YOLO
    # ========================================================

    for record in records:

        roi_file = record["roi_file"]

        roi_key = Path(
            roi_file
        ).stem

        sr_path = sr_map.get(
            roi_key
        )

        # ----------------------------------------------------
        # ROI-SR图片不存在
        # ----------------------------------------------------

        if sr_path is None:

            total_missing_roi += 1

            continue


        # ----------------------------------------------------
        # ROI YOLO
        # ----------------------------------------------------

        roi_result = model.predict(
            source=str(sr_path),
            imgsz=IMG_SIZE,
            conf=CONF,
            device=DEVICE,
            verbose=False
        )


        # ----------------------------------------------------
        # 统计检测框
        # ----------------------------------------------------

        for result in roi_result:

            if result.boxes is None:
                continue

            boxes = result.boxes

            total_roi_boxes += len(boxes)


            # =================================================
            # 筛选 < 32×32
            # =================================================

            xyxy = (
                boxes.xyxy
                .cpu()
                .numpy()
            )

            for box in xyxy:

                x1, y1, x2, y2 = box

                width = x2 - x1

                height = y2 - y1

                area = width * height


                if area < SMALL_MAX:

                    total_filtered_small_boxes += 1


    # ========================================================
    # GPU同步
    # ========================================================

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()


    # ========================================================
    # 单张图片Latency
    # ========================================================

    latency_ms = (
        end - start
    ) * 1000

    times.append(
        latency_ms
    )


    # ========================================================
    # 当前FPS
    # ========================================================

    if (i + 1) % 10 == 0:

        current_avg = (
            sum(times) /
            len(times)
        )

        current_fps = (
            1000 /
            current_avg
        )

        print(
            f"{i + 1}/{len(images)} | "
            f"Average: "
            f"{current_avg:.2f} ms/image | "
            f"FPS: "
            f"{current_fps:.2f} | "
            f"Small boxes: "
            f"{total_filtered_small_boxes}"
        )


# ============================================================
# 最终计算
# ============================================================

avg_latency = (
    sum(times) /
    len(times)
)

fps = (
    1000 /
    avg_latency
)


# ============================================================
# 保存结果
# ============================================================

result_text = f"""
D04 ROI-SR Filtered FPS Evaluation

========================================

Mode:
roi_sr_cached_filtered

Model:
{MODEL_PATH}

Original Image Directory:
{ORIGINAL_IMAGE_DIR}

ROI-SR Directory:
{ROI_SR_DIR}

Images:
{len(images)}

Image Size:
{IMG_SIZE}

Device:
CUDA:0

Confidence:
{CONF}

Small Object Definition:
area < 32x32

Average Latency:
{avg_latency:.4f} ms/image

FPS:
{fps:.2f} images/s

Original YOLO Boxes:
{total_original_boxes}

ROI-SR YOLO Boxes:
{total_roi_boxes}

Filtered Small Boxes:
{total_filtered_small_boxes}

Missing ROI-SR Images:
{total_missing_roi}

========================================
"""


with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        result_text
    )


# ============================================================
# 输出结果
# ============================================================

print()
print("=" * 70)
print("D04 ROI-SR Filtered FPS Result")
print("=" * 70)
print()

print(
    "Experiment: roi_sr_cached_filtered"
)

print(
    f"Average Latency: "
    f"{avg_latency:.4f} ms/image"
)

print(
    f"FPS: "
    f"{fps:.2f} images/s"
)

print()

print(
    f"Original YOLO Boxes: "
    f"{total_original_boxes}"
)

print(
    f"ROI-SR YOLO Boxes: "
    f"{total_roi_boxes}"
)

print(
    f"Filtered <32x32 Boxes: "
    f"{total_filtered_small_boxes}"
)

print()

print(
    "Result saved to:"
)

print(
    RESULT_FILE
)

print()

