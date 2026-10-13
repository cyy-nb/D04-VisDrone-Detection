from ultralytics import YOLO
from pathlib import Path
import time
import json
import subprocess
import tempfile
import shutil
import torch


# ============================================================
# D04 Latency / FPS Unified Evaluation
# ============================================================
#
# MODE 可选：
#
# baseline
#     原始图片 → YOLO
#
# full_sr
#     Full Image SR图片 → YOLO
#
# roi_sr_cached
#     原图 → YOLO → 已经生成好的ROI-SR图片 → YOLO
#
# roi_sr_end_to_end
#     原图 → YOLO → ROI裁剪 → Real-ESRGAN ×2
#          → ROI YOLO
#
# ============================================================


# ============================================================
# 只需要修改这里
# ============================================================

MODE = "roi_sr_cached"


# ============================================================
# 基础配置
# ============================================================

MODEL_PATH = r"F:\D04\experiments\baseline_640\weights\best.pt"

IMG_SIZE = 640

WARMUP = 20

DEVICE = 0


# ============================================================
# Real-ESRGAN配置
# ============================================================

REALESRGAN_DIR = Path(
    r"F:\D04\super_resolution\Real-ESRGAN"
)

REALESRGAN_SCRIPT = (
    REALESRGAN_DIR /
    "inference_realesrgan.py"
)

REALESRGAN_MODEL = "RealESRGAN_x2plus"

REALESRGAN_TILE = "256"


# ============================================================
# 数据路径
# ============================================================

ORIGINAL_IMAGE_DIR = Path(
    r"F:\D04\datasets\VisDroneYOLO\images\val"
)

FULL_SR_IMAGE_DIR = Path(
    r"F:\D04\datasets\VisDroneYOLO_SR\images\val"
)

ROI_ROOT = Path(
    r"F:\D04\experiments\roi_sr_x2"
)

ROI_CROP_DIR = ROI_ROOT / "roi_crops"

ROI_SR_DIR = ROI_ROOT / "roi_sr"

ROI_METADATA = ROI_ROOT / "roi_metadata.json"


# ============================================================
# 输出目录
# ============================================================

OUTPUT_ROOT = Path(
    r"F:\D04\experiments\roi_sr_x2"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 根据MODE设置
# ============================================================

if MODE == "baseline":

    EXP_NAME = "baseline"

    IMAGE_DIR = ORIGINAL_IMAGE_DIR

    RESULT_FILE = (
        OUTPUT_ROOT /
        "baseline_latency_results.txt"
    )


elif MODE == "full_sr":

    EXP_NAME = "full_image_sr_x2"

    IMAGE_DIR = FULL_SR_IMAGE_DIR

    RESULT_FILE = (
        OUTPUT_ROOT /
        "full_sr_latency_results.txt"
    )


elif MODE == "roi_sr_cached":

    EXP_NAME = "roi_sr_cached"

    IMAGE_DIR = ORIGINAL_IMAGE_DIR

    RESULT_FILE = (
        OUTPUT_ROOT /
        "roi_sr_cached_latency_results.txt"
    )


elif MODE == "roi_sr_end_to_end":

    EXP_NAME = "roi_sr_end_to_end"

    IMAGE_DIR = ORIGINAL_IMAGE_DIR

    RESULT_FILE = (
        OUTPUT_ROOT /
        "roi_sr_end_to_end_latency_results.txt"
    )


else:

    raise ValueError(
        f"Unknown MODE: {MODE}"
    )


# ============================================================
# 加载YOLO
# ============================================================

print("=" * 70)

print(
    "D04 Latency / FPS Unified Evaluation"
)

print("=" * 70)

print()

print(
    f"Mode: {MODE}"
)

print(
    f"Experiment: {EXP_NAME}"
)

print()

print(
    "Loading YOLO model..."
)

model = YOLO(
    MODEL_PATH
)


# ============================================================
# 获取图片
# ============================================================

images = sorted(
    IMAGE_DIR.glob("*.jpg")
)

if len(images) == 0:

    raise RuntimeError(
        f"No JPG images found in:\n{IMAGE_DIR}"
    )


print(
    f"Images: {len(images)}"
)

print(
    f"Image size: {IMG_SIZE}"
)

print()


# ============================================================
# ROI模式加载metadata
# ============================================================

metadata = None

if MODE in [
    "roi_sr_cached",
    "roi_sr_end_to_end"
]:

    if not ROI_METADATA.exists():

        raise FileNotFoundError(
            f"ROI metadata not found:\n"
            f"{ROI_METADATA}"
        )

    with open(
        ROI_METADATA,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)


# ============================================================
# ROI-SR缓存图片映射
# ============================================================

sr_map = {}

if MODE == "roi_sr_cached":

    print(
        "Building ROI-SR image map..."
    )

    for p in ROI_SR_DIR.glob(
        "*_out.png"
    ):

        key = p.stem[:-4]

        sr_map[key] = p


    print(
        f"ROI-SR images found: "
        f"{len(sr_map)}"
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
        device=DEVICE,
        verbose=False
    )


if torch.cuda.is_available():

    torch.cuda.synchronize()


print(
    "Warm-up finished."
)

print()


# ============================================================
# ROI-SR Cached
# ============================================================

def run_roi_sr_cached(image_path):

    """
    使用已经生成好的ROI-SR图片。

    原图
      ↓
    YOLO第一次检测
      ↓
    已生成ROI-SR
      ↓
    ROI YOLO

    注意：
    不重新运行Real-ESRGAN。
    """

    records = metadata.get(
        image_path.name,
        []
    )


    # --------------------------------------------------------
    # 第一阶段：原图YOLO
    # --------------------------------------------------------

    original_result = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=0.001,
        device=DEVICE,
        verbose=False
    )


    # --------------------------------------------------------
    # 第二阶段：逐个ROI-SR YOLO
    #
    # 不再一次性送入几百个图片
    # --------------------------------------------------------

    roi_box_count = 0


    for record in records:

        roi_file = record["roi_file"]

        roi_key = Path(
            roi_file
        ).stem


        sr_path = sr_map.get(
            roi_key
        )


        if sr_path is None:

            continue


        roi_result = model.predict(
            source=str(sr_path),
            imgsz=IMG_SIZE,
            conf=0.001,
            device=DEVICE,
            verbose=False
        )


        for result in roi_result:

            if result.boxes is not None:

                roi_box_count += len(
                    result.boxes
                )


    return roi_box_count


# ============================================================
# ROI-SR End-to-End
# ============================================================

def run_roi_sr_end_to_end(
    image_path,
    temp_root
):

    """
    严格端到端ROI-SR：

    原始图片
        ↓
    YOLO第一次检测
        ↓
    根据小目标检测结果裁剪ROI
        ↓
    Real-ESRGAN ×2
        ↓
    ROI YOLO
        ↓
    返回

    注意：
    这个模式会非常慢。

    它真正把Real-ESRGAN时间算入Latency。
    """


    # --------------------------------------------------------
    # 第一阶段：原始图片YOLO
    # --------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=0.05,
        device=DEVICE,
        verbose=False
    )


    boxes = results[0].boxes


    # --------------------------------------------------------
    # 创建临时目录
    # --------------------------------------------------------

    image_temp_dir = (
        temp_root /
        image_path.stem
    )

    crop_dir = (
        image_temp_dir /
        "crops"
    )

    sr_dir = (
        image_temp_dir /
        "sr"
    )


    crop_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sr_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # ROI裁剪
    #
    # 与之前D04_generate_roi.py保持一致：
    #
    # area < 32×32
    # padding scale = 2.0
    # --------------------------------------------------------

    from PIL import Image

    image = Image.open(
        image_path
    ).convert("RGB")


    width, height = image.size


    roi_files = []


    for box_id, box in enumerate(boxes):

        xyxy = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )


        x1, y1, x2, y2 = xyxy

        bw = x2 - x1

        bh = y2 - y1

        area = bw * bh


        # 只处理小目标

        if area >= 32 ** 2:

            continue


        cx = (
            x1 + x2
        ) / 2

        cy = (
            y1 + y2
        ) / 2


        new_w = bw * 2.0

        new_h = bh * 2.0


        rx1 = max(
            0,
            int(
                cx - new_w / 2
            )
        )

        ry1 = max(
            0,
            int(
                cy - new_h / 2
            )
        )

        rx2 = min(
            width,
            int(
                cx + new_w / 2
            )
        )

        ry2 = min(
            height,
            int(
                cy + new_h / 2
            )
        )


        if (
            rx2 - rx1 < 16
            or
            ry2 - ry1 < 16
        ):

            continue


        roi = image.crop(
            (
                rx1,
                ry1,
                rx2,
                ry2
            )
        )


        roi_name = (
            f"{image_path.stem}"
            f"_roi_{box_id}.png"
        )


        roi_path = (
            crop_dir /
            roi_name
        )


        roi.save(
            roi_path
        )


        roi_files.append(
            roi_path
        )


    # --------------------------------------------------------
    # 没有ROI
    # --------------------------------------------------------

    if len(roi_files) == 0:

        return 0


    # --------------------------------------------------------
    # Real-ESRGAN ×2
    # --------------------------------------------------------

    cmd = [
        "python",
        str(REALESRGAN_SCRIPT),

        "-i",
        str(crop_dir),

        "-o",
        str(sr_dir),

        "-n",
        REALESRGAN_MODEL,

        "--outscale",
        "2",

        "--tile",
        REALESRGAN_TILE,

        "--half"
    ]


    subprocess.run(
        cmd,
        cwd=str(REALESRGAN_DIR),
        check=True
    )


    # --------------------------------------------------------
    # ROI-SR YOLO
    #
    # 逐张处理，避免内存爆炸
    # --------------------------------------------------------

    roi_box_count = 0


    for roi_path in sorted(
        sr_dir.glob("*_out.png")
    ):

        roi_result = model.predict(
            source=str(roi_path),
            imgsz=IMG_SIZE,
            conf=0.001,
            device=DEVICE,
            verbose=False
        )


        for result in roi_result:

            if result.boxes is not None:

                roi_box_count += len(
                    result.boxes
                )


    return roi_box_count


# ============================================================
# Measure
# ============================================================

times = []


print(
    "Measuring..."
)

print()


# ------------------------------------------------------------
# End-to-End模式临时目录
# ------------------------------------------------------------

temp_root = None


if MODE == "roi_sr_end_to_end":

    temp_root = Path(
        tempfile.mkdtemp(
            prefix="D04_roi_latency_"
        )
    )


try:

    for i, image_path in enumerate(
        images
    ):


        # ----------------------------------------------------
        # GPU同步
        # ----------------------------------------------------

        if torch.cuda.is_available():

            torch.cuda.synchronize()


        start = time.perf_counter()


        # ====================================================
        # Baseline
        # ====================================================

        if MODE == "baseline":

            model.predict(
                source=str(image_path),
                imgsz=IMG_SIZE,
                device=DEVICE,
                verbose=False
            )


        # ====================================================
        # Full Image SR
        # ====================================================

        elif MODE == "full_sr":

            model.predict(
                source=str(image_path),
                imgsz=IMG_SIZE,
                device=DEVICE,
                verbose=False
            )


        # ====================================================
        # ROI-SR Cached
        # ====================================================

        elif MODE == "roi_sr_cached":

            run_roi_sr_cached(
                image_path
            )


        # ====================================================
        # ROI-SR End-to-End
        # ====================================================

        elif MODE == "roi_sr_end_to_end":

            run_roi_sr_end_to_end(
                image_path,
                temp_root
            )


        # ----------------------------------------------------
        # GPU同步
        # ----------------------------------------------------

        if torch.cuda.is_available():

            torch.cuda.synchronize()


        end = time.perf_counter()


        # ----------------------------------------------------
        # Latency
        # ----------------------------------------------------

        latency_ms = (
            end - start
        ) * 1000


        times.append(
            latency_ms
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (i + 1) % 25 == 0
        ):

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
                f"{current_avg:.4f} ms/image | "
                f"FPS: {current_fps:.2f}"
            )


finally:

    # --------------------------------------------------------
    # 删除End-to-End临时文件
    # --------------------------------------------------------

    if temp_root is not None:

        try:

            shutil.rmtree(
                temp_root,
                ignore_errors=True
            )

        except Exception:

            pass


# ============================================================
# Calculate
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
# Save TXT
# ============================================================

result_text = f"""
D04 {EXP_NAME} Latency Evaluation
========================================

Mode:
{MODE}

Model:
{MODEL_PATH}

Image Directory:
{IMAGE_DIR}

Images:
{len(images)}

Image Size:
{IMG_SIZE}

Device:
CUDA:0

Average Latency:
{avg_latency:.4f} ms/image

FPS:
{fps:.2f} images/s
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
# Print
# ============================================================

print()

print(
    "=" * 70
)

print(
    "D04 Latency Result"
)

print(
    "=" * 70
)

print()

print(
    f"Experiment: {EXP_NAME}"
)

print(
    f"Average Latency: "
    f"{avg_latency:.4f} ms/image"
)

print(
    f"FPS: {fps:.2f}"
)

print()

print(
    "Result saved to:"
)

print(
    RESULT_FILE
)

print()

