from ultralytics import YOLO
from pathlib import Path
import json

# ==================================================
# D04 ROI-SR
# Restore ROI metadata from the ORIGINAL ROI rule
# 不重新生成 ROI 图片
# ==================================================

MODEL_PATH = r"F:\D04\experiments\baseline_640\weights\best.pt"

IMAGE_DIR = Path(
    r"F:\D04\datasets\VisDroneYOLO\images\val"
)

OUTPUT_DIR = Path(
    r"F:\D04\experiments\roi_sr_x2"
)

ROI_DIR = OUTPUT_DIR / "roi_crops"

METADATA_FILE = OUTPUT_DIR / "roi_metadata.json"


# --------------------------------------------------
# Parameters
# 必须与原始 D04_generate_roi.py 完全一致
# --------------------------------------------------

IMG_SIZE = 640
CONF = 0.05
SMALL_MAX = 32 ** 2
PADDING_SCALE = 2.0


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)


# --------------------------------------------------
# IMPORTANT:
# 和原始脚本完全一样，只读取 *.jpg
# --------------------------------------------------

images = sorted(
    IMAGE_DIR.glob("*.jpg")
)


print()
print("======================================")
print("D04 ROI-SR : Restore Metadata")
print("======================================")

print(f"Images: {len(images)}")
print(f"Confidence: {CONF}")
print(f"Small object max area: {SMALL_MAX}")
print(f"Padding scale: {PADDING_SCALE}")
print(f"Existing ROI PNG: {len(list(ROI_DIR.glob('*.png')))}")
print()


# --------------------------------------------------
# metadata
# --------------------------------------------------

metadata = {}

matched = 0
not_matched = 0

total_candidates = 0


# --------------------------------------------------
# Process images
# --------------------------------------------------

for index, image_path in enumerate(images):

    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF,
        device=0,
        verbose=False
    )

    boxes = results[0].boxes

    image_records = []


    # ------------------------------------------------
    # IMPORTANT:
    # box_id 必须使用原始 YOLO boxes 的编号
    # ------------------------------------------------

    for box_id, box in enumerate(boxes):

        xyxy = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = xyxy

        bw = x2 - x1
        bh = y2 - y1

        area = bw * bh


        # --------------------------------------------
        # 原始规则
        # --------------------------------------------

        if area >= SMALL_MAX:
            continue


        total_candidates += 1


        # --------------------------------------------
        # ROI expansion
        # --------------------------------------------

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        new_w = bw * PADDING_SCALE
        new_h = bh * PADDING_SCALE


        rx1 = max(
            0,
            int(cx - new_w / 2)
        )

        ry1 = max(
            0,
            int(cy - new_h / 2)
        )

        # --------------------------------------------
        # 获取原图尺寸
        # --------------------------------------------

        from PIL import Image

        with Image.open(image_path) as im:

            width, height = im.size


        rx2 = min(
            width,
            int(cx + new_w / 2)
        )

        ry2 = min(
            height,
            int(cy + new_h / 2)
        )


        # --------------------------------------------
        # 原始脚本的最小 ROI 限制
        # --------------------------------------------

        if rx2 - rx1 < 16 or ry2 - ry1 < 16:

            continue


        # --------------------------------------------
        # 关键：
        # 必须使用 box_id，而不是 global roi_counter
        # --------------------------------------------

        roi_name = (
            f"{image_path.stem}_roi_{box_id}.png"
        )


        # --------------------------------------------
        # 检查对应 ROI 图片是否已经存在
        # --------------------------------------------

        roi_path = ROI_DIR / roi_name


        if not roi_path.exists():

            not_matched += 1
            continue


        matched += 1


        cls = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )


        # --------------------------------------------
        # 保存 metadata
        # --------------------------------------------

        image_records.append({

            "roi_file": roi_name,

            "x1": rx1,
            "y1": ry1,
            "x2": rx2,
            "y2": ry2,

            "class": cls,

            "confidence": confidence,

            "original_box": [
                float(x1),
                float(y1),
                float(x2),
                float(y2)
            ]

        })


    metadata[image_path.name] = image_records


    if (index + 1) % 25 == 0:

        print(
            f"{index + 1}/{len(images)} images processed | "
            f"Matched ROIs: {matched}"
        )


# --------------------------------------------------
# Save metadata
# --------------------------------------------------

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False
    )


# --------------------------------------------------
# Final
# --------------------------------------------------

print()
print("======================================")
print("ROI metadata restore finished")
print("======================================")

print(f"Images: {len(images)}")

print(
    f"Existing ROI PNG: "
    f"{len(list(ROI_DIR.glob('*.png')))}"
)

print(
    f"Total ROI candidates: "
    f"{total_candidates}"
)

print(
    f"Matched existing ROIs: "
    f"{matched}"
)

print(
    f"Not matched: "
    f"{not_matched}"
)

print()

print(
    f"Metadata: "
    f"{METADATA_FILE}"
)