from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import json

# ==================================================
# D04 ROI-SR Step 1
# Generate ROI candidates from baseline YOLO
# ==================================================

MODEL_PATH = r"F:\D04\experiments\baseline_640\weights\best.pt"

IMAGE_DIR = Path(r"F:\D04\datasets\VisDroneYOLO\images\val")

OUTPUT_DIR = Path(r"F:\D04\experiments\roi_sr_x2")
ROI_DIR = OUTPUT_DIR / "roi_crops"

ROI_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Parameters
# --------------------------------------------------

IMG_SIZE = 640

# 为了尽可能减少漏掉的小目标，
# 第一次检测使用较低 confidence
CONF = 0.05

# 小目标面积阈值：32 × 32
SMALL_MAX = 32 ** 2

# ROI 扩大比例
# 例如原框 20×20，会扩大到约 40×40
PADDING_SCALE = 2.0

# --------------------------------------------------
# Load model
# --------------------------------------------------

model = YOLO(MODEL_PATH)

images = sorted(IMAGE_DIR.glob("*.jpg"))

print("======================================")
print("D04 ROI-SR : Generate ROI Candidates")
print("======================================")
print(f"Images: {len(images)}")
print(f"Confidence: {CONF}")
print(f"Small object max area: {SMALL_MAX}")
print()

metadata = {}

roi_count = 0

# --------------------------------------------------
# Process images
# --------------------------------------------------

for index, image_path in enumerate(images):

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    results = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF,
        device=0,
        verbose=False
    )

    boxes = results[0].boxes

    image_records = []

    for box_id, box in enumerate(boxes):

        xyxy = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = xyxy

        bw = x2 - x1
        bh = y2 - y1

        area = bw * bh

        # ------------------------------------------
        # Only keep small-object candidates
        # ------------------------------------------

        if area >= SMALL_MAX:
            continue

        # ------------------------------------------
        # Expand ROI
        # ------------------------------------------

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        new_w = bw * PADDING_SCALE
        new_h = bh * PADDING_SCALE

        rx1 = max(0, int(cx - new_w / 2))
        ry1 = max(0, int(cy - new_h / 2))

        rx2 = min(width, int(cx + new_w / 2))
        ry2 = min(height, int(cy + new_h / 2))

        # 避免 ROI 太小
        if rx2 - rx1 < 16 or ry2 - ry1 < 16:
            continue

        roi = image.crop((rx1, ry1, rx2, ry2))

        roi_name = f"{image_path.stem}_roi_{box_id}.png"

        roi_path = ROI_DIR / roi_name

        roi.save(roi_path)

        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        image_records.append({
            "roi_file": roi_name,
            "x1": rx1,
            "y1": ry1,
            "x2": rx2,
            "y2": ry2,
            "class": cls,
            "confidence": conf,
            "original_box": [
                float(x1),
                float(y1),
                float(x2),
                float(y2)
            ]
        })

        roi_count += 1

    metadata[image_path.name] = image_records

    if (index + 1) % 50 == 0:
        print(
            f"{index + 1}/{len(images)} images processed | "
            f"ROIs: {roi_count}"
        )

# --------------------------------------------------
# Save metadata
# --------------------------------------------------

metadata_file = OUTPUT_DIR / "roi_metadata.json"

with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print()
print("======================================")
print("ROI generation finished")
print("======================================")
print(f"Images: {len(images)}")
print(f"ROIs: {roi_count}")
print(f"ROI folder: {ROI_DIR}")
print(f"Metadata: {metadata_file}")