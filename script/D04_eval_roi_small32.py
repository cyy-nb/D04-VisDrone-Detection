import json
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from .d04_common import project_path
except ImportError:
    from d04_common import project_path


# =========================================================
# 路径
# =========================================================

IMAGE_DIR = Path(
    project_path('datasets', 'VisDroneYOLO', 'images', 'val')
)

LABEL_DIR = Path(
    project_path('datasets', 'VisDroneYOLO', 'labels', 'val')
)

PRED_FILE = Path(
    project_path('experiments', 'roi_sr_x2', 'roi_fused_predictions_small32.json')
)

OUTPUT_DIR = Path(
    project_path('experiments', 'roi_sr_x2')
)

GT_FILE = OUTPUT_DIR / "small32_eval_ground_truth.json"
COCO_PRED_FILE = OUTPUT_DIR / "small32_eval_predictions_coco.json"
RESULT_FILE = OUTPUT_DIR / "small32_eval_results.txt"


CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor"
]


# =========================================================
# 检查
# =========================================================

print("=" * 70)
print("D04 ROI-SR Small32 Filter Evaluation")
print("=" * 70)

if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"找不到图片目录:\n{IMAGE_DIR}")

if not LABEL_DIR.exists():
    raise FileNotFoundError(f"找不到标签目录:\n{LABEL_DIR}")

if not PRED_FILE.exists():
    raise FileNotFoundError(f"找不到预测文件:\n{PRED_FILE}")


# =========================================================
# 获取验证图片
# =========================================================

images = sorted(IMAGE_DIR.glob("*.jpg"))

print(f"验证图片数量: {len(images)}")

if len(images) != 548:
    print(
        f"警告：当前检测到 {len(images)} 张图片，"
        f"预期为 548 张。"
    )


# =========================================================
# 创建 COCO Ground Truth
# =========================================================

coco_gt = {
    "images": [],
    "annotations": [],
    "categories": []
}


for class_id, class_name in enumerate(CLASS_NAMES):

    coco_gt["categories"].append({
        "id": class_id,
        "name": class_name
    })


annotation_id = 1


for image_id, image_path in enumerate(images, start=1):

    with Image.open(image_path) as im:
        width, height = im.size

    coco_gt["images"].append({
        "id": image_id,
        "file_name": image_path.name,
        "width": width,
        "height": height
    })

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            continue

        cls = int(parts[0])

        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        # YOLO normalized coordinates
        x = (cx - w / 2) * width
        y = (cy - h / 2) * height

        box_w = w * width
        box_h = h * height

        area = box_w * box_h

        coco_gt["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": cls,
            "bbox": [
                x,
                y,
                box_w,
                box_h
            ],
            "area": area,
            "iscrowd": 0
        })

        annotation_id += 1


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    GT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        coco_gt,
        f,
        indent=2,
        ensure_ascii=False
    )


print(f"GT 已生成: {GT_FILE}")
print(
    f"GT 数量: "
    f"{len(coco_gt['annotations'])}"
)


# =========================================================
# 读取筛选后的预测
#
# 实际结构：
#
# [
#   {
#       "image": "...jpg",
#       "predictions": [
#           {
#               "bbox": [x1,y1,x2,y2],
#               "score": ...,
#               "class": ...
#           }
#       ]
#   },
#   ...
# ]
# =========================================================

with open(
    PRED_FILE,
    "r",
    encoding="utf-8"
) as f:

    prediction_data = json.load(f)


if not isinstance(prediction_data, list):
    raise ValueError(
        "预测 JSON 最外层不是 list。"
    )


print(
    f"预测 JSON 图片记录: "
    f"{len(prediction_data)}"
)


# =========================================================
# 建立图片名 -> image_id
# =========================================================

image_id_map = {
    image_path.name: index + 1
    for index, image_path in enumerate(images)
}


# =========================================================
# 转换 COCO Predictions
# =========================================================

coco_predictions = []

prediction_count = 0

missing_images = 0


for image_record in prediction_data:

    if not isinstance(image_record, dict):
        continue

    image_name = image_record.get("image")

    predictions = image_record.get(
        "predictions",
        []
    )

    if image_name not in image_id_map:

        missing_images += 1
        continue

    image_id = image_id_map[image_name]

    if not isinstance(predictions, list):
        continue

    for pred in predictions:

        if not isinstance(pred, dict):
            continue

        bbox = pred.get("bbox")

        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
        ):
            continue

        try:
            x1 = float(bbox[0])
            y1 = float(bbox[1])
            x2 = float(bbox[2])
            y2 = float(bbox[3])

            score = float(
                pred.get("score", 0.0)
            )

            cls = int(
                pred.get("class", 0)
            )

        except Exception:
            continue

        box_w = max(
            0.0,
            x2 - x1
        )

        box_h = max(
            0.0,
            y2 - y1
        )

        if box_w <= 0 or box_h <= 0:
            continue

        coco_predictions.append({
            "image_id": image_id,
            "category_id": cls,
            "bbox": [
                x1,
                y1,
                box_w,
                box_h
            ],
            "score": score
        })

        prediction_count += 1


print(
    f"成功转换预测框: "
    f"{prediction_count}"
)

print(
    f"未匹配图片: "
    f"{missing_images}"
)


# =========================================================
# 保存 COCO Predictions
# =========================================================

with open(
    COCO_PRED_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        coco_predictions,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    f"COCO 预测已生成: "
    f"{COCO_PRED_FILE}"
)


# =========================================================
# COCO Evaluation
# =========================================================

print()
print("=" * 70)
print("开始 COCO Evaluation")
print("=" * 70)


if prediction_count == 0:
    raise RuntimeError(
        "预测框数量为 0，请检查 JSON 图片名称是否匹配。"
    )


coco = COCO(
    str(GT_FILE)
)

coco_dt = coco.loadRes(
    str(COCO_PRED_FILE)
)


evaluator = COCOeval(
    coco,
    coco_dt,
    "bbox"
)


evaluator.params.imgIds = list(
    range(1, len(images) + 1)
)


evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()


# =========================================================
# 提取指标
# =========================================================

stats = evaluator.stats

AP = float(stats[0])
AP50 = float(stats[1])
AP75 = float(stats[2])
AP_s = float(stats[3])
AP_m = float(stats[4])
AP_l = float(stats[5])


# =========================================================
# 保存结果
# =========================================================

with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "D04 ROI-SR Small32 Filter Evaluation\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Prediction file: {PRED_FILE}\n"
    )

    f.write(
        f"Images: {len(images)}\n"
    )


