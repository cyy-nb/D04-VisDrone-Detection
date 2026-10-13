import json
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# =========================================================
# 路径
# =========================================================

IMAGE_DIR = Path(
    r"F:\D04\datasets\VisDroneYOLO\images\val"
)

LABEL_DIR = Path(
    r"F:\D04\datasets\VisDroneYOLO\labels\val"
)

PRED_FILE = Path(
    r"F:\D04\experiments\roi_sr_x2\roi_fused_predictions.json"
)

OUTPUT_DIR = Path(
    r"F:\D04\experiments\roi_sr_x2"
)

GT_FILE = OUTPUT_DIR / "strict_small_gt.json"
PRED_COCO_FILE = OUTPUT_DIR / "strict_small_predictions.json"
RESULT_FILE = OUTPUT_DIR / "strict_small_ap_results.txt"


# =========================================================
# D04 小目标定义
# =========================================================

SMALL_MAX = 32 * 32


# =========================================================
# VisDrone 类别
# =========================================================

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


print("=" * 70)
print("D04 ROI-SR Strict Small Object Evaluation")
print("=" * 70)

print(f"Small object definition: GT area < {SMALL_MAX}")
print()


# =========================================================
# 读取图片
# =========================================================

images = sorted(
    IMAGE_DIR.glob("*.jpg")
)

print(
    f"验证图片数量: {len(images)}"
)


# =========================================================
# 创建 COCO GT
# 只保留 GT 面积 < 32×32
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
small_gt_count = 0
all_gt_count = 0


for image_id, image_path in enumerate(
    images,
    start=1
):

    with Image.open(image_path) as im:
        img_w, img_h = im.size

    coco_gt["images"].append({
        "id": image_id,
        "file_name": image_path.name,
        "width": img_w,
        "height": img_h
    })

    label_path = (
        LABEL_DIR /
        f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        continue

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) < 5:
            continue

        cls = int(parts[0])

        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        box_w = w * img_w
        box_h = h * img_h

        area = box_w * box_h

        all_gt_count += 1

        # =================================================
        # 只保留真正的小目标
        # =================================================

        if area >= SMALL_MAX:
            continue

        x = (
            cx * img_w
            - box_w / 2
        )

        y = (
            cy * img_h
            - box_h / 2
        )

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
        small_gt_count += 1


print()
print("=" * 70)
print("GT 统计")
print("=" * 70)

print(
    f"全部 GT       : {all_gt_count}"
)

print(
    f"GT 小目标     : {small_gt_count}"
)

print(
    f"小目标比例    : "
    f"{small_gt_count / all_gt_count * 100:.2f}%"
)


# =========================================================
# 保存 GT
# =========================================================

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


# =========================================================
# 读取 ROI-SR 融合预测
# =========================================================

with open(
    PRED_FILE,
    "r",
    encoding="utf-8"
) as f:

    prediction_data = json.load(f)


print()
print(
    f"预测图片记录: "
    f"{len(prediction_data)}"
)


# =========================================================
# 图片名称 → image_id
# =========================================================

image_id_map = {
    image_path.name: index + 1
    for index, image_path
    in enumerate(images)
}


# =========================================================
# 转 COCO prediction
#
# 注意：
# 这里不筛预测框面积。
#
# 因为真正的小目标 AP 应该由 GT
# 的面积定义，而不是预测框面积定义。
# =========================================================

coco_predictions = []

prediction_count = 0


for image_record in prediction_data:

    if not isinstance(
        image_record,
        dict
    ):
        continue

    image_name = image_record.get(
        "image"
    )

    predictions = image_record.get(
        "predictions",
        []
    )

    if image_name not in image_id_map:
        continue

    image_id = image_id_map[
        image_name
    ]

    for pred in predictions:

        if not isinstance(
            pred,
            dict
        ):
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
                pred.get(
                    "score",
                    0
                )
            )

            cls = int(
                pred.get(
                    "class",
                    0
                )
            )

        except Exception:
            continue

        box_w = x2 - x1
        box_h = y2 - y1

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
    f"COCO 预测框数量: "
    f"{prediction_count}"
)


# =========================================================
# 保存预测
# =========================================================

with open(
    PRED_COCO_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        coco_predictions,
        f,
        indent=2,
        ensure_ascii=False
    )


# =========================================================
# COCO Evaluation
# =========================================================

print()
print("=" * 70)
print("开始严格小目标 COCO Evaluation")
print("=" * 70)


coco = COCO(
    str(GT_FILE)
)

coco_dt = coco.loadRes(
    str(PRED_COCO_FILE)
)


evaluator = COCOeval(
    coco,
    coco_dt,
    "bbox"
)

evaluator.params.imgIds = [
    i
    for i in range(
        1,
        len(images) + 1
    )
]


evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()


# =========================================================
# 保存结果
# =========================================================

stats = evaluator.stats

AP = float(stats[0])
AP50 = float(stats[1])
AP75 = float(stats[2])
AP_s = float(stats[3])
AP_m = float(stats[4])
AP_l = float(stats[5])


with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "D04 ROI-SR Strict Small Object Evaluation\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        "Definition: GT area < 32x32\n\n"
    )

    f.write(
        f"All GT: {all_gt_count}\n"
    )

    f.write(
        f"Small GT: {small_gt_count}\n\n"
    )

    f.write(
        f"AP       : {AP:.6f}\n"
    )

    f.write(
        f"AP50     : {AP50:.6f}\n"
    )

    f.write(
        f"AP75     : {AP75:.6f}\n"
    )

    f.write(
        f"AP_s     : {AP_s:.6f}\n"
    )

    f.write(
        f"AP_m     : {AP_m:.6f}\n"
    )

    f.write(
        f"AP_l     : {AP_l:.6f}\n"
    )


print()
print("=" * 70)
print("严格小目标评估完成")
print("=" * 70)

print(
    f"AP       : {AP:.6f}"
)

print(
    f"AP50     : {AP50:.6f}"
)

print(
    f"AP75     : {AP75:.6f}"
)

print(
    f"AP_s     : {AP_s:.6f}"
)

print(
    f"AP_m     : {AP_m:.6f}"
)

print(
    f"AP_l     : {AP_l:.6f}"
)

print()
print(
    f"结果文件: {RESULT_FILE}"
)

print("=" * 70)

