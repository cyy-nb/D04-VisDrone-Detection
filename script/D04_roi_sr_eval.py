from pathlib import Path
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

try:
    from .d04_common import project_path
except ImportError:
    from d04_common import project_path


# ============================================================
# 路径
# ============================================================

ROOT = Path(project_path())

IMAGE_DIR = (
    ROOT
    / "datasets"
    / "VisDroneYOLO"
    / "images"
    / "val"
)

LABEL_DIR = (
    ROOT
    / "datasets"
    / "VisDroneYOLO"
    / "labels"
    / "val"
)

PREDICTION_FILE = (
    ROOT
    / "experiments"
    / "roi_sr_x2"
    / "roi_fused_predictions.json"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "roi_sr_x2"
)

RESULT_FILE = (
    OUTPUT_DIR
    / "roi_sr_ap_results.txt"
)

GT_FILE = (
    OUTPUT_DIR
    / "roi_sr_ground_truth.json"
)

PRED_COCO_FILE = (
    OUTPUT_DIR
    / "roi_sr_predictions_coco.json"
)


# ============================================================
# VisDrone 类别
# ============================================================

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


# ============================================================
# 创建输出目录
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 开始
# ============================================================

print("=" * 70)
print("D04 ROI-SR AP Evaluation")
print("=" * 70)
print()


# ============================================================
# 检查文件
# ============================================================

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGE_DIR}"
    )

if not LABEL_DIR.exists():
    raise FileNotFoundError(
        f"Label directory not found:\n{LABEL_DIR}"
    )

if not PREDICTION_FILE.exists():
    raise FileNotFoundError(
        f"Prediction file not found:\n{PREDICTION_FILE}"
    )


# ============================================================
# 获取 548 张验证图片
# ============================================================

image_files = sorted(
    IMAGE_DIR.glob("*.jpg")
)

print(
    "Validation images:",
    len(image_files)
)

if len(image_files) == 0:
    raise RuntimeError(
        "No JPG images found."
    )

print()


# ============================================================
# 建立 image -> COCO image_id
# ============================================================

image_id_map = {}

images = []

for image_id, image_path in enumerate(
    image_files,
    start=1
):

    image_id_map[
        image_path.name
    ] = image_id

    # 获取图片尺寸
    from PIL import Image

    with Image.open(image_path) as img:
        width, height = img.size

    images.append({
        "id": image_id,
        "file_name": image_path.name,
        "width": width,
        "height": height
    })


# ============================================================
# COCO categories
# ============================================================

categories = []

for class_id, class_name in enumerate(
    CLASS_NAMES
):

    categories.append({
        "id": class_id + 1,
        "name": class_name
    })


# ============================================================
# YOLO label → COCO GT
# ============================================================

annotations = []

annotation_id = 1


for image_path in image_files:

    image_name = image_path.name

    image_id = image_id_map[
        image_name
    ]

    label_path = (
        LABEL_DIR
        / f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        continue


    # 图片尺寸
    image_info = images[
        image_id - 1
    ]

    width = image_info["width"]
    height = image_info["height"]


    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = [
            line.strip()
            for line in f
            if line.strip()
        ]


    for line in lines:

        parts = line.split()

        if len(parts) < 5:
            continue


        class_id = int(
            float(parts[0])
        )

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])


        # YOLO normalized
        # → pixel coordinates

        bw = box_width * width
        bh = box_height * height

        cx = x_center * width
        cy = y_center * height

        x = cx - bw / 2
        y = cy - bh / 2


        # 限制在图片范围内

        x = max(
            0.0,
            min(x, width)
        )

        y = max(
            0.0,
            min(y, height)
        )

        bw = min(
            bw,
            width - x
        )

        bh = min(
            bh,
            height - y
        )


        if bw <= 0 or bh <= 0:
            continue


        annotations.append({

            "id": annotation_id,

            "image_id": image_id,

            "category_id": class_id + 1,

            "bbox": [
                x,
                y,
                bw,
                bh
            ],

            "area": bw * bh,

            "iscrowd": 0

        })


        annotation_id += 1


# ============================================================
# 保存 COCO Ground Truth
# ============================================================

ground_truth = {

    "images": images,

    "annotations": annotations,

    "categories": categories

}


with open(
    GT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        ground_truth,
        f,
        indent=2,
        ensure_ascii=False
    )


print(
    "Ground truth annotations:",
    len(annotations)
)

print(
    "Ground truth:",
    GT_FILE
)

print()


# ============================================================
# 读取 ROI-SR 融合预测
# ============================================================

print("Loading ROI-SR predictions...")

with open(
    PREDICTION_FILE,
    "r",
    encoding="utf-8"
) as f:

    predictions = json.load(f)


print(
    "Prediction images:",
    len(predictions)
)

print()


# ============================================================
# 转换成 COCO prediction 格式
# ============================================================

coco_predictions = []

prediction_id = 1


for image_record in predictions:

    image_name = image_record[
        "image"
    ]

    if image_name not in image_id_map:
        continue


    image_id = image_id_map[
        image_name
    ]


    for pred in image_record.get(
        "predictions",
        []
    ):

        bbox = pred["bbox"]

        if len(bbox) != 4:
            continue


        x1 = float(bbox[0])
        y1 = float(bbox[1])
        x2 = float(bbox[2])
        y2 = float(bbox[3])


        # x1,y1,x2,y2
        # →
        # x,y,width,height

        bw = x2 - x1
        bh = y2 - y1


        if bw <= 0 or bh <= 0:
            continue


        score = float(
            pred["score"]
        )

        class_id = int(
            pred["class"]
        )


        if class_id < 0 or class_id >= len(
            CLASS_NAMES
        ):
            continue


        coco_predictions.append({

            "image_id": image_id,

            "category_id": class_id + 1,

            "bbox": [
                x1,
                y1,
                bw,
                bh
            ],

            "score": score

        })


    prediction_id += 1


# ============================================================
# 保存 COCO Prediction
# ============================================================

with open(
    PRED_COCO_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        coco_predictions,
        f,
        indent=2
    )


print(
    "COCO predictions:",
    len(coco_predictions)
)

print(
    "Predictions:",
    PRED_COCO_FILE
)

print()


# ============================================================
# COCO Evaluation
# ============================================================

print("=" * 70)
print("Running COCO evaluation...")
print("=" * 70)
print()


coco_gt = COCO(
    str(GT_FILE)
)

coco_dt = coco_gt.loadRes(
    str(PRED_COCO_FILE)
)


coco_eval = COCOeval(
    coco_gt,
    coco_dt,
    "bbox"
)


coco_eval.params.imgIds = [
    image_id_map[
        image_path.name
    ]
    for image_path in image_files
]


coco_eval.evaluate()

coco_eval.accumulate()

coco_eval.summarize()


# ============================================================
# 提取标准 COCO 指标
#
# stats:
#
# 0 = mAP 50:95
# 1 = mAP 50
# 2 = mAP 75
# 3 = AP small
# 4 = AP medium
# 5 = AP large
# ============================================================

AP = float(
    coco_eval.stats[0]
)

AP50 = float(
    coco_eval.stats[1]
)

AP75 = float(
    coco_eval.stats[2]
)

AP_s = float(
    coco_eval.stats[3]
)

AP_m = float(
    coco_eval.stats[4]
)

AP_l = float(
    coco_eval.stats[5]
)


# ============================================================
# 保存结果
# ============================================================

result_text = f"""
D04 ROI-SR x2 AP Evaluation
============================

Images:
{len(image_files)}

Ground Truth annotations:
{len(annotations)}

COCO predictions:
{len(coco_predictions)}

AP (mAP50-95):
{AP:.6f}

AP50:
{AP50:.6f}

AP75:
{AP75:.6f}

AP_s:
{AP_s:.6f}

AP_m:
{AP_m:.6f}

AP_l:
{AP_l:.6f}
""".strip()


with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        result_text
    )


# ============================================================
# 最终输出
# ============================================================

print()
print("=" * 70)
print("ROI-SR AP Evaluation Finished")
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
    "Result:",
    RESULT_FILE
)

print(
    "Ground Truth:",
    GT_FILE
)

print(
    "COCO predictions:",
    PRED_COCO_FILE
)
