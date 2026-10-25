from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

try:
    from .d04_common import project_path
except ImportError:
    from d04_common import project_path

# ============================================================
# D04 Baseline Error Analysis
# ============================================================

MODEL_PATH = project_path('experiments', 'baseline_640', 'weights', 'best.pt')
IMAGE_DIR = Path(project_path('datasets', 'VisDroneYOLO', 'images', 'val'))
LABEL_DIR = Path(project_path('datasets', 'VisDroneYOLO', 'labels', 'val'))

OUTPUT_DIR = Path(project_path('experiments', 'baseline_640', 'error_analysis'))

IMG_SIZE = 640

# 匹配阈值
IOU_THRESHOLD = 0.5

# 预测置信度
CONF_THRESHOLD = 0.25

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
    "motor",
]


# ============================================================
# IoU
# ============================================================

def calculate_iou(box1, box2):
    """
    box格式：
    [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


# ============================================================
# 读取YOLO标签
# ============================================================

def load_ground_truth(label_path, width, height):

    gt_objects = []

    if not label_path.exists():
        return gt_objects

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            xc = float(parts[1]) * width
            yc = float(parts[2]) * height

            w = float(parts[3]) * width
            h = float(parts[4]) * height

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            gt_objects.append({
                "class_id": class_id,
                "box": [x1, y1, x2, y2],
                "width": w,
                "height": h,
                "area": w * h,
                "matched": False
            })

    return gt_objects


# ============================================================
# 尺寸分类
# ============================================================

def get_size_group(width, height):

    # 使用目标的最大边近似表示目标尺寸
    size = max(width, height)

    if size <= 8:
        return "0-8"

    elif size <= 16:
        return "8-16"

    elif size <= 24:
        return "16-24"

    elif size <= 32:
        return "24-32"

    else:
        return "32+"


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)
    print("D04 BASELINE ERROR ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 找图片
    # --------------------------------------------------------

    images = []

    for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.png"]:

        images.extend(IMAGE_DIR.glob(ext))

    images = sorted(images)

    print()
    print("Images:", len(images))

    # --------------------------------------------------------
    # 加载模型
    # --------------------------------------------------------

    print()
    print("Loading model...")

    model = YOLO(MODEL_PATH)

    print("Model loaded.")

    # --------------------------------------------------------
    # 错误统计
    # --------------------------------------------------------

    error_counts = {
        "Correct": 0,
        "Miss": 0,
        "Localization Error": 0,
        "Classification Error": 0
    }

    # 尺寸统计

    size_groups = {
        "0-8": {
            "total": 0,
            "correct": 0,
            "miss": 0,
            "localization": 0,
            "classification": 0
        },

        "8-16": {
            "total": 0,
            "correct": 0,
            "miss": 0,
            "localization": 0,
            "classification": 0
        },

        "16-24": {
            "total": 0,
            "correct": 0,
            "miss": 0,
            "localization": 0,
            "classification": 0
        },

        "24-32": {
            "total": 0,
            "correct": 0,
            "miss": 0,
            "localization": 0,
            "classification": 0
        },

        "32+": {
            "total": 0,
            "correct": 0,
            "miss": 0,
            "localization": 0,
            "classification": 0
        }
    }

    # 保存详细结果

    detailed_results = []

    # --------------------------------------------------------
    # 开始分析
    # --------------------------------------------------------

    for image_index, image_path in enumerate(images, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        gt_objects = load_ground_truth(
            label_path,
            width,
            height
        )

        # ----------------------------------------------------
        # 模型预测
        # ----------------------------------------------------

        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            max_det=1000,
            verbose=False
        )

        result = results[0]

        predictions = []

        if result.boxes is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()

            for box, cls, score in zip(
                boxes,
                classes,
                scores
            ):

                predictions.append({
                    "class_id": int(cls),
                    "box": box.tolist(),
                    "score": float(score)
                })

        # ----------------------------------------------------
        # 给GT寻找最佳预测
        # ----------------------------------------------------

        for gt in gt_objects:

            best_iou = 0
            best_prediction = None

            for pred in predictions:

                iou = calculate_iou(
                    gt["box"],
                    pred["box"]
                )

                if iou > best_iou:

                    best_iou = iou
                    best_prediction = pred

            # -----------------------------------------------
            # 尺寸类别
            # -----------------------------------------------

            size_group = get_size_group(
                gt["width"],
                gt["height"]
            )

            size_groups[size_group]["total"] += 1

            # -----------------------------------------------
            # 没有任何匹配
            # -----------------------------------------------

            if best_prediction is None:

                error_counts["Miss"] += 1
                size_groups[size_group]["miss"] += 1

                detailed_results.append({
                    "image": image_path.name,
                    "class": CLASS_NAMES[gt["class_id"]],
                    "size_group": size_group,
                    "error": "Miss",
                    "iou": 0
                })

                continue

            # -----------------------------------------------
            # IoU >= threshold
            # -----------------------------------------------

            if best_iou >= IOU_THRESHOLD:

                if (
                    best_prediction["class_id"]
                    == gt["class_id"]
                ):

                    error_counts["Correct"] += 1
                    size_groups[size_group]["correct"] += 1

                    error_type = "Correct"

                else:

                    error_counts["Classification Error"] += 1
                    size_groups[size_group]["classification"] += 1

                    error_type = "Classification Error"

            # -----------------------------------------------
            # IoU不足
            # -----------------------------------------------

            else:

                # 如果预测类别正确，但框的位置明显不准
                if (
                    best_prediction["class_id"]
                    == gt["class_id"]
                ):

                    error_counts["Localization Error"] += 1
                    size_groups[size_group]["localization"] += 1

                    error_type = "Localization Error"

                else:

                    # IoU低且类别也错误
                    error_counts["Classification Error"] += 1
                    size_groups[size_group]["classification"] += 1

                    error_type = "Classification Error"

            detailed_results.append({
                "image": image_path.name,
                "class": CLASS_NAMES[gt["class_id"]],
                "size_group": size_group,
                "error": error_type,
                "iou": round(float(best_iou), 4)
            })

        # ----------------------------------------------------
        # 进度
        # ----------------------------------------------------

        if (
            image_index % 25 == 0
            or image_index == len(images)
        ):

            print(
                f"Progress: "
                f"{image_index}/{len(images)}"
            )

    # ========================================================
    # 输出统计
    # ========================================================

    print()
    print("=" * 70)
    print("ERROR ANALYSIS RESULTS")
    print("=" * 70)

    print()

    for key, value in error_counts.items():

        print(
            f"{key:<25}: {value}"
        )

    # ========================================================
    # 保存JSON
    # ========================================================

    json_path = OUTPUT_DIR / "error_analysis.json"

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "error_counts": error_counts,
                "size_groups": size_groups,
                "details": detailed_results
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("JSON saved:")
    print(json_path)

    # ========================================================
    # 保存TXT
    # ========================================================

    txt_path = OUTPUT_DIR / "error_analysis.txt"

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "D04 Baseline Error Analysis\n"
        )

        f.write(
            "===========================\n\n"
        )

        f.write(
            "Overall Error Statistics\n"
        )

        f.write(
            "-------------------------\n"
        )

        for key, value in error_counts.items():

            f.write(
                f"{key}: {value}\n"
            )

        f.write(
            "\n\nSmall Object Size Analysis\n"
        )

        f.write(
            "--------------------------\n"
        )

        for group, data in size_groups.items():

            f.write(
                f"\n{group} px\n"
            )

            f.write(
                f"Total: {data['total']}\n"
            )

            f.write(
                f"Correct: {data['correct']}\n"
            )

            f.write(
                f"Miss: {data['miss']}\n"
            )

            f.write(
                f"Localization: {data['localization']}\n"
            )

            f.write(
                f"Classification: {data['classification']}\n"
            )

    print()
    print("TXT saved:")
    print(txt_path)

    # ========================================================
    # 错误类型柱状图
    # ========================================================

    labels = list(error_counts.keys())
    values = list(error_counts.values())

    plt.figure(figsize=(10, 6))

    plt.bar(labels, values)

    plt.title(
        "D04 Baseline Error Analysis"
    )

    plt.ylabel("Number of Objects")

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    error_chart = (
        OUTPUT_DIR /
        "error_type_bar.png"
    )

    plt.savefig(
        error_chart,
        dpi=300
    )

    plt.close()

    print()
    print("Error chart saved:")
    print(error_chart)

    # ========================================================
    # 小目标尺寸错误图
    # ========================================================

    groups = list(size_groups.keys())

    miss_values = [
        size_groups[x]["miss"]
        for x in groups
    ]

    localization_values = [
        size_groups[x]["localization"]
        for x in groups
    ]

    classification_values = [
        size_groups[x]["classification"]
        for x in groups
    ]

    correct_values = [
        size_groups[x]["correct"]
        for x in groups
    ]

    x = np.arange(len(groups))

    width_bar = 0.2

    plt.figure(figsize=(11, 6))

    plt.bar(
        x - 1.5 * width_bar,
        correct_values,
        width_bar,
        label="Correct"
    )

    plt.bar(
        x - 0.5 * width_bar,
        miss_values,
        width_bar,
        label="Miss"
    )

    plt.bar(
        x + 0.5 * width_bar,
        localization_values,
        width_bar,
        label="Localization Error"
    )

    plt.bar(
        x + 1.5 * width_bar,
        classification_values,
        width_bar,
        label="Classification Error"
    )

    plt.xticks(
        x,
        groups
    )

    plt.xlabel(
        "Object Size (max side, pixels)"
    )

    plt.ylabel(
        "Number of Objects"
    )

    plt.title(
        "Detection Errors by Object Size"
    )

    plt.legend()

    plt.tight_layout()

    size_chart = (
        OUTPUT_DIR /
        "size_error_analysis.png"
    )

    plt.savefig(
        size_chart,
        dpi=300
    )

    plt.close()

    print()
    print("Size chart saved:")
    print(size_chart)

    # ========================================================
    # 完成
    # ========================================================

    print()
    print("=" * 70)
    print("ERROR ANALYSIS FINISHED")
    print("=" * 70)

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
