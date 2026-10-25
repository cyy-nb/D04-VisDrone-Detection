from pathlib import Path
import cv2
from ultralytics import YOLO

try:
    from .d04_common import project_path
except ImportError:
    from d04_common import project_path

# ============================================================
# D04 Baseline Error Visualization
# ============================================================

MODEL_PATH = project_path('experiments', 'baseline_640', 'weights', 'best.pt')

IMAGE_DIR = Path(
    project_path('datasets', 'VisDroneYOLO', 'images', 'val')
)

LABEL_DIR = Path(
    project_path('datasets', 'VisDroneYOLO', 'labels', 'val')
)

OUTPUT_DIR = Path(
    project_path('experiments', 'baseline_640', 'error_analysis_visual')
)

IMG_SIZE = 640

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5

# 每一种错误、每一个尺寸最多保存多少张
MAX_IMAGES_PER_GROUP = 5

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

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    inter_area = inter_w * inter_h

    area1 = max(
        0,
        box1[2] - box1[0]
    ) * max(
        0,
        box1[3] - box1[1]
    )

    area2 = max(
        0,
        box2[2] - box2[0]
    ) * max(
        0,
        box2[3] - box2[1]
    )

    union = area1 + area2 - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


# ============================================================
# 目标尺寸
# ============================================================

def get_size_group(width, height):

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
# 读取GT
# ============================================================

def load_gt(label_path, width, height):

    objects = []

    if not label_path.exists():
        return objects

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls = int(parts[0])

            xc = float(parts[1]) * width
            yc = float(parts[2]) * height

            w = float(parts[3]) * width
            h = float(parts[4]) * height

            x1 = xc - w / 2
            y1 = yc - h / 2

            x2 = xc + w / 2
            y2 = yc + h / 2

            objects.append({
                "class_id": cls,
                "box": [x1, y1, x2, y2],
                "width": w,
                "height": h
            })

    return objects


# ============================================================
# 创建输出目录
# ============================================================

def create_output_dirs():

    error_types = [
        "miss",
        "localization",
        "classification",
        "correct"
    ]

    size_groups = [
        "0-8",
        "8-16",
        "16-24",
        "24-32",
        "32+"
    ]

    for error_type in error_types:

        for size_group in size_groups:

            path = (
                OUTPUT_DIR
                / error_type
                / size_group
            )

            path.mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# 画GT框
# ============================================================

def draw_gt(
    image,
    box,
    class_id
):

    x1, y1, x2, y2 = [
        int(v) for v in box
    ]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = (
        "GT: "
        + CLASS_NAMES[class_id]
    )

    cv2.putText(
        image,
        label,
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )


# ============================================================
# 画预测框
# ============================================================

def draw_prediction(
    image,
    box,
    class_id,
    confidence,
    iou
):

    x1, y1, x2, y2 = [
        int(v) for v in box
    ]

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        2
    )

    label = (
        "Pred: "
        + CLASS_NAMES[class_id]
        + f" {confidence:.2f}"
        + f" IoU={iou:.2f}"
    )

    cv2.putText(
        image,
        label,
        (x1, min(image.shape[0] - 5, y2 + 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        1,
        cv2.LINE_AA
    )


# ============================================================
# 保存图片
# ============================================================

def save_visualization(
    image,
    output_dir,
    image_name,
    error_type,
    size_group,
    index
):

    filename = (
        f"{index:02d}_"
        f"{Path(image_name).stem}_"
        f"{error_type}_"
        f"{size_group}.jpg"
    )

    output_path = (
        output_dir
        / error_type
        / size_group
        / filename
    )

    cv2.imwrite(
        str(output_path),
        image
    )


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)
    print("D04 BASELINE ERROR VISUALIZATION")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    create_output_dirs()

    # --------------------------------------------------------
    # 图片
    # --------------------------------------------------------

    images = []

    for ext in [
        "*.jpg",
        "*.JPG",
        "*.jpeg",
        "*.png"
    ]:

        images.extend(
            IMAGE_DIR.glob(ext)
        )

    images = sorted(images)

    print()
    print(
        "Images:",
        len(images)
    )

    # --------------------------------------------------------
    # 模型
    # --------------------------------------------------------

    print()
    print("Loading model...")

    model = YOLO(
        MODEL_PATH
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # 图片计数器
    # --------------------------------------------------------

    saved_count = {}

    for error_type in [
        "miss",
        "localization",
        "classification",
        "correct"
    ]:

        for size_group in [
            "0-8",
            "8-16",
            "16-24",
            "24-32",
            "32+"
        ]:

            saved_count[
                (error_type, size_group)
            ] = 0

    # --------------------------------------------------------
    # 开始处理
    # --------------------------------------------------------

    for image_index, image_path in enumerate(
        images,
        start=1
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        height, width = image.shape[:2]

        label_path = (
            LABEL_DIR
            / f"{image_path.stem}.txt"
        )

        gt_objects = load_gt(
            label_path,
            width,
            height
        )

        # ----------------------------------------------------
        # 预测
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

            boxes = (
                result.boxes
                .xyxy
                .cpu()
                .numpy()
            )

            classes = (
                result.boxes
                .cls
                .cpu()
                .numpy()
            )

            scores = (
                result.boxes
                .conf
                .cpu()
                .numpy()
            )

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
        # GT逐个分析
        # ----------------------------------------------------

        for gt in gt_objects:

            size_group = get_size_group(
                gt["width"],
                gt["height"]
            )

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

            # ================================================
            # 1. 漏检
            # ================================================

            if best_prediction is None:

                error_type = "miss"

                canvas = image.copy()

                draw_gt(
                    canvas,
                    gt["box"],
                    gt["class_id"]
                )

                cv2.putText(
                    canvas,
                    "MISS",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )

            # ================================================
            # 2. 正确
            # ================================================

            elif best_iou >= IOU_THRESHOLD:

                if (
                    best_prediction["class_id"]
                    == gt["class_id"]
                ):

                    error_type = "correct"

                else:

                    error_type = "classification"

                canvas = image.copy()

                draw_gt(
                    canvas,
                    gt["box"],
                    gt["class_id"]
                )

                draw_prediction(
                    canvas,
                    best_prediction["box"],
                    best_prediction["class_id"],
                    best_prediction["score"],
                    best_iou
                )

                cv2.putText(
                    canvas,
                    error_type.upper(),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 255),
                    2
                )

            # ================================================
            # 3. IoU < 0.5
            # ================================================

            else:

                if (
                    best_prediction["class_id"]
                    == gt["class_id"]
                ):

                    error_type = "localization"

                else:

                    error_type = "classification"

                canvas = image.copy()

                draw_gt(
                    canvas,
                    gt["box"],
                    gt["class_id"]
                )

                draw_prediction(
                    canvas,
                    best_prediction["box"],
                    best_prediction["class_id"],
                    best_prediction["score"],
                    best_iou
                )

                cv2.putText(
                    canvas,
                    error_type.upper(),
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 0, 255),
                    2
                )

            # ------------------------------------------------
            # 限制每组图片数量
            # ------------------------------------------------

            key = (
                error_type,
                size_group
            )

            if (
                saved_count[key]
                < MAX_IMAGES_PER_GROUP
            ):

                saved_count[key] += 1

                save_visualization(
                    canvas,
                    OUTPUT_DIR,
                    image_path.name,
                    error_type,
                    size_group,
                    saved_count[key]
                )

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
    print("VISUALIZATION FINISHED")
    print("=" * 70)

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print()

    print(
        "Example folders:"
    )

    print(
        OUTPUT_DIR / "miss"
    )

    print(
        OUTPUT_DIR / "localization"
    )

    print(
        OUTPUT_DIR / "classification"
    )

    print(
        OUTPUT_DIR / "correct"
    )


if __name__ == "__main__":
    main()
