from pathlib import Path
import json
import cv2
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


# ============================================================
# D04 Baseline - COCO-style AP_s / AP_m / AP_l
# Dataset: VisDroneYOLO
# Model: baseline_640/best.pt
# ============================================================

MODEL_PATH = r"F:\D04\experiments\detection_embedded\dee_yolov8s_v1\weights\best.pt"

IMAGE_DIR = Path(r"F:\D04\datasets\VisDroneYOLO\images\val")
LABEL_DIR = Path(r"F:\D04\datasets\VisDroneYOLO\labels\val")

OUTPUT_DIR = Path(r"F:\D04\experiments\detection_embedded\dee_yolov8s_v1_best_eval\ap_eval")

IMG_SIZE = 640

# COCO area definitions
SMALL_MAX = 32 ** 2
MEDIUM_MAX = 96 ** 2


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


def get_images():
    images = []

    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        images.extend(IMAGE_DIR.glob(ext))

    return sorted(images)


def read_yolo_label(label_path, width, height):

    objects = []

    if not label_path.exists():
        return objects

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height

            box_width = float(parts[3]) * width
            box_height = float(parts[4]) * height

            x = x_center - box_width / 2
            y = y_center - box_height / 2

            area = box_width * box_height

            objects.append({
                "class_id": class_id,
                "bbox": [
                    x,
                    y,
                    box_width,
                    box_height
                ],
                "area": area
            })

    return objects


def build_coco_ground_truth(images):

    coco = {
        "info": {
            "description": "D04 VisDrone Baseline Evaluation"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    for i, name in enumerate(CLASS_NAMES):
        coco["categories"].append({
            "id": i + 1,
            "name": name,
            "supercategory": "object"
        })

    annotation_id = 1

    print("Building ground-truth annotations...")

    for image_id, image_path in enumerate(images, start=1):

        image = cv2.imread(str(image_path))

        if image is None:
            print("WARNING: Cannot read:", image_path)
            continue

        height, width = image.shape[:2]

        coco["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height
        })

        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        objects = read_yolo_label(
            label_path,
            width,
            height
        )

        for obj in objects:

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": obj["class_id"] + 1,
                "bbox": obj["bbox"],
                "area": obj["area"],
                "iscrowd": 0
            })

            annotation_id += 1

    return coco


def main():

    print("=" * 70)
    print("D04 Small Object Evaluation")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    images = get_images()

    print()
    print("Images:", len(images))
    print("Model:", MODEL_PATH)
    print()

    # --------------------------------------------------------
    # 1. Build COCO ground truth
    # --------------------------------------------------------

    gt_data = build_coco_ground_truth(images)

    gt_json = OUTPUT_DIR / "ground_truth.json"

    with open(
        gt_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            gt_data,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print("Ground truth saved:")
    print(gt_json)

    # --------------------------------------------------------
    # 2. Load model
    # --------------------------------------------------------

    print()
    print("Loading YOLO model...")

    model = YOLO(MODEL_PATH)

    print("Model loaded.")
    print()

    # --------------------------------------------------------
    # 3. Run prediction
    # --------------------------------------------------------

    predictions = []

    print("Running predictions...")

    for image_id, image_path in enumerate(
        images,
        start=1
    ):

        results = model.predict(
            source=str(image_path),
            imgsz=IMG_SIZE,
            conf=0.001,
            max_det=317,
            device=0,
            verbose=False
        )

        result = results[0]

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        for box, cls, score in zip(
            boxes,
            classes,
            scores
        ):

            x1, y1, x2, y2 = box

            width = max(0, x2 - x1)
            height = max(0, y2 - y1)

            predictions.append({
                "image_id": image_id,
                "category_id": int(cls) + 1,
                "bbox": [
                    float(x1),
                    float(y1),
                    float(width),
                    float(height)
                ],
                "score": float(score)
            })

        if image_id % 50 == 0 or image_id == len(images):

            print(
                f"Progress: "
                f"{image_id}/{len(images)}"
            )

    pred_json = OUTPUT_DIR / "predictions.json"

    with open(
        pred_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            predictions,
            f
        )

    print()
    print("Predictions saved:")
    print(pred_json)

    # --------------------------------------------------------
    # 4. COCO evaluation
    # --------------------------------------------------------

    print()
    print("Running COCO evaluation...")
    print()

    coco_gt = COCO(str(gt_json))

    coco_dt = coco_gt.loadRes(
        str(pred_json)
    )

    evaluator = COCOeval(
        coco_gt,
        coco_dt,
        "bbox"
    )

    evaluator.params.imgIds = [
        image_id
        for image_id in range(
            1,
            len(images) + 1
            evaluator.params.maxDets = [1, 10, 317]
        )
    ]

    # COCO IoU thresholds:
    # 0.50, 0.55, ..., 0.95

    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    # --------------------------------------------------------
    # 5. Extract AP values
    # --------------------------------------------------------

    stats = evaluator.stats

    print()
    print("=" * 70)
    print("D04 BASELINE RESULTS")
    print("=" * 70)

    print()
    print(f"mAP50-95 : {stats[0]:.4f}")
    print(f"mAP50    : {stats[1]:.4f}")
    print(f"mAP75    : {stats[2]:.4f}")

    print()
    print(f"AP_s     : {stats[3]:.4f}")
    print(f"AP_m     : {stats[4]:.4f}")
    print(f"AP_l     : {stats[5]:.4f}")

    print()
    print("=" * 70)

    # --------------------------------------------------------
    # 6. Save results
    # --------------------------------------------------------

    result_file = OUTPUT_DIR / "d04_best_ap_results.txt"

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("D04 Baseline AP Evaluation\n")
        f.write("===========================\n\n")

        f.write(
            f"mAP50-95: {stats[0]:.6f}\n"
        )

        f.write(
            f"mAP50: {stats[1]:.6f}\n"
        )

        f.write(
            f"mAP75: {stats[2]:.6f}\n"
        )

        f.write(
            f"AP_s: {stats[3]:.6f}\n"
        )

        f.write(
            f"AP_m: {stats[4]:.6f}\n"
        )

        f.write(
            f"AP_l: {stats[5]:.6f}\n"
        )

    print()
    print("Results saved to:")
    print(result_file)


if __name__ == "__main__":
    main()