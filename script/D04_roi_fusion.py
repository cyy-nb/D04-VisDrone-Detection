from pathlib import Path
import json

import torch
from ultralytics import YOLO
from torchvision.ops import nms

try:
    from .d04_common import project_path
except ImportError:
    from d04_common import project_path


# ============================================================
# D04 ROI-SR Fusion
# ============================================================

# ============================================================
# 路径
# ============================================================

ROOT = Path(project_path())

MODEL_PATH = (
    ROOT
    / "experiments"
    / "baseline_640"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = (
    ROOT
    / "datasets"
    / "VisDroneYOLO"
    / "images"
    / "val"
)

ROI_SR_DIR = (
    ROOT
    / "experiments"
    / "roi_sr_x2"
    / "roi_sr"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "roi_sr_x2"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "roi_metadata.json"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "roi_fused_predictions.json"
)


# ============================================================
# 参数
# ============================================================

IMG_SIZE = 640

# 保留低置信度候选框，后面统一 NMS
CONF = 0.001

DEVICE = 0

# Real-ESRGAN x2
SR_SCALE = 2.0

# NMS IoU
NMS_IOU = 0.5


# ============================================================
# 开始
# ============================================================

print("=" * 70)
print("D04 ROI-SR Fusion")
print("=" * 70)
print()


# ============================================================
# 检查文件
# ============================================================

print("Checking files...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"YOLO model not found:\n{MODEL_PATH}"
    )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGE_DIR}"
    )

if not ROI_SR_DIR.exists():
    raise FileNotFoundError(
        f"ROI-SR directory not found:\n{ROI_SR_DIR}"
    )

if not METADATA_FILE.exists():
    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_FILE}"
    )

print("Model:", MODEL_PATH)
print("Images:", IMAGE_DIR)
print("ROI-SR:", ROI_SR_DIR)
print("Metadata:", METADATA_FILE)
print()


# ============================================================
# 加载 metadata
# ============================================================

print("Loading metadata...")

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

print(
    "Metadata images:",
    len(metadata)
)

print()


# ============================================================
# 建立 ROI-SR 文件映射
#
# metadata:
# xxx_roi_103.png
#
# Real-ESRGAN:
# xxx_roi_103_out.png
#
# 最终统一映射成：
# xxx_roi_103 -> xxx_roi_103_out.png
# ============================================================

print("Building ROI-SR file map...")

sr_files = list(
    ROI_SR_DIR.glob("*_out.png")
)

sr_map = {}

for sr_path in sr_files:

    stem = sr_path.stem

    if stem.endswith("_out"):

        original_stem = stem[:-4]

    else:

        original_stem = stem

    sr_map[original_stem] = sr_path


print(
    "ROI-SR files found:",
    len(sr_files)
)

print(
    "ROI-SR map entries:",
    len(sr_map)
)

print()


# ============================================================
# 加载 YOLO 模型
# ============================================================

print("Loading YOLO model...")

model = YOLO(
    str(MODEL_PATH)
)

print("Model loaded.")
print()


# ============================================================
# 原图列表
# ============================================================

image_files = sorted(
    IMAGE_DIR.glob("*.jpg")
)

print(
    "Images:",
    len(image_files)
)

print()


# ============================================================
# 最终预测
# ============================================================

final_predictions = []


# ============================================================
# 统计量
# ============================================================

total_original_boxes = 0

total_roi_candidates = 0

total_roi_matched = 0

total_roi_unmatched = 0

total_roi_boxes = 0

total_after_nms = 0


# ============================================================
# 遍历 548 张原图
# ============================================================

for image_index, image_path in enumerate(
    image_files
):

    image_name = image_path.name


    # ========================================================
    # 第一阶段：原图 YOLO
    # ========================================================

    original_result = model.predict(
        source=str(image_path),
        imgsz=IMG_SIZE,
        conf=CONF,
        device=DEVICE,
        verbose=False
    )[0]


    # ========================================================
    # 保存所有候选框
    # ========================================================

    all_boxes = []

    all_scores = []

    all_classes = []


    # ========================================================
    # 原图检测结果
    # ========================================================

    if (
        original_result.boxes is not None
        and len(original_result.boxes) > 0
    ):

        boxes = (
            original_result
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        scores = (
            original_result
            .boxes
            .conf
            .cpu()
            .numpy()
        )

        classes = (
            original_result
            .boxes
            .cls
            .cpu()
            .numpy()
            .astype(int)
        )


        for box, score, cls in zip(
            boxes,
            scores,
            classes
        ):

            all_boxes.append(
                box.tolist()
            )

            all_scores.append(
                float(score)
            )

            all_classes.append(
                int(cls)
            )


        total_original_boxes += len(boxes)


    # ========================================================
    # 第二阶段：ROI-SR YOLO
    # ========================================================

    image_metadata = metadata.get(
        image_name,
        []
    )


    total_roi_candidates += len(
        image_metadata
    )


    # --------------------------------------------------------
    # 遍历当前图片的所有 ROI
    # --------------------------------------------------------

    for roi_record in image_metadata:

        roi_filename = roi_record[
            "roi_file"
        ]


        # ----------------------------------------------------
        # metadata 文件名
        #
        # xxx_roi_103.png
        #
        # 对应 SR 文件
        #
        # xxx_roi_103_out.png
        # ----------------------------------------------------

        roi_stem = Path(
            roi_filename
        ).stem


        # ----------------------------------------------------
        # 从 SR 文件映射中查找
        # ----------------------------------------------------

        roi_sr_path = sr_map.get(
            roi_stem
        )


        # ----------------------------------------------------
        # 没找到
        # ----------------------------------------------------

        if roi_sr_path is None:

            total_roi_unmatched += 1

            continue


        # ----------------------------------------------------
        # 找到了
        # ----------------------------------------------------

        total_roi_matched += 1


        # ====================================================
        # YOLO 检测 SR ROI
        # ====================================================

        roi_result = model.predict(
            source=str(roi_sr_path),
            imgsz=IMG_SIZE,
            conf=CONF,
            device=DEVICE,
            verbose=False
        )[0]


        # ====================================================
        # 没有检测框
        # ====================================================

        if (
            roi_result.boxes is None
            or len(roi_result.boxes) == 0
        ):

            continue


        # ====================================================
        # 读取 SR ROI 检测框
        # ====================================================

        sr_boxes = (
            roi_result
            .boxes
            .xyxy
            .cpu()
            .numpy()
        )

        sr_scores = (
            roi_result
            .boxes
            .conf
            .cpu()
            .numpy()
        )

        sr_classes = (
            roi_result
            .boxes
            .cls
            .cpu()
            .numpy()
            .astype(int)
        )


        # ====================================================
        # 当前 ROI 在原图中的左上角坐标
        # ====================================================

        roi_x1 = float(
            roi_record["x1"]
        )

        roi_y1 = float(
            roi_record["y1"]
        )


        # ====================================================
        # SR 坐标 → 原始 ROI 坐标
        #
        # Real-ESRGAN ×2
        #
        # 所以：
        #
        # x_original = x_SR / 2
        # y_original = y_SR / 2
        # ====================================================

        for box, score, cls in zip(
            sr_boxes,
            sr_scores,
            sr_classes
        ):

            sx1, sy1, sx2, sy2 = box


            # ------------------------------------------------
            # SR → ROI
            # ------------------------------------------------

            rx1 = sx1 / SR_SCALE

            ry1 = sy1 / SR_SCALE

            rx2 = sx2 / SR_SCALE

            ry2 = sy2 / SR_SCALE


            # ------------------------------------------------
            # ROI → 原图
            # ------------------------------------------------

            ox1 = rx1 + roi_x1

            oy1 = ry1 + roi_y1

            ox2 = rx2 + roi_x1

            oy2 = ry2 + roi_y1


            # ------------------------------------------------
            # 防止坐标超出原图
            # ------------------------------------------------

            all_boxes.append([
                float(ox1),
                float(oy1),
                float(ox2),
                float(oy2)
            ])

            all_scores.append(
                float(score)
            )

            all_classes.append(
                int(cls)
            )

            total_roi_boxes += 1


    # ========================================================
    # 当前图片没有任何检测框
    # ========================================================

    if len(all_boxes) == 0:

        continue


    # ========================================================
    # 转 Tensor
    # ========================================================

    boxes_tensor = torch.tensor(
        all_boxes,
        dtype=torch.float32
    )

    scores_tensor = torch.tensor(
        all_scores,
        dtype=torch.float32
    )

    classes_tensor = torch.tensor(
        all_classes,
        dtype=torch.int64
    )


    # ========================================================
    # Class-aware NMS
    #
    # 不同类别之间不互相抑制
    # ========================================================

    max_coordinate = 10000


    offsets = (
        classes_tensor.float()
        * max_coordinate
    )


    nms_boxes = (
        boxes_tensor.clone()
    )


    nms_boxes[:, 0] += offsets
    nms_boxes[:, 1] += offsets
    nms_boxes[:, 2] += offsets
    nms_boxes[:, 3] += offsets


    # ========================================================
    # NMS
    # ========================================================

    keep = nms(
        nms_boxes,
        scores_tensor,
        NMS_IOU
    )


    keep = keep.cpu().numpy()


    total_after_nms += len(
        keep
    )


    # ========================================================
    # 保存当前图片最终预测
    # ========================================================

    image_predictions = []


    for idx in keep:

        box = (
            boxes_tensor[idx]
            .tolist()
        )

        score = float(
            scores_tensor[idx]
        )

        cls = int(
            classes_tensor[idx]
        )


        image_predictions.append({

            "bbox": [
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3])
            ],

            "score": score,

            "class": cls

        })


    final_predictions.append({

        "image": image_name,

        "predictions": image_predictions

    })


    # ========================================================
    # 进度
    # ========================================================

    if (
        (image_index + 1) % 25
        == 0
    ):

        print(
            f"{image_index + 1}/{len(image_files)} | "
            f"Original boxes: {total_original_boxes} | "
            f"ROI candidates: {total_roi_candidates} | "
            f"ROI matched: {total_roi_matched} | "
            f"ROI unmatched: {total_roi_unmatched} | "
            f"ROI boxes: {total_roi_boxes} | "
            f"After NMS: {total_after_nms}"
        )


# ============================================================
# 保存 JSON
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_predictions,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 完成
# ============================================================

print()

print("=" * 70)
print("ROI-SR Fusion Finished")
print("=" * 70)

print(
    "Images:",
    len(image_files)
)

print(
    "ROI candidates:",
    total_roi_candidates
)

print(
    "ROI-SR matched:",
    total_roi_matched
)

print(
    "ROI-SR unmatched:",
    total_roi_unmatched
)

print(
    "Original detection boxes:",
    total_original_boxes
)

print(
    "ROI-SR detection boxes:",
    total_roi_boxes
)

print(
    "Final boxes after NMS:",
    total_after_nms
)

print()

print(
    "Output:",
    OUTPUT_FILE
)
