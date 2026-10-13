from pathlib import Path
import argparse
import math
import shutil
from PIL import Image

# 原始数据位置
DATA_ROOT = Path(r"F:\D04\datasets\VisDrone")

# 转换后的 YOLO 数据位置
OUTPUT_ROOT = Path(r"F:\D04\datasets\VisDroneYOLO")

# VisDrone 类别
# 0 pedestrian
# 1 people
# 2 bicycle
# 3 car
# 4 van
# 5 truck
# 6 tricycle
# 7 awning-tricycle
# 8 bus
# 9 motor
# 类别 11 (others) 不属于本项目的 10 类检测目标。

splits = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def convert_split(split_name, source_name, data_root=DATA_ROOT, output_root=OUTPUT_ROOT):
    source_dir = Path(data_root) / source_name
    image_dir = source_dir / "images"
    annotation_dir = source_dir / "annotations"

    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"缺少图片或标注目录：{source_dir}")
    image_files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() == ".jpg")
    if not image_files:
        raise ValueError(f"没有找到 JPG 图片：{image_dir}")
    missing = [p.stem for p in image_files if not (annotation_dir / f"{p.stem}.txt").is_file()]
    if missing:
        raise FileNotFoundError(f"缺少标注：{missing[0]}.txt（共 {len(missing)} 个）")
    output_image_dir = Path(output_root) / "images" / split_name
    output_label_dir = Path(output_root) / "labels" / split_name

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)


    print(f"\n处理 {split_name}: {len(image_files)} 张图片")

    converted = 0

    for image_path in image_files:

        annotation_path = annotation_dir / f"{image_path.stem}.txt"

        # 获取图片尺寸
        with Image.open(image_path) as img:
            img_width, img_height = img.size

        yolo_lines = []

        if annotation_path.exists():

            with open(annotation_path, "r", encoding="utf-8") as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split(",")

                    if len(parts) < 8:
                        continue

                    # VisDrone:
                    # bbox_left, bbox_top, bbox_width, bbox_height,
                    # score, object_category, truncation, occlusion

                    x = float(parts[0])
                    y = float(parts[1])
                    w = float(parts[2])
                    h = float(parts[3])

                    score = int(parts[4])
                    category = int(parts[5])

                    # score=0 表示忽略区域
                    if score == 0:
                        continue

                    # 与模型 nc: 10 和 COCO 类别 0~9 保持一致。
                    if category < 1 or category > 10:
                        continue

                    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                        continue
                    # 先裁剪两个角点，再计算中心和尺寸，保证整个框在图内。
                    x2 = min(x + w, img_width)
                    y2 = min(y + h, img_height)
                    x, y = max(x, 0), max(y, 0)
                    w, h = x2 - x, y2 - y
                    if w <= 0 or h <= 0:
                        continue

                    class_id = category - 1

                    # 转换为中心点坐标
                    x_center = x + w / 2
                    y_center = y + h / 2

                    # 归一化
                    x_center /= img_width
                    y_center /= img_height
                    w /= img_width
                    h /= img_height

                    # 防止异常数据超出范围
                    x_center = min(max(x_center, 0), 1)
                    y_center = min(max(y_center, 0), 1)
                    w = min(max(w, 0), 1)
                    h = min(max(h, 0), 1)

                    yolo_lines.append(
                        f"{class_id} "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{w:.6f} "
                        f"{h:.6f}"
                    )

        # 写 YOLO 标签
        output_label_path = output_label_dir / f"{image_path.stem}.txt"

        with open(output_label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        # 复制图片
        output_image_path = output_image_dir / image_path.name

        if not output_image_path.exists():
            shutil.copy2(image_path, output_image_path)

        converted += 1

        if converted % 500 == 0:
            print(f"已处理 {converted}/{len(image_files)}")

    print(f"{split_name} 完成，共处理 {converted} 张图片")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将 VisDrone 标注转换为 10 类 YOLO 数据集")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--splits", nargs="+", choices=list(splits), default=list(splits))
    args = parser.parse_args()
    for split_name in args.splits:
        convert_split(split_name, splits[split_name], args.data_root, args.output_root)

    print("\n================================")
    print("VisDrone → YOLO 转换完成！")
    print("================================")
