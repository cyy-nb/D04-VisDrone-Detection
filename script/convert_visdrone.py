from pathlib import Path
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
# 10 others

splits = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def convert_split(split_name, source_name):
    source_dir = DATA_ROOT / source_name
    image_dir = source_dir / "images"
    annotation_dir = source_dir / "annotations"

    output_image_dir = OUTPUT_ROOT / "images" / split_name
    output_label_dir = OUTPUT_ROOT / "labels" / split_name

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    image_files = list(image_dir.glob("*.jpg"))

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

                    # VisDrone 类别 1~11
                    # YOLO 类别需要 0~10
                    if category < 1 or category > 11:
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
            output_image_path.write_bytes(image_path.read_bytes())

        converted += 1

        if converted % 500 == 0:
            print(f"已处理 {converted}/{len(image_files)}")

    print(f"{split_name} 完成，共处理 {converted} 张图片")


if __name__ == "__main__":

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split_name, source_name in splits.items():
        convert_split(split_name, source_name)

    print("\n================================")
    print("VisDrone → YOLO 转换完成！")
    print("================================")