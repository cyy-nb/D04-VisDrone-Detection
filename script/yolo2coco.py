import os
import json
import argparse
import math
from PIL import Image

# 默认路径，可通过命令行参数覆盖。
img_dir = r"F:\D04\datasets\VisDroneYOLO\images\val"   # val图片文件夹
label_dir = r"F:\D04\datasets\VisDroneYOLO\labels\val" # val标签txt文件夹
output_json = r"F:\D04\datasets\VisDroneYOLO\annotations\val_coco.json"
class_names = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
# =====================================================

def convert(img_dir, label_dir, output_json):
    """Export ten-class labels without performing work at import time."""
    if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
        raise FileNotFoundError("Image and label directories must exist")
    coco = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # 类别
    for idx, name in enumerate(class_names):
        coco["categories"].append({"id": idx, "name": name})

    img_id = 0
    ann_id = 0

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(img_dir, fname)
        # 每张图片分别读取真实尺寸。
        with Image.open(img_path) as im:
            w, h = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": fname,
            "width": w,
            "height": h
        })

        # 对应的txt标签
        txt_name = os.path.splitext(fname)[0] + ".txt"
        txt_path = os.path.join(label_dir, txt_name)
        if not os.path.isfile(txt_path):
            raise FileNotFoundError(f"Missing label: {txt_path}")
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                cls_id, xc, yc, bw, bh = map(float, line.split())
                if not all(math.isfinite(v) for v in (cls_id, xc, yc, bw, bh)):
                    raise ValueError(f"Non-finite label in {txt_path}: {line}")
                if not cls_id.is_integer() or not 0 <= cls_id < len(class_names):
                    raise ValueError(f"Invalid class in {txt_path}: {line}")
                if not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                    raise ValueError(f"Invalid box in {txt_path}: {line}")
                cls_id = int(cls_id)
                # YOLO归一化坐标 → COCO像素坐标
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                bbox_w = bw * w
                bbox_h = bh * h
                area = bbox_w * bbox_h
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls_id,
                    "bbox": [x1, y1, bbox_w, bbox_h],
                    "area": area,
                    "iscrowd":0
                })
                ann_id +=1
        img_id +=1

    if not coco["images"]:
        raise ValueError(f"No supported images in {img_dir}")
    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)
    print(f"COCO标注文件生成完成：{output_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO labels to COCO JSON")
    parser.add_argument("--image-dir", default=img_dir)
    parser.add_argument("--label-dir", default=label_dir)
    parser.add_argument("--output-json", default=output_json)
    args = parser.parse_args()
    convert(args.image_dir, args.label_dir, args.output_json)
