import os
import json

# ===================== 改这里！！ =====================
img_dir = r"F:\D04\datasets\VisDroneYOLO\images\val"   # val图片文件夹
label_dir = r"F:\D04\datasets\VisDroneYOLO\labels\val" # val标签txt文件夹
output_json = r"F:\D04\datasets\VisDroneYOLO\annotations\val_coco.json"
class_names = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
# =====================================================

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

for fname in os.listdir(img_dir):
    if not fname.endswith((".jpg",".png")):
        continue
    img_path = os.path.join(img_dir, fname)
    # 读取图片宽高（如果不想读图片，可以手动写死VisDrone固定尺寸，VisDrone原图大多1920×1080）
    from PIL import Image
    im = Image.open(img_path)
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
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            cls_id, xc, yc, bw, bh = map(float, line.split())
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

os.makedirs(os.path.dirname(output_json), exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(coco, f, indent=2)
print(f"COCO标注文件生成完成：{output_json}")
