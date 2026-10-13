import json
from pathlib import Path

INPUT_FILE = Path(
    r"F:\D04\experiments\roi_sr_x2\roi_fused_predictions.json"
)

OUTPUT_FILE = Path(
    r"F:\D04\experiments\roi_sr_x2\roi_fused_predictions_small32.json"
)

SMALL_MAX = 32 * 32


print("=" * 70)
print("D04 ROI-SR Small Object Filter")
print("=" * 70)

print(f"Input : {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")
print("Rule  : width × height < 32 × 32")
print(f"       area < {SMALL_MAX}")
print()


# =========================================================
# 读取
# =========================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"找不到输入文件：\n{INPUT_FILE}"
    )

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

print("JSON 类型:", type(data).__name__)


# =========================================================
# 递归筛选
# =========================================================

total_before = 0
total_after = 0
filtered_count = 0
invalid_boxes = 0


def filter_predictions(obj):

    global total_before
    global total_after
    global filtered_count
    global invalid_boxes

    # -----------------------------------------------------
    # list
    # -----------------------------------------------------

    if isinstance(obj, list):

        new_list = []

        for item in obj:

            # 如果是预测框 dict
            if isinstance(item, dict) and "bbox" in item:

                bbox = item.get("bbox")

                if (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                ):
                    invalid_boxes += 1
                    continue

                try:
                    x1 = float(bbox[0])
                    y1 = float(bbox[1])
                    x2 = float(bbox[2])
                    y2 = float(bbox[3])
                except Exception:
                    invalid_boxes += 1
                    continue

                width = max(0.0, x2 - x1)
                height = max(0.0, y2 - y1)

                area = width * height

                total_before += 1

                if area < SMALL_MAX:
                    new_list.append(item)
                    total_after += 1
                else:
                    filtered_count += 1

            else:
                # 普通嵌套结构，继续递归
                new_list.append(
                    filter_predictions(item)
                )

        return new_list


    # -----------------------------------------------------
    # dict
    # -----------------------------------------------------

    elif isinstance(obj, dict):

        new_dict = {}

        for key, value in obj.items():

            # bbox 预测框
            if key == "bbox":
                new_dict[key] = value

            else:
                new_dict[key] = filter_predictions(value)

        return new_dict


    # -----------------------------------------------------
    # 其他类型
    # -----------------------------------------------------

    else:
        return obj


# =========================================================
# 执行
# =========================================================

filtered_data = filter_predictions(data)


# =========================================================
# 保存
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        filtered_data,
        f,
        indent=2,
        ensure_ascii=False
    )


# =========================================================
# 统计
# =========================================================

ratio = (
    total_after / total_before * 100
    if total_before > 0
    else 0
)


print()
print("=" * 70)
print("筛选完成")
print("=" * 70)

print(f"原始预测框     : {total_before}")
print(f"< 32×32 小目标 : {total_after}")
print(f"被过滤框       : {filtered_count}")
print(f"小目标比例     : {ratio:.2f}%")
print(f"无效框         : {invalid_boxes}")

print()
print("输出文件:")
print(OUTPUT_FILE)

print("=" * 70)

