from ultralytics import YOLO

# 改成你自己best.pt和yaml路径
model = YOLO(r"F:\D04\experiments\baseline_640\weights\best.pt")
res = model.val(
    data=r"F:\D04\datasets\VisDroneYOLO\visdrone.yaml",
    save_json=True,  # 关键：开启保存预测json
    save_plots=True
)
# 打印输出文件夹，直接复制这个路径！
print("✅ val结果保存目录：", res.save_dir)
