# D04 VisDrone Detection

基于 YOLOv8 的 VisDrone 小目标检测实验，包含检测内嵌增强、三阶段训练、ROI 超分辨率和误差分析脚本。

## 目录

| 路径 | 用途 |
| --- | --- |
| `models/`、`yolov8s_d04_cbam.yaml` | 检测模型结构 |
| `D04_three_stage_train.py` | 冻结增强模块、冻结检测器、联合微调 |
| `D04_compare_latency.py` | 基线与增强模型的推理延迟对比 |
| `script/convert_visdrone.py` | VisDrone → YOLO |
| `script/yolo2coco.py` | YOLO → COCO JSON |
| `script/D04_*` | 小目标评估、ROI 处理及错误分析 |
| `super_resolution/Real-ESRGAN/` | 上游超分辨率子模块 |

## 环境

Python 3.10 或以上。仅数据转换和测试需要 Pillow：

```powershell
python -m pip install Pillow
python -m unittest discover -s tests -v
```

训练和评估还需要与本机 CUDA 匹配的 PyTorch，以及 `ultralytics`、`opencv-python`、`numpy`、`pycocotools` 等依赖。仓库未锁定经过完整训练验证的依赖版本。自定义模型中的 `D04EnhanceStem` 等模块必须在使用的 Ultralytics 环境中实现并注册；仅下载本仓库不能保证直接训练自定义模型。

按需拉取超分辨率子模块：

```powershell
git submodule update --init --recursive
```

## 数据转换

输入目录下应有 `VisDrone2019-DET-train`、`VisDrone2019-DET-val` 等子目录，每个子目录包含 `images/` 和 `annotations/`。每张 JPG 图片必须有同名 TXT 标注，空标注文件表示背景图。

```powershell
python script/convert_visdrone.py --data-root "F:/data/VisDrone" --output-root "F:/data/VisDroneYOLO" --splits train val
python script/yolo2coco.py --image-dir "F:/data/VisDroneYOLO/images/val" --label-dir "F:/data/VisDroneYOLO/labels/val" --output-json "F:/data/VisDroneYOLO/annotations/val_coco.json"
```

转换使用 10 个类别：pedestrian、people、bicycle、car、van、truck、tricycle、awning-tricycle、bus、motor。跳过忽略区域和类别 11（others），输出 YOLO/COCO 类别编号为 0–9，与模型的 `nc: 10` 一致。转换会覆盖输出标签和 COCO JSON；已有图片保持不变，替换源数据时请使用新的输出目录。

## 训练与评估

现有实验脚本大多使用 `F:\D04` 下的数据和权重路径。运行前检查各脚本顶部的路径、设备及超参数，并准备包含 `path`、`train`、`val`、`names` 的 `visdrone.yaml`。模型权重、数据集和实验结果未包含在 Git 仓库中。

```powershell
python D04_three_stage_train.py
python D04_compare_latency.py
```

以上命令需要对应环境、数据和初始权重；单元测试不代表训练效果或 GPU 性能已验证。改动说明见 [docs/CHANGES.md](docs/CHANGES.md)。
