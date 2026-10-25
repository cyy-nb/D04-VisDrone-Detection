# D04 VisDrone Detection

基于 YOLOv8 的 VisDrone 小目标检测实验项目，包含数据转换、检测内嵌增强、三阶段训练、ROI 超分辨率、COCO 指标评估和误差分析脚本。

> 本仓库不包含 VisDrone 数据集、模型权重或完整训练结果。真实训练效果和 GPU 性能仍需在匹配的数据、权重与 CUDA 环境中复现。

## 目录

| 路径 | 用途 |
| --- | --- |
| `models/`、`yolov8s_d04_cbam.yaml` | 检测模型结构 |
| `D04_three_stage_train.py` | 冻结增强模块、冻结检测器、联合微调 |
| `D04_compare_latency.py` | 基线与增强模型的推理延迟对比 |
| `script/convert_visdrone.py` | VisDrone DET → YOLO |
| `script/yolo2coco.py` | YOLO → COCO JSON |
| `script/D04_*_small_object_eval.py` | COCO AP、AP_s/AP_m/AP_l 评估 |
| `script/D04_*` | ROI 处理、ROI-SR 融合及误差分析 |
| `script/d04_common.py` | 路径、标签与 JSON 公共工具 |
| `script/d04_registry.py` | 向 Ultralytics 注册 `D04EnhanceStem` |
| `super_resolution/Real-ESRGAN/` | 上游超分辨率子模块 |

## 环境安装

核心数据转换和单元测试只依赖 Pillow：

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

开发检查：

```powershell
python -m pip install -r requirements-dev.txt
python -m pyflakes D04_compare_latency.py D04_three_stage_train.py script tests
python -m compileall -q D04_compare_latency.py D04_three_stage_train.py script tests
python -m pytest
```

训练/评估依赖：

```powershell
python -m pip install -r requirements-train.txt
```

PyTorch 应优先按照本机 CUDA 版本从官方安装命令安装；`requirements-train.txt` 只约束最低版本。拉取 Real-ESRGAN 子模块：

```powershell
git submodule update --init --recursive
```

## 配置项目根目录

默认沿用原项目的 `F:\D04` 布局。迁移到其他位置时无需改源码，设置环境变量即可：

```powershell
$env:D04_HOME = "D:\D04"
```

所有关键脚本同时支持 `--help` 和命令行路径覆盖。例如：

```powershell
python D04_three_stage_train.py --weights "D:\D04\experiments\detection_embedded\dee_yolov8s_v1\weights\best.pt" --device 0
python D04_compare_latency.py --image-dir "D:\D04\datasets\VisDroneYOLO\images\val" --limit 100
python script\D04_baseline_small_object_eval.py --model "D:\D04\experiments\baseline_640\weights\best.pt" --device 0
```

自定义层会在训练和评估加载模型前自动注册，不再需要手工修改 Ultralytics 包。

## 数据转换

输入目录下应有 `VisDrone2019-DET-train`、`VisDrone2019-DET-val` 等子目录，每个子目录包含 `images/` 和 `annotations/`。每张 JPG 必须有同名 TXT 标注，空标注文件表示背景图。

```powershell
python script\convert_visdrone.py --data-root "F:/data/VisDrone" --output-root "F:/data/VisDroneYOLO" --splits train val
python script\yolo2coco.py --image-dir "F:/data/VisDroneYOLO/images/val" --label-dir "F:/data/VisDroneYOLO/labels/val" --output-json "F:/data/VisDroneYOLO/annotations/val_coco.json"
```

转换只输出项目使用的 10 个类别。类别 11（`others`）和忽略区域会被过滤：

- YOLO 类别编号：`0–9`
- COCO 类别编号：`1–10`（COCO 标准的一基编号）
- 缺失标注、格式错误、非有限数值和非法类别会直接报错，不会再静默生成坏数据
- 标签和 JSON 使用临时文件原子替换，避免中断后留下半写文件

旧数据若使用过 0 基 COCO 类别编号，请重新运行转换。比较模型时必须统一使用同一份重新转换的数据。

## 训练与评估

现有研究和实验参数仍以 VisDrone、`best.pt` 和 `visdrone.yaml` 为前提。训练前检查数据集 YAML 包含 `path`、`train`、`val`、`names`：

```powershell
python D04_three_stage_train.py --help
python D04_three_stage_train.py --allow-cpu --smoke-test --batch 1 --workers 0
```

常规三阶段训练：

```powershell
python D04_three_stage_train.py --device 0 --batch 4 --workers 2
```

COCO 指标评估：

```powershell
python script\D04_baseline_small_object_eval.py --device 0
python script\D04_best_small_object_eval.py --device 0
python script\run_val.py --model "F:/D04/experiments/baseline_640/weights/best.pt" --data "F:/D04/datasets/VisDroneYOLO/visdrone.yaml"
```

评估脚本会生成 `ground_truth.json`、`predictions.json` 和可读的 AP 文本报告。`D04_small_object_eval.py` 保留为向后兼容入口。

## ROI / ROI-SR 流程

基本处理顺序如下，具体参数可按实验配置覆盖：

```powershell
python script\D04_generate_roi.py --device 0
# 使用 Real-ESRGAN 对 roi_crops 做 x2 超分
python script\D04_restore_roi_metadata.py
python script\D04_roi_fusion.py
python script\D04_filter_roi_small32.py
python script\D04_roi_sr_eval.py
```

其余 `D04_latency_eval.py`、`D04_roi_sr_filtered_fps.py`、`D04_error_analysis.py` 和 `D04_error_visualization.py` 保留为实验脚本。它们依赖项目约定的中间文件，运行前请检查脚本顶部的默认路径或通过环境变量 `D04_HOME` 调整根目录。

## 质量保障

- `tests/` 使用合成数据验证 VisDrone 转换、YOLO/COCO 类别一致性、标签校验、原子写入、评估产物和 ROI 过滤。
- `.github/workflows/ci.yml` 在 Ubuntu/Windows、Python 3.10/3.13 上运行语法编译、Pyflakes 和测试。
- 单元测试不加载 GPU、真实数据集或模型权重，因此不能替代训练和性能验证。

改动记录见 [docs/CHANGES.md](docs/CHANGES.md)。
