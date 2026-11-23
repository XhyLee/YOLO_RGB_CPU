# YOLO RGB 火焰与烟雾检测（CPU）

基于 Ultralytics YOLO11s，仅使用 RGB 图像，检测 `fire`（0）和 `smoke`（1）。支持电脑 CPU 训练、评估、ONNX 导出，以及面向 Orange Pi 的 ONNX CPU 图片/摄像头推理。

## 下载数据和模型

[v1.0.0 下载页面](https://github.com/XhyLee/YOLO_RGB_CPU/releases/tag/v1.0.0)

| 附件 | 解压/放置位置 |
| --- | --- |
| `rgb_dataset.zip` | 解压到项目根目录，得到 `data/images`、`data/labels` |
| `trained_models.zip` | 解压到项目根目录，得到 `models/best.pt`、`models/last.pt` 和训练来源记录 |
| `yolo11s.pt` | 放到项目根目录；用于从预训练模型开始训练 |
| `training_results.zip` | 解压到项目根目录，得到原训练结果、曲线及预测示例；不包含重复权重 |
| `SHA256SUMS.txt` | 附件 SHA-256 校验值 |

数据含 1280 对图像和标签：训练 841、验证 191、测试 248。来源与标注清理记录见 `records/`。数据、权重和训练结果通过 Release 保存，虚拟环境与缓存无需下载。

## 快速使用（Windows PowerShell）

```powershell
git clone https://github.com/XhyLee/YOLO_RGB_CPU.git
cd YOLO_RGB_CPU
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 下载并解压上述数据集和训练模型后：
.\.venv\Scripts\python.exe evaluate.py --weights models/best.pt --split val --imgsz 640
# 或重新训练：
.\.venv\Scripts\python.exe train.py --epochs 50 --imgsz 640 --batch 4 --threads 8
# 导出板端推理模型：
.\.venv\Scripts\python.exe export_onnx.py --imgsz 320
```

完整安装、训练、导出与开发板使用步骤见 [README_从这里开始.md](README_从这里开始.md)。训练和评估脚本会自动将 `data/data.yaml` 的路径更新为当前项目位置。

## 本次上传状态

- 附件来自现有 `runs/train_20261118_134455` 的 50 轮训练；上传过程没有重新训练。
- 末轮验证记录：precision 0.56858、recall 0.50626、mAP50 0.52124、mAP50-95 0.24565。它们是第 50 轮记录，不代表 `best.pt` 的独立评估结果，也不是测试集成绩。
- 尚无随项目提供的已导出 ONNX，开发板速度和实景性能仍需实测。
- 原详细说明及 `VERIFIED.json` 是项目准备阶段记录，其中“尚未训练/安装”的描述属于历史状态；当前模型以本次 Release 和 `models/training_run.json` 为准。

Ultralytics 依赖及其模型的使用条件请查阅 [Ultralytics](https://github.com/ultralytics/ultralytics)。原数据的来源记录保留在 `records/`。
