# YOLO11s：RGB 火焰/烟雾检测完整步骤

项目：`C:\Users\12541\Desktop\YOLO_RGB_CPU`。
默认模型已改成 `yolo11s.pt`。仅用 RGB 数据，电脑 CPU 训练，Orange Pi 4 Pro（4GB）推理。

当前已完成数据准备、训练/评估/导出/板端 ONNX 推理脚本。没有安装训练依赖、没有启动训练、没有实际导出模型，也没有连接开发板测试。

## 1. 确认项目内容

```text
YOLO_RGB_CPU/
  data/
    images/train/    841 张
    images/val/      191 张
    images/test/     248 张
    labels/train/
    labels/val/
    labels/test/
    data.yaml
  train.py
  evaluate.py
  export_onnx.py
  requirements.txt
  models/           正式训练完成后生成 best.pt
  deploy/
    infer_onnx.py
    requirements-board.txt
    sample.jpg      来自验证集的演示图片
  records/          来源和标注清理记录
```

类别为 0=fire、1=smoke。数据已清理，无需再移动图片或重新转换标签。
每种模态原有 1280 张，这个项目只包含 RGB。划分依据是文件名前缀推断的视频组，尚未核实不同前缀一定是独立场景。

YOLO11s 比 YOLO11n 计算量更大。电脑配置为 Ryzen 7 8845HS、8 核 16 线程、约 31.3 GiB 可见内存，可用 CPU 训练；4GB 板子的帧率与内存占用仍需实测。

## 2. 在 VS Code 打开文件夹

1. VS Code → 文件 → 打开文件夹 → 桌面的 `YOLO_RGB_CPU`。
2. 扩展面板安装 Microsoft 的 Python 扩展。
3. 终端 → 新建终端，使用 PowerShell。

## 3. 创建 Python 虚拟环境

下面命令逐条执行。项目已有可用的 Python 3.10.11，无需先装 Anaconda。

```powershell
cd "C:\Users\12541\Desktop\YOLO_RGB_CPU"
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

如果终端找不到 python，可将创建环境命令替换为：

```powershell
& "E:\python_3.10\python.exe" -m venv .venv
```

如果 `.venv` 已创建且可正常使用，跳过重复创建。
按 Ctrl+Shift+P → `Python: Select Interpreter` → 选择项目的 `.venv\Scripts\python.exe`。
下文显式调用虚拟环境 Python，不必运行 Activate.ps1，也不用改执行策略。

## 4. 安装 CPU 训练依赖

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch,ultralytics; print('torch=',torch.__version__,'YOLO=',ultralytics.__version__,'CUDA=',torch.cuda.is_available())"
```

先安装 CPU 版 torch，再装 requirements；pip 会选择兼容本机 Python 的 torch/torchvision。最后 CUDA=False 是正常结果。如果 pip check 报依赖冲突，先处理报错，再开始训练。

保存当前环境版本：

```powershell
.\.venv\Scripts\python.exe -m pip freeze > environment-lock.txt
```

首次训练会从网络下载 `yolo11s.pt`，应保持网络可用。下载失败时需要解决网络或将可信来源的同名预训练权重放在项目根目录，不需要换成 CUDA 安装包。

## 5. 运行一次快速检查

```powershell
.\.venv\Scripts\python.exe train.py --quick
```

这会用 YOLO11s、CPU、320 输入、batch=2、约 10% 的训练图跑 1 轮。仍使用 val 集验证。
确认能完成加载、训练和验证，没有损坏图、路径或依赖报错。它仅检查流程，快速权重在 `runs/quick_时间/weights`，不会复制成正式 best.pt。

## 6. 正式训练 YOLO11s

```powershell
.\.venv\Scripts\python.exe train.py --epochs 50 --imgsz 640 --batch 4 --threads 8
```

默认就是上述参数：CPU、workers=0、cache=False、amp=False、seed=42、patience=15。
50 轮是初始设置，不保证必然收敛，验证效果连续不改善时可提前停止。
使用 640 训练是为了尽量保留小火焰细节；不等于板上必须使用 640。

保持接电、散热，关闭自动睡眠。看正式训练前几轮的实际时间再估算，快速检查使用了子集和较小输入，不能简单乘 50。

如果 CPU 太慢，可以另开一次 416 输入训练做比较：

```powershell
.\.venv\Scripts\python.exe train.py --epochs 50 --imgsz 416 --batch 4 --threads 8
```

不要同时运行这两条训练。小输入可能降低小火焰召回率，需要用 val 比较。

每次训练保存在独立的 `runs/train_时间` 中；完成后本次最佳权重复制到 `models/best.pt`，最后权重复制到 `models/last.pt`。重复正式训练会更新这两个方便使用的副本，旧模型仍在各自 runs 目录里。

中断后从原训练目录恢复：

```powershell
.\.venv\Scripts\python.exe train.py --resume "runs\train_实际时间\weights\last.pt"
```

把“实际时间”替换为实际目录。恢复的是原来的设置，不要拿快速检查的 last.pt 当正式恢复点；正常完成的权重可能已去掉优化器状态，追加训练需另行设置微调方案。

## 7. 检查验证集精度和预测图片

```powershell
.\.venv\Scripts\python.exe evaluate.py --imgsz 640
```

如果正式训练用了 416，可先使用 `--imgsz 416` 对应评估。
查看控制台和 `runs/eval_val_...` 的 precision、recall、mAP50、mAP50-95、混淆矩阵、预测叠框图。分别检查 fire 和 smoke；特别看烟雾误报和小火焰漏检。
不要只看总体 mAP，也不要用训练图准确率判断部署效果。

## 8. 导出 ONNX，并比较部署尺寸

先尝试 320 输入：

```powershell
.\.venv\Scripts\python.exe export_onnx.py --imgsz 320
.\.venv\Scripts\python.exe evaluate.py --weights deploy/rgb_yolo11s_320.onnx --imgsz 320
```

导出结果是 `deploy/rgb_yolo11s_320.onnx` 和同名 JSON。模型为静态 FP32、batch=1、opset=12、不含 NMS。导出脚本检查模型结构并执行一次零输入 CPU 推理；第二条命令再使用真实 val 图片测精度。

可再试 416：

```powershell
.\.venv\Scripts\python.exe export_onnx.py --imgsz 416
.\.venv\Scripts\python.exe evaluate.py --weights deploy/rgb_yolo11s_416.onnx --imgsz 416
```

静态 ONNX 的评估输入尺寸必须与导出尺寸一致。
结合 val 精度和板子速度选择 320 或 416；640 也可导出，但 YOLO11s 在 4GB 板上用 CPU 跑 640 的速度可能较低，不能事先承诺实时。

## 9. 在电脑上先检查独立推理脚本

下面示例使用 320 导出模型：

```powershell
.\.venv\Scripts\python.exe deploy/infer_onnx.py --source deploy/sample.jpg --output deploy/prediction_pc.jpg
```

检查输出图中的类别和框是否合理。该脚本完成 letterbox、RGB 转换、归一化、CPU ONNX 推理、置信度过滤、按类别 NMS 和坐标还原，不依赖 PyTorch。
默认模型是脚本旁边的 `rgb_yolo11s_320.onnx`；如果最终选 416，要传入 `--model deploy/rgb_yolo11s_416.onnx`。
图片命令输出模型推理时间和预处理+模型+NMS时间；它们不含图片读写时间，也不是板子性能。

## 10. 确定方案后，最终测试一次

如果最后选择 320：

```powershell
.\.venv\Scripts\python.exe evaluate.py --weights deploy/rgb_yolo11s_320.onnx --imgsz 320 --split test
```

416 时同时替换文件名和 imgsz。test 留作最终评估，不应反复依据 test 调参。

## 11. 确认 Orange Pi 系统

以下命令在板子终端执行，不是在 Windows PowerShell：

```bash
tr -d '\0' < /proc/device-tree/model; echo
tr -d '\0' < /proc/device-tree/compatible; echo
uname -m
cat /etc/os-release
python3 --version
```

下文 CPU 部署示例适用于 Debian/Ubuntu 类的 64 位 ARM Linux（`uname -m` 为 `aarch64`），并以 Python 3.10/3.11/3.12 为起点。
如果是 32 位系统、Android，或 Python/ONNX Runtime 没有对应 wheel，不要照搬安装命令，需要按实际系统调整。当前尚未连接你的板子核实芯片和系统。

## 12. 把部署文件传到板子

电脑可以通过 U 盘或 VS Code Remote-SSH 复制整个项目的 `deploy` 文件夹到板子的 `~/rgb_yolo11s`。
也可以在 Windows PowerShell 用 scp（替换实际用户名和 IP；板子需已启用 SSH）：

```powershell
scp -r .\deploy "你的板子用户名@你的板子IP:~/rgb_yolo11s"
```

目标目录应直接包含 infer_onnx.py、requirements-board.txt、sample.jpg、导出模型和 JSON。无需复制 data、.venv 或训练环境，也无需在板子上训练。

## 13. 板子安装轻量 CPU 推理环境

以下命令在板子终端执行，假定系统符合第 11 步：

```bash
sudo apt update
sudo apt install -y python3-venv
cd ~/rgb_yolo11s
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-board.txt
.venv/bin/python -m pip check
.venv/bin/python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

此路线使用 CPUExecutionProvider。它不会自动启用 NPU，板上也不需要安装 torch/ultralytics。
依赖是否能成功安装取决于板子系统/Python 版本和网络；遇到 No matching distribution 时先核实架构与版本。

## 14. 板子先测一张图片

```bash
.venv/bin/python infer_onnx.py --source sample.jpg --output prediction_board.jpg --threads 4
```

通过 VS Code Remote-SSH 或复制回电脑查看 prediction_board.jpg，检查是否与电脑输出一致。
320 是默认模型；416 示例：

```bash
.venv/bin/python infer_onnx.py --model rgb_yolo11s_416.onnx --source sample.jpg --output prediction_board_416.jpg --threads 4
```

## 15. 接 USB 摄像头，测连续推理

以下示例用于能被 OpenCV 识别的 USB 摄像头。CSI 摄像头可能需要对应系统驱动和专门取流管线，不能保证用数字 0 直接打开。
在无图形桌面的板子上安装 headless 版本：

```bash
.venv/bin/python -m pip install opencv-python-headless==4.11.0.86
.venv/bin/python infer_onnx.py --source 0 --output result.avi --max-frames 300 --threads 4
```

这会处理 300 帧并保存检测视频；摄像头打开失败时先确认系统识别设备和摄像头编号。
每 10 帧打印包含取帧、预处理、模型、NMS、画框和视频写入的实测循环 FPS。录像回放固定为 10fps，与实测处理速度不同，不要从回放速度推算检测帧率。

如果板上有图形桌面，需要窗口显示，应该改装普通 opencv-python（不要同时安装两种 OpenCV 包）：

```bash
.venv/bin/python -m pip uninstall -y opencv-python-headless
.venv/bin/python -m pip install opencv-python==4.11.0.86
.venv/bin/python infer_onnx.py --source 0 --show --max-frames 0 --threads 4
```

按 q 退出。无桌面或普通 SSH 会话使用前面的保存视频方式。

另开一个板子终端，用 `free -h`、`top` 观察内存和 CPU；持续运行时也应检查散热。4GB 内存不能单独决定实际帧率。

## 16. CPU 太慢时再启用 NPU 路线

先确认第 11 步得到的实际芯片、系统和 NPU 驱动，再决定是否适用 Rockchip RKNN。不要根据板子名字猜 `rk3576` 或 `rk3588` 参数。
ONNX CPU 路线用于先跑通和测基线，YOLO11s 在 ARM CPU 上的实时性能不保证。

若实际硬件/驱动支持：在电脑的 x86-64 Linux 或 WSL2 中按兼容版本安装 RKNN Toolkit2，使用 best.pt 或合适的导出图转换到对应芯片，取代表性的训练图做 INT8 校准，再在板上安装匹配运行时并比较精度、内存、FPS。
转换通常无需 GPU，但不能原生在 Windows 安装 RKNN Toolkit2；RKNN 输出、归一化和后处理也可能不同，不能直接套用本项目的 ONNX CPU 脚本。NPU 的具体命令须依据实际芯片与系统版本再补充。

## 检查范围与参考

数据复制已校验图像和标签 SHA-256，1280 对齐全。脚本做了语法、帮助命令检查；ONNX 推理的几何变换和 NMS 用构造输入做了功能检查。没有真实模型时无法验证最终推理精度、速度和板端摄像头兼容性。

- CPU PyTorch：https://pytorch.org/get-started/locally/
- 训练：https://docs.ultralytics.com/modes/train/
- ONNX：https://docs.ultralytics.com/integrations/onnx/
- RKNN：https://docs.ultralytics.com/integrations/rockchip-rknn/


## CPU 线程补充

已修正 train.py：Ultralytics 选择 CPU 时会将 PyTorch 线程数重置为最多 8；脚本现在会在训练开始、每轮开始和验证开始后应用 --threads，并打印实际线程数。已经启动的进程不会自动载入修改。新启动可用 `--threads 16` 测试 16 个逻辑线程；16 不保证比 8 快，应比较相同 batch/imgsz 下稳定后的整轮时间。当前默认仍为 8。

新训练命令：

```powershell
.\.venv\Scripts\python.exe train.py --epochs 50 --imgsz 640 --batch 4 --threads 16
```

电脑当前是平衡电源方案，可在接电时自行切换 Windows 电源模式为最佳性能。没有修改系统电源方案或中断正在运行的训练。

Radeon 780M 是 AMD 集成 GPU；当前 torch +cpu 构建不会用它训练，CUDA 也不支持 AMD。不能只改 device=0 就使用核显。DirectML 属于另一条需要改造和验证的路线，不建议直接装进当前训练环境；CPU 与核显也不会在现有脚本里自动合并算力。
