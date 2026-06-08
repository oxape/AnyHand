# AnyHand 快速开始（oxape fork / CUDA 12.8）

面向本仓库 `semg-rgb-annotation` 分支：作为 **semg-rgbd** 项目的 vendored 手部姿态后端（HaMeR / 未来 AnyHand-Net-D）。  
多模态采集与标注架构见兄弟仓库 [**semg-rgbd**](../semg-rgbd/docs/ARCHITECTURE.md)（sEMG + RGB-D）。

上游 README 仍写 cu118 / `torch<2.6`；**本 fork 实际使用 cu128 + PyTorch 2.11**（`rgb_predictor.py` 已处理 `weights_only` 兼容性）。

---

## 1. 克隆

```bash
git clone --recurse-submodules -b semg-rgb-annotation https://github.com/oxape/AnyHand.git
cd AnyHand
```

SSH：

```bash
git clone --recurse-submodules -b semg-rgb-annotation git@github.com:oxape/AnyHand.git
cd AnyHand
```

已克隆但未拉 submodule：

```bash
git submodule update --init --recursive
```

仅 HTTPS、WiLoR submodule 拉取失败时：

```bash
git config submodule.WiLoR.url https://github.com/oxape/WiLoR.git
git submodule sync
git submodule update --init --recursive
```

Submodule 来源：

| 路径 | 远程 |
|------|------|
| `WiLoR/` | `oxape/WiLoR` |
| `third_party/hamer/` | `geopavlakos/hamer`（官方） |

---

## 2. Python 环境

需要 **Python 3.10**（与 HaMeR / WiLoR 一致）。

```bash
python3.10 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install -U pip
```

---

## 3. 安装 PyTorch（CUDA 12.8）

版本见仓库根目录 `requirements-oxape-cu128.txt`（避免与上游 / submodule 的 `requirements.txt` 重名）：

| 包 | 版本 |
|----|------|
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| torchaudio | 2.11.0+cu128 |

```bash
pip install -r requirements-oxape-cu128.txt --index-url https://download.pytorch.org/whl/cu128
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望: 2.11.0+cu128 True
```

其余 Python 依赖由 `prepare_hamer.sh` / `prepare_wilor.sh` 安装（含 submodule 可编辑包）。

---

## 4. 准备 HaMeR（推荐本分支主路径）

**Linux / macOS / Git Bash（已识别 Windows 版 `.venv`）：**

```bash
bash scripts/prepare_hamer.sh
```

**Windows PowerShell（推荐原生 Windows 环境）：**

```powershell
# 在仓库根目录；先激活 venv 或确保 .venv\Scripts\python.exe 存在
.\scripts\prepare_hamer.ps1
```

若 bash 脚本报 `Python not found` 但已有 `.venv`，多半是路径问题：Windows venv 在 `.venv\Scripts\python.exe`，不是 Linux 的 `.venv/bin/python`。可改用上面的 `.ps1`，或拉取已修复的 `prepare_hamer.sh`。

脚本会：

- 初始化 `third_party/hamer` 及嵌套 ViTPose
- `pip install -e third_party/hamer` 与 ViTPose
- 从 HuggingFace 下载 `anyhand_hamer.ckpt`、`detector.pt` 等

### MANO（手动，许可证限制）

按 [README.md](README.md) §1.3 注册并下载 MANO，解压到：

```
mano_data/
├── MANO_RIGHT.pkl
├── MANO_LEFT.pkl
└── mano_mean_params.npz   # prepare_hamer.sh 也会尝试下载
```

---

## 5. （可选）WiLoR

若需要 WiLoR 后端或 mesh 渲染：

```bash
bash scripts/prepare_wilor.sh
```

---

## 6. 冒烟测试

单张图片（HaMeR）：

```bash
python semg_annotation/test_image_hamer.py assets/demo_img_hamer.jpg
```

摄像头实时预览：

```bash
python semg_annotation/webcam_hamer.py --every-n 2
```

代码调用：

```python
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # 无显示器时

from scripts.rgb_predictor import AnyHandPredictor

predictor = AnyHandPredictor(backend="hamer")
hands = predictor.predict("photo.jpg")
```

---

## 7. 目录布局（安装完成后）

```
AnyHand/
├── .venv/
├── WiLoR/                      # submodule
├── third_party/hamer/          # submodule（推理用这个）
├── pretrained_models/          # prepare 脚本下载（gitignore）
├── mano_data/                  # 手动 + prepare 下载（gitignore）
├── scripts/rgb_predictor.py
├── semg_annotation/
└── requirements-oxape-cu128.txt
```

---

## 8. 更新代码

```bash
git pull
git submodule update --init --recursive
```

若作者 bump 了 submodule 指针，上述第二条必须执行。

---

## 9. Windows 与 submodule 说明

| 组件 | Windows 支持 |
|------|----------------|
| `git submodule`（WiLoR、hamer、ViTPose） | 支持，与平台无关 |
| PyTorch cu128 | 支持（NVIDIA 驱动 + CUDA 12.x） |
| HaMeR 推理 | 一般可行；需装好 PyTorch、MANO、checkpoint |
| ViTPose (`pip install -e`) | 可能较慢；部分环境需 [mmcv](https://github.com/open-mmlab/mmcv) 预编译 wheel |
| `chumpy`（git 依赖） | 偶发编译问题；可尝试先装 `pip install Cython` |
| `pyrender` / 无头渲染 | 服务器无显示器时需配置 EGL/OSMesa；桌面 Windows 通常可直接用 |

**更省事的选择**：在 **WSL2** 里按 Linux 流程走（与当前 `.venv` 的 `bin/python` 布局一致）。

---

## 10. 常见问题

**`torch.load` / `weights_only` 报错**  
本 fork 的 `rgb_predictor.py` 已对 HaMeR checkpoint 传入 `weights_only=False`。请使用本仓库代码，不要混用上游未修补版本。

**Submodule commit not found**  
确认 `oxape/WiLoR` 上存在 AnyHand 记录的 commit（见 `git ls-tree HEAD WiLoR`）。

**无 GPU / 无显示器**  
- CPU：PyTorch 改用 [pytorch.org](https://pytorch.org/get-started/locally/) 的 CPU 命令  
- 无头渲染：`export PYOPENGL_PLATFORM=egl`

---

更完整的 API 说明见 [README.md](README.md)。
