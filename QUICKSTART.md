# AnyHand 快速开始（oxape fork / CUDA 12.8）

面向本仓库 `semg-rgb-annotation` 分支：作为 **semg-rgbd** 项目的 vendored 手部姿态后端（HaMeR / 未来 AnyHand-Net-D）。  
多模态采集与标注架构见兄弟仓库 [**semg-rgbd**](../semg-rgbd/docs/ARCHITECTURE.md)（sEMG + RGB-D）。

上游 README 仍写 conda / cu118 / `torch<2.6`；**本 fork 实际使用 cu128 + PyTorch 2.11**（`rgb_predictor.py` 已处理 `weights_only` 兼容性）。

**推荐安装顺序（Windows 已验证）：**

1. `git submodule update --init --recursive`
2. `uv venv --python 3.10` → 激活 `.venv`
3. `pip install -r requirements-oxape-cu128.txt`（cu128 索引）
4. `.\scripts\prepare_wilor.ps1`（检测器 + ultralytics，HaMeR 也必需）
5. `.\scripts\prepare_hamer.ps1`（HaMeR 重建 + checkpoint）
6. 手动放置 `mano_data/MANO_RIGHT.pkl`

Linux / macOS 将 `.ps1` 换成对应的 `.sh` 即可。`prepare_*` 脚本会自动使用仓库根目录下的 `.venv`，**不激活 venv 也可运行**。

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

### 推荐：`uv` 创建虚拟环境

本项目在 Windows 上使用 [uv](https://docs.astral.sh/uv/) 创建初始 venv（与标准 `venv` 布局兼容，`prepare_*` 脚本可直接识别）：

```bash
# 安装 uv: https://docs.astral.sh/uv/getting-started/installation/
uv venv --python 3.10
```

激活：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

升级 pip（可选，uv 创建的 venv 通常已带 pip）：

```bash
python -m pip install -U pip
```

### 备选：标准库 `venv`

```bash
python3.10 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
python -m pip install -U pip
```

也可用 `uv pip` 代替 `pip` 安装下文依赖，例如：  
`uv pip install -r requirements-oxape-cu128.txt --index-url https://download.pytorch.org/whl/cu128`

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

其余 Python 依赖由下面的 prepare 脚本安装（含 submodule 可编辑包）。

---

## 4. 准备 WiLoR 检测器（必需）

即使只用 **HaMeR 后端**，`AnyHandPredictor` 也会先加载 WiLoR 的 YOLO 手部检测器（`ultralytics` + `pretrained_models/detector.pt`）。因此 **`prepare_wilor` 不是可选项**，需在 HaMeR 推理前完成。

**Linux / macOS / Git Bash：**

```bash
bash scripts/prepare_wilor.sh
```

**Windows PowerShell：**

```powershell
.\scripts\prepare_wilor.ps1
```

脚本会：

- 初始化 `WiLoR/` submodule
- 先以 `--no-build-isolation` 安装 `chumpy`，再安装 `WiLoR/requirements.txt` 的其余依赖（含 `ultralytics`）
- 下载 `pretrained_models/detector.pt`

若还需 **WiLoR 重建后端** 或 mesh 渲染，同一脚本也会下载 `anyhand_wilor.ckpt` 与 `model_config_wilor.yaml`。

---

## 5. 准备 HaMeR

**Linux / macOS / Git Bash：**

```bash
bash scripts/prepare_hamer.sh
```

**Windows PowerShell：**

```powershell
.\scripts\prepare_hamer.ps1
```

若 bash 脚本报 `Python not found` 但已有 `.venv`，多半是路径问题：Windows venv 在 `.venv\Scripts\python.exe`，不是 Linux 的 `.venv/bin/python`。可改用上面的 `.ps1`，或拉取已修复的 `prepare_hamer.sh`。

脚本会：

- 初始化 `third_party/hamer` 及嵌套 ViTPose
- 先以 `--no-build-isolation` 安装 `chumpy`，再 `pip install -e third_party/hamer` 与其余核心依赖、ViTPose
- 从 HuggingFace 下载 `anyhand_hamer.ckpt`、`model_config.yaml`
- 下载 `mano_data/mano_mean_params.npz`（bash 版若发现旧路径 `pretrained_models/hamer_ckpts/data/` 会自动迁移）

### MANO（手动，许可证限制）

按 [README.md](README.md) §1.3 注册并下载 MANO，解压到：

```
mano_data/
├── MANO_RIGHT.pkl
├── MANO_LEFT.pkl
└── mano_mean_params.npz   # prepare_hamer 也会尝试下载
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
├── WiLoR/                      # submodule（检测器 + 可选 WiLoR 后端）
├── third_party/hamer/          # submodule（HaMeR 重建）
├── pretrained_models/          # detector.pt、checkpoint 等（gitignore）
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
| `uv venv --python 3.10` | 推荐；生成标准 `.venv`，与 `prepare_*.ps1` 兼容 |
| `git submodule`（WiLoR、hamer、ViTPose） | 支持，与平台无关 |
| `prepare_wilor.ps1` / `prepare_hamer.ps1` | 推荐原生 PowerShell；已修复下文已知问题 |
| PyTorch cu128 | 支持（NVIDIA 驱动 + CUDA 12.x） |
| HaMeR 推理 | 一般可行；需装好 PyTorch、MANO、checkpoint、`detector.pt` |
| ViTPose (`pip install -e`) | 可能较慢；部分环境需 [mmcv](https://github.com/open-mmlab/mmcv) 预编译 wheel |
| `chumpy`（git 依赖） | `prepare_hamer` / `prepare_wilor` 已用 `--no-build-isolation` 自动处理；`prepare_wilor` 会从 `requirements.txt` 中排除 chumpy 行避免重复构建 |
| `pyrender` / 无头渲染 | 服务器无显示器时需配置 EGL/OSMesa；桌面 Windows 通常可直接用 |

**更省事的选择**：在 **WSL2** 里按 Linux 流程走（`uv venv` 或 `python -m venv` 均可）。

---

## 10. 常见问题

**`chumpy` 安装报 `No module named 'pip'`**  
现代 pip 的 PEP 517 构建隔离与 chumpy 老旧 `setup.py` 不兼容（`setup.py` 在构建阶段 `import pip`）。  
**已修复：** 最新 `prepare_hamer.ps1` / `prepare_wilor.ps1`（及 `.sh`）会先 `--no-build-isolation` 安装 chumpy；`prepare_wilor` 还会从 `WiLoR/requirements.txt` 过滤掉 chumpy 行再 `pip install -r`。  
手动绕过：  
`pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"`

**`prepare_hamer.ps1` 报 `parameter name 'e' is ambiguous`**  
**已修复：** PowerShell 会把 `Pip-Install -e ...` 的 `-e` 误认为 `-ErrorAction`。脚本已改为数组传参，例如 `Pip-Install @('-e', $HamerDir, '--no-deps')`。

**`prepare_hamer.ps1` 报 PyTorch `IndentationError` 或 pip 提示 “give at least one requirement”**  
**已修复：** 旧版 `Invoke-PythonCli` 用 `ValueFromRemainingArguments` 会把 `@('-c', '...')` 合并成单个参数；`pip install` 的参数拼接优先级也有误。请使用最新脚本。

**`torch.load` / `weights_only` 报错**  
本 fork 的 `rgb_predictor.py` 已对 HaMeR checkpoint 传入 `weights_only=False`。请使用本仓库代码，不要混用上游未修补版本。

**Submodule commit not found**  
确认 `oxape/WiLoR` 上存在 AnyHand 记录的 commit（见 `git ls-tree HEAD WiLoR`）。

**`No module named 'ultralytics'`**  
未运行 `prepare_wilor`（或 `.ps1`）。HaMeR 后端同样需要 WiLoR 的 YOLO 检测器，请先完成 §4。

**无 GPU / 无显示器**  
- CPU：PyTorch 改用 [pytorch.org](https://pytorch.org/get-started/locally/) 的 CPU 命令  
- 无头渲染：`export PYOPENGL_PLATFORM=egl`

---

更完整的 API 说明见 [README.md](README.md)。
