# Local Run Guide

本指南说明如何在本地 VS Code 中运行当前 EEGNet 框架。

## 1. 克隆仓库

在 Windows CMD 或 PowerShell 中执行：

```cmd
D:
cd D:\
git clone https://github.com/pykier/smes-mi-competition.git
cd smes-mi-competition
```

如果你已经克隆过，进入项目后更新代码：

```cmd
git pull
```

## 2. 放置本地数据

不要把原始数据上传 GitHub。请在本地保持如下结构：

```text
smes-mi-competition/
└── data/
    └── raw/
        └── feel_MI_2026/
            ├── sub_1/
            │   ├── session1/
            │   │   ├── xxx_vme_run1.dat
            │   │   ├── xxx_vme_run1_meta
            │   │   ├── xxx_vmi_run1.dat
            │   │   └── xxx_vmi_run1_meta
            │   └── session2/
            ├── sub_2/
            └── ...
```

也就是说，把你的整个 `feel_MI_2026` 文件夹复制到：

```text
D:\smes-mi-competition\data\raw\
```

## 3. 创建 Python 环境

推荐 Python 3.10 或 3.11。

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果安装 PyTorch 很慢或失败，可以先去 PyTorch 官网按你的 CUDA/CPU 情况复制安装命令。只用 CPU 也可以先跑通框架。

## 4. 在 VS Code 中打开项目

```cmd
code .
```

然后在 VS Code 右下角选择解释器：

```text
.venv\Scripts\python.exe
```

## 5. 先做无数据 smoke test

这个步骤不依赖真实数据，只检查 EEGNet 和 PyTorch 是否能正常前向、反向传播。

```cmd
python scripts/smoke_test.py
```

看到类似输出即可：

```text
Smoke test passed.
logits shape: (8, 2)
loss: ...
```

## 6. 检查本地数据是否能被扫描

```cmd
python scripts/inspect_data.py --config configs/default.yaml
```

它会输出：

- 扫描到多少个 `.dat` 文件。
- 根据文件名推断出的标签。
- 每个 `.dat` 的大小。
- 按 `n_channels=64` 和 `float32` 推算是否能整除。

如果 `remainder` 不是 0，说明当前 `configs/default.yaml` 中的 `dat_dtype` 或 `n_channels` 可能不对，需要调整。

## 7. 训练 EEGNet 基线

默认配置为了先跑通，只读取前 4 个 `.dat` 文件，每个文件最多取 20 个 4 秒窗口。

```cmd
python scripts/train_baseline.py --config configs/default.yaml
```

训练结束后会生成：

```text
outputs/eegnet_model.pt
outputs/training_result.json
```

## 8. 单个 DAT 文件推理

示例：

```cmd
python scripts/run_inference.py --model outputs/eegnet_model.pt --dat data/raw/feel_MI_2026/sub_1/session1/sub015_sub_1_vmi_run1.dat
```

程序会输出该连续文件切成多个窗口后的预测标签。

## 9. 当前框架的重要说明

当前版本是“框架优先、跑通优先”的 EEGNet baseline：

1. 标签暂时从文件名推断。
   - `vmi` 或 `mi` 会被归为 `mi`。
   - `vme` 或 `me` 会被归为 `me`。
   - `rest`、`idle`、`baseline` 会被归为 `rest`。
2. 如果官方真实标签在 `_meta` 或事件标记中，后续需要改 `src/data_io.py`，用事件标记切 trial。
3. 当前默认只选前 8 个通道，满足比赛通道数限制，但不代表这是最优通道选择。
4. 当前默认 1000 Hz 读取，8-30 Hz 滤波，降采样到 250 Hz，再切 4 秒窗口。
5. 当前 EEGNet 只作为基线模型，不应直接视为最终高分方案。

## 10. 常见问题

### 问题 1：Raw data directory not found

检查 `configs/default.yaml`：

```yaml
paths:
  raw_data_dir: data/raw/feel_MI_2026
```

确认本地确实存在：

```text
smes-mi-competition/data/raw/feel_MI_2026
```

### 问题 2：No DAT files matched

确认 `.dat` 文件确实在 `feel_MI_2026` 子目录下，而不是多套了一层目录。

### 问题 3：DAT 文件不能 reshape

优先运行：

```cmd
python scripts/inspect_data.py --config configs/default.yaml
```

然后根据输出调整：

```yaml
data:
  n_channels: 64
  dat_dtype: float32
  dat_layout: sample_major
```

可能需要尝试：

```yaml
dat_dtype: int16
```

或：

```yaml
dat_layout: channel_major
```

### 问题 4：训练很慢

先降低：

```yaml
data:
  max_files: 2
  max_windows_per_file: 5
training:
  epochs: 2
```

确认流程无误后再逐步增大。
