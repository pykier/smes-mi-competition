# SMES-MI Competition

本仓库用于感觉肌肉电刺激辅助运动想象（SMES-MI）算法比赛的代码、数据组织与实验管理。

当前版本已经加入一个可运行的 EEGNet baseline。目标不是追求最终最高分，而是先完成完整工程链路：本地数据扫描、DAT 文件读取、预处理、4 秒窗口切分、EEGNet 训练、模型保存、单文件推理与结果记录。

## 任务约束

- 单试次有效分析窗口不超过 4 s。
- 单次推理时间需小于 1 s。
- 使用 EEG 通道数不超过 8 个。
- 模型文件大小不超过 150 MB。
- 校准数据规模有限，应避免依赖大规模被试内训练。
- 不使用非开源外部数据进行训练或校准。

## 项目结构

```text
smes-mi-competition/
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── competition_notes.md
│   ├── design_plan.md
│   └── local_run_guide.md
├── outputs/
├── scripts/
│   ├── inspect_data.py
│   ├── smoke_test.py
│   ├── train_baseline.py
│   └── run_inference.py
└── src/
    ├── config.py
    ├── data_io.py
    ├── dataset.py
    ├── preprocess.py
    ├── eegnet.py
    ├── train.py
    ├── predict.py
    └── evaluate.py
```

## 本地数据放置

真实比赛数据不要上传 GitHub。请在本地放成：

```text
data/raw/feel_MI_2026/sub_1/session1/*.dat
data/raw/feel_MI_2026/sub_1/session1/*_meta
data/raw/feel_MI_2026/sub_1/session2/*.dat
...
```

如果你的数据目录不同，修改：

```yaml
paths:
  raw_data_dir: data/raw/feel_MI_2026
```

## 快速运行

安装依赖：

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

先做无数据 smoke test：

```cmd
python scripts/smoke_test.py
```

检查本地数据：

```cmd
python scripts/inspect_data.py --config configs/default.yaml
```

训练 EEGNet baseline：

```cmd
python scripts/train_baseline.py --config configs/default.yaml
```

对单个 DAT 文件推理：

```cmd
python scripts/run_inference.py --model outputs/eegnet_model.pt --dat data/raw/feel_MI_2026/sub_1/session1/sub015_sub_1_vmi_run1.dat
```

更详细的 VS Code 本地运行步骤见：

```text
docs/local_run_guide.md
```

## 当前实现说明

当前代码为了优先跑通，采用以下假设：

1. `.dat` 为二进制连续 EEG 文件。
2. 默认 64 通道，1000 Hz。
3. 默认数据类型为 `float32`，排列方式为 `sample_major`。
4. 默认只取前 8 个通道。
5. 默认从文件名推断标签：`vmi/mi` 为 `mi`，`vme/me` 为 `me`，`rest/idle/baseline` 为 `rest`。
6. 默认 8-30 Hz 滤波，降采样至 250 Hz，切成 4 秒窗口。
7. 默认只读取前 4 个文件、每文件最多 20 个窗口，便于先验证流程。

如果真实标签在 `_meta` 或事件标记中，后续需要把 `src/data_io.py` 中的文件名标签逻辑替换为事件标签解析逻辑。
