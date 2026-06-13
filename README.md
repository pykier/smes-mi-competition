# SMES-MI Competition

本仓库用于感觉肌肉电刺激辅助运动想象比赛的代码、数据组织、离线训练和提交接口整理。

当前版本已从临时跑通版升级为正式任务框架：读取 69 通道 DAT 文件，拆分 68 个 EEG 通道和 1 个触发通道，根据触发码切出 trial 的 0–4 s 任务期，构建四个二分类任务，训练四个 EEGNet 模型，并导出 `model_artifacts`。

## 四个任务

```text
vme_left_vs_rest
vme_right_vs_rest
vmi_left_vs_rest
vmi_right_vs_rest
```

输出标签：

```text
0 = rest
1 = target movement or imagery
```

## 关键约束

- 单试次有效窗口不超过 4 s。
- 单次推理时间小于 1 s。
- EEG 通道数不超过 8 个。
- 模型文件大小不超过 150 MB。
- 当前默认不申请校准 trial。
- 不使用非开源外部数据训练或校准。

## 数据格式

根据本地 `*_meta.txt`，当前配置使用：

```text
binary float32 little-endian
sample major
total channels = 69
eeg channels = 68
sampling rate = 1000 Hz
trigger channel = last channel
```

默认选用 8 个运动区通道：

```text
C3, C4, CZ, FC3, FC4, CP3, CP4, FCZ
```

## 快速运行

进入项目并激活环境：

```powershell
cd D:\bisai_all\feel_MI_2026\github\smes-mi-competition
.venv\Scripts\activate
```

修改 `configs/default.yaml`：

```yaml
paths:
  raw_data_dir: D:/bisai_all/feel_MI_2026
```

检查数据：

```powershell
python scripts/inspect_data.py --config configs/default.yaml
```

调试一个 run 的触发码和 trial：

```powershell
python scripts/debug_trials.py --config configs/default.yaml --file D:/bisai_all/feel_MI_2026/sub_1/session1/sub015_sub_1_vme_run1.dat
```

训练四个任务模型：

```powershell
python scripts/train_baseline.py --config configs/default.yaml
```

训练后生成：

```text
outputs/training_result.json
model_artifacts/artifact_config.json
model_artifacts/eegnet_vme_left_vs_rest.pt
model_artifacts/eegnet_vme_right_vs_rest.pt
model_artifacts/eegnet_vmi_left_vs_rest.pt
model_artifacts/eegnet_vmi_right_vs_rest.pt
```

打包：

```powershell
python scripts/package_submission.py
```

输出：

```text
outputs/smes_mi_submission.tar.gz
```

## 提交说明

`outputs/training_result.json` 是本地验证结果，不是平台正式分数。正式分数由比赛平台使用隐藏被试测试集运行提交包后给出。

如果官方提供完整评测框架，应将以下内容放入官方框架后再按官方方式打包：

```text
submission/algorithm_impl.py
src/
model_artifacts/
requirements.txt
```

更详细步骤见：

```text
docs/local_run_guide.md
```
