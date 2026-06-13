# Local Run Guide

本指南对应当前“触发器切 trial + 四个正式二分类任务 + EEGNet + model_artifacts 导出”版本。

## 1. 使用 ZIP 版本更新代码

如果 GitHub 网络不能 `git pull`，直接重新下载 ZIP，解压后覆盖旧项目代码即可。注意不要删除你的虚拟环境和本地数据。

建议路径保持为：

```text
D:\bisai_all\feel_MI_2026\github\smes-mi-competition
```

项目根目录下应能看到：

```text
configs
scripts
src
submission
requirements.txt
README.md
```

## 2. 激活已有环境

你已经创建过 `.venv`，以后只需进入项目根目录后激活：

```powershell
cd D:\bisai_all\feel_MI_2026\github\smes-mi-competition
.venv\Scripts\activate
```

确认当前 Python 来自项目环境：

```powershell
python -c "import sys; print(sys.executable)"
```

应输出：

```text
D:\bisai_all\feel_MI_2026\github\smes-mi-competition\.venv\Scripts\python.exe
```

若 ZIP 更新后依赖有变化，执行：

```powershell
python -m pip install -r requirements.txt
```

## 3. 修改数据路径

打开：

```powershell
notepad configs\default.yaml
```

把：

```yaml
paths:
  raw_data_dir: data/raw/feel_MI_2026
```

改成你的本地数据根目录：

```yaml
paths:
  raw_data_dir: D:/bisai_all/feel_MI_2026
```

你的数据结构应类似：

```text
D:\bisai_all\feel_MI_2026\sub_1\session1\sub015_sub_1_vme_run1.dat
D:\bisai_all\feel_MI_2026\sub_1\session1\sub015_sub_1_vme_run1_meta.txt
D:\bisai_all\feel_MI_2026\sub_1\session1\sub015_sub_1_vmi_run1.dat
D:\bisai_all\feel_MI_2026\sub_1\session1\sub015_sub_1_vmi_run1_meta.txt
```

## 4. 检查数据格式和 trigger

执行：

```powershell
python scripts/inspect_data.py --config configs/default.yaml
```

当前正确格式应满足：

```text
total_channels = 69
eeg_channels = 68
trigger last
sample_major
float32
remainder = 0
```

输出里还会显示第一份文件的 trigger 统计，例如 1、2、3、101、241 等事件计数。

如果你想单独调试某一个 run 的 trigger 切分：

```powershell
python scripts/debug_trials.py --config configs/default.yaml --file D:/bisai_all/feel_MI_2026/sub_1/session1/sub015_sub_1_vme_run1.dat
```

理论上每个 run 应接近 30 个 trial。若提取 trial 数异常，应先修正 trigger 解析，不要直接训练。

## 5. 训练四个正式任务模型

执行：

```powershell
python scripts/train_baseline.py --config configs/default.yaml
```

当前会训练四个二分类任务：

```text
vme_left_vs_rest
vme_right_vs_rest
vmi_left_vs_rest
vmi_right_vs_rest
```

标签定义：

```text
0 = rest
1 = target movement / imagery
```

训练输出：

```text
outputs/training_result.json
model_artifacts/artifact_config.json
model_artifacts/eegnet_vme_left_vs_rest.pt
model_artifacts/eegnet_vme_right_vs_rest.pt
model_artifacts/eegnet_vmi_left_vs_rest.pt
model_artifacts/eegnet_vmi_right_vs_rest.pt
```

其中 `outputs/training_result.json` 是本地验证结果，不是平台正式分数。

## 6. 本地验证的意义

当前默认用：

```yaml
training:
  validation_mode: leave_subjects_out
  validation_subjects: ["sub_9", "sub_10", "sub_11"]
```

即训练时留出若干被试做验证，比随机划分更接近跨被试比赛设置。平台正式分数仍以隐藏被试在线评测为准。

## 7. 生成提交压缩包

训练完成后执行：

```powershell
python scripts/package_submission.py
```

会生成：

```text
outputs/smes_mi_submission.tar.gz
```

压缩包内包含：

```text
src/
submission/
model_artifacts/
requirements.txt
README.md
```

如果官方平台要求使用它提供的完整框架，则应把：

```text
submission/algorithm_impl.py
model_artifacts/
src/
```

复制进官方框架对应位置，再运行官方 `run_tests.bat` 或 `debug_pipeline.py`。平台最终提交一般是官方框架的 `.tar.gz` 包，而不是原始训练数据。

## 8. 通道和评分约束

当前默认使用 8 个运动区通道：

```text
C3, C4, CZ, FC3, FC4, CP3, CP4, FCZ
```

满足比赛通道数不超过 8 的要求。当前默认不申请校准 trial：

```text
calibration_trials_per_class = 0
```

模型为小型 EEGNet，通常远小于 150 MB，单 trial 推理时间也应小于 1 秒。

## 9. 后续优化方向

1. 先确认 trigger 切 trial 是否完全正确。
2. 比较不同通道组合：4、6、8 通道。
3. 比较 EEGNet 与 FBCSP+LDA、FBCSP+SVM、Riemannian 方法。
4. 按 subject 做更严格的 leave-one-subject-out 验证。
5. 再将最优模型封装到 `submission/algorithm_impl.py`。
