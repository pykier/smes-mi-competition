# Traditional Model Runs

本文件说明如何分别运行三套传统模型，并比较结果。

## 1. 更新代码

如果不能使用 `git pull`，重新下载 GitHub ZIP 后，把新版代码覆盖到：

```text
D:\bisai_all\feel_MI_2026\github\smes-mi-competition
```

不要删除 `.venv`，不要删除本地数据。

## 2. 激活环境

```powershell
cd D:\bisai_all\feel_MI_2026\github\smes-mi-competition
.venv\Scripts\activate
```

确认配置文件里的数据路径仍然是：

```yaml
paths:
  raw_data_dir: D:/bisai_all/feel_MI_2026
```

## 3. 先检查数据

```powershell
python scripts/inspect_data.py --config configs/default.yaml
```

确保：

```text
remainder = 0
meta files found = 83 / 83
```

## 4. 三次运行

### 4.1 FBCSP + LDA

```powershell
python scripts/train_traditional.py --model fbcsp_lda --config configs/default.yaml --disable-broad-bandpass
```

输出：

```text
outputs/fbcsp_lda_result.json
model_artifacts_fbcsp_lda/
```

### 4.2 FBCSP + SVM

```powershell
python scripts/train_traditional.py --model fbcsp_svm --config configs/default.yaml --disable-broad-bandpass
```

输出：

```text
outputs/fbcsp_svm_result.json
model_artifacts_fbcsp_svm/
```

### 4.3 Riemannian Log-Cov + Logistic Regression

```powershell
python scripts/train_traditional.py --model riemann_lr --config configs/default.yaml --disable-broad-bandpass
```

输出：

```text
outputs/riemann_lr_result.json
model_artifacts_riemann_lr/
```

## 5. 比较结果

```powershell
python scripts/compare_results.py
```

该脚本会输出每个模型的四个任务验证准确率和平均准确率。

## 6. 打包最优模型

假设 `fbcsp_lda` 最优：

```powershell
python scripts/package_submission.py --artifacts model_artifacts_fbcsp_lda --out outputs/fbcsp_lda_submission.tar.gz
```

假设 `riemann_lr` 最优：

```powershell
python scripts/package_submission.py --artifacts model_artifacts_riemann_lr --out outputs/riemann_lr_submission.tar.gz
```

压缩包中会统一使用：

```text
model_artifacts/
submission/algorithm_impl.py
src/
requirements.txt
```

`submission/algorithm_impl.py` 已支持 EEGNet 和传统 joblib 模型两种 artifact。

## 7. 结果解释

`outputs/*_result.json` 是本地跨被试验证结果，不是平台正式分数。平台正式分数仍需上传官方系统，用隐藏测试集评测。

本地优先比较：

```text
mean_best_val_accuracy
四个 task 的 val accuracy
single_trial_inference_time_seconds
```

当前比赛总分还会受通道数、校准 trial 数、推理时间、模型大小影响。当前传统模型默认使用 8 个通道、0 个校准 trial，模型体积通常很小，推理时间通常小于 1 秒。
