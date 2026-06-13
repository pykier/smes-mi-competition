# Design Plan

本文件用于记录项目设计思路。当前版本只定义工程框架，不实现具体算法。

## 总体流程

```text
raw EEG data
    ↓
trial extraction
    ↓
preprocessing
    ↓
channel selection
    ↓
feature extraction
    ↓
classifier training
    ↓
inference and evaluation
```

## 模块划分

### 1. 数据读取

目标文件：`src/data_io.py`

职责：

- 读取原始 EEG 数据。
- 解析事件标记和标签。
- 按 trial 输出统一格式的数据结构。

### 2. 预处理

目标文件：`src/preprocess.py`

职责：

- 选择时间窗。
- 滤波。
- 降采样。
- 标准化。
- 伪迹或异常 trial 检查。

### 3. 特征提取

目标文件：`src/features.py`

职责：

- 提取频带能量、CSP、FBCSP、协方差矩阵或其他轻量特征。
- 保持接口统一，便于替换不同方法。

### 4. 模型训练

目标文件：`src/model.py`、`src/train.py`

职责：

- 封装分类器。
- 支持保存和加载模型。
- 保证模型大小和推理速度满足比赛要求。

### 5. 推理与评估

目标文件：`src/predict.py`、`src/evaluate.py`

职责：

- 对单 trial 或批量 trial 输出预测类别。
- 统计准确率、混淆矩阵、单次推理耗时。

## 初始阶段不做的事情

- 不预设最终算法。
- 不写死数据格式。
- 不上传真实数据。
- 不引入复杂深度学习框架。
- 不在代码中写死绝对路径。

## 推荐实验记录字段

每次实验应至少记录：

- 数据版本。
- 使用通道。
- 时间窗。
- 滤波频段。
- 特征类型。
- 分类器类型。
- 交叉验证方式。
- 随机种子。
- 准确率或比赛得分。
- 平均推理时间。
- 模型文件大小。
