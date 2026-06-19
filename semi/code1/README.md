# DH-CAN Reproduction Code

This folder implements **Dynamic Hierarchical Convolutional Attention Network (DH-CAN)** for motor imagery EEG classification.

The implementation follows the algorithm described in the paper:

> Dynamic Hierarchical Convolutional Attention Network for Recognizing Motor Imagery Intention

Core modules implemented here:

1. **Dynamic temporal convolution**: three temporal convolution branches with kernel sizes tied to the sampling rate. For `sfreq=250`, the default kernels are approximately `64`, `32`, and `16`, matching the paper's multiscale 4/8/16 Hz design idea.
2. **Hierarchical spatial convolution**: one global spatial block over all electrodes, plus multiple local spatial blocks over predefined brain regions.
3. **Symmetric region parameter sharing**: left/right symmetric region pairs can share local spatial convolution parameters when their region sizes are equal.
4. **Region-level graph attention**: a dense GAT/GATv2-like module learns inter-region connectivity without requiring `torch_geometric`.
5. **High-level fusion classifier**: global and GAT-optimized local features are fused by a convolution layer and then classified.

## File structure

```text
semi/code1/
  README.md
  requirements.txt
  dhcan_model.py          # DH-CAN model implementation
  region_config.py        # 22-channel and 16-channel region presets
  data_utils.py           # NPZ dataset, split, metrics
  train.py                # training script for NPZ EEG data
  quick_test.py           # forward/backward smoke test
```

## Install

```bash
pip install -r requirements.txt
```

For only running `quick_test.py`, `torch` and `numpy` are enough.

## Quick test

```bash
cd semi/code1
python quick_test.py
```

Expected output includes:

```text
logits shape: (2, 4)
attention shape: (2, 1, 6, 6)
```

## Input data format

The training script expects an `.npz` file:

```python
X.shape == (n_trials, n_channels, n_samples)
y.shape == (n_trials,)
```

Labels must be integer labels from `0` to `n_classes - 1`.

Optional split arrays are supported:

```python
train_idx, val_idx, test_idx
```

If these arrays are absent, `train.py` will create a stratified train/validation/test split.

## Train on BCI Competition IV 2a style data

For 22-channel, 250 Hz, 4-class MI data:

```bash
python train.py \
  --data ./A01T_dhcan.npz \
  --preset bciciv2a_22 \
  --sfreq 250 \
  --n-classes 4 \
  --epochs 500 \
  --batch-size 32 \
  --device cuda
```

If CPU is very slow because of excessive OpenMP/PyTorch threads, use:

```bash
python train.py --data ./A01T_dhcan.npz --torch-threads 1
```

## Train on the 16-channel MetaBCI montage

The included `meta16` preset uses the common 16 channels:

```text
FC3, FC1, FCz, FC2, FC4,
C5, C3, C1, Cz, C2, C4, C6,
CP3, CP1, CP2, CP4
```

Run:

```bash
python train.py \
  --data ./your_meta16_data.npz \
  --preset meta16 \
  --sfreq 250 \
  --n-classes 2 \
  --epochs 300
```

## Strict reproduction notes

The original paper provides the high-level architecture, table-level tensor sizes, and region segmentation figure, but not official source code. Therefore this implementation reproduces the **algorithmic design** rather than a byte-identical implementation. The important design choices are preserved:

- multiscale dynamic temporal convolution;
- global and local hierarchical spatial feature extraction;
- brain-region grouping;
- symmetric-region parameter sharing;
- region-level GAT connectivity;
- global/local feature fusion classifier.

For strict result reproduction, keep these settings aligned with the paper:

- BCI IV 2a: use 22 EEG channels, 250 Hz, 2--6 s MI segment or cue-aligned 4 s segment depending on your event extraction pipeline;
- run subject-dependent evaluation;
- use one session for training and split the other session into validation/test if following the paper's Dataset-1 setting;
- average over multiple random runs.
