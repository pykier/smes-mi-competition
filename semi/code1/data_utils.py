"""Data utilities for DH-CAN.

The training script expects an NPZ file with:
    X: float array, shape (n_trials, n_channels, n_samples)
    y: int array, shape (n_trials,), labels 0..n_classes-1
Optional:
    train_idx, val_idx, test_idx: integer arrays for explicit splits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


class EEGNpzDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, normalize: str = "trial") -> None:
        if X.ndim != 3:
            raise ValueError(f"X must have shape (N, C, T), got {X.shape}")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must have shape (N,) and match X length")
        self.X = X.astype(np.float32, copy=False)
        self.y = y.astype(np.int64, copy=False)
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        x = self.X[idx]
        if self.normalize == "trial":
            mean = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-6
            x = (x - mean) / std
        elif self.normalize == "none":
            pass
        else:
            raise ValueError(f"Unknown normalize mode: {self.normalize}")
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


@dataclass
class NpzData:
    X: np.ndarray
    y: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def stratified_split(y: np.ndarray, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        n_val = int(round(len(idx) * val_ratio))
        test_idx.extend(idx[:n_test])
        val_idx.extend(idx[n_test:n_test + n_val])
        train_idx.extend(idx[n_test + n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def load_npz_data(path: str, val_ratio: float = 0.2, test_ratio: float = 0.2, seed: int = 42) -> NpzData:
    data = np.load(path)
    if "X" not in data or "y" not in data:
        raise KeyError("NPZ must contain arrays named 'X' and 'y'.")
    X = data["X"]
    y = data["y"].astype(np.int64)
    if {"train_idx", "val_idx", "test_idx"}.issubset(data.files):
        train_idx = data["train_idx"].astype(np.int64)
        val_idx = data["val_idx"].astype(np.int64)
        test_idx = data["test_idx"].astype(np.int64)
    else:
        train_idx, val_idx, test_idx = stratified_split(y, val_ratio, test_ratio, seed)
    return NpzData(X=X, y=y, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def make_datasets(data: NpzData, normalize: str = "trial") -> Tuple[EEGNpzDataset, EEGNpzDataset, EEGNpzDataset]:
    return (
        EEGNpzDataset(data.X[data.train_idx], data.y[data.train_idx], normalize=normalize),
        EEGNpzDataset(data.X[data.val_idx], data.y[data.val_idx], normalize=normalize),
        EEGNpzDataset(data.X[data.test_idx], data.y[data.test_idx], normalize=normalize),
    )


def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def cohen_kappa_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if len(labels) == 0:
        return 0.0
    label_to_i = {v: i for i, v in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        cm[label_to_i[t], label_to_i[p]] += 1
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.trace(cm) / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n)
    if abs(1.0 - pe) < 1e-12:
        return 0.0
    return float((po - pe) / (1.0 - pe))


def macro_precision_f1_np(y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None) -> Tuple[float, float]:
    if n_classes is None:
        n_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0))) + 1
    precisions, f1s = [], []
    for cls in range(n_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        precisions.append(precision)
        f1s.append(f1)
    return float(np.mean(precisions)), float(np.mean(f1s))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: Optional[int] = None) -> Dict[str, float]:
    precision, f1 = macro_precision_f1_np(y_true, y_pred, n_classes=n_classes)
    return {
        "acc": accuracy_score_np(y_true, y_pred),
        "kappa": cohen_kappa_np(y_true, y_pred),
        "precision_macro": precision,
        "f1_macro": f1,
    }
