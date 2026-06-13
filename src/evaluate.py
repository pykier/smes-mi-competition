"""Evaluation utilities."""

import time
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def classification_report_dict(y_true, y_pred) -> dict:
    """Return basic classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def measure_inference_time(predict_fn: Callable, x: np.ndarray, repeat: int = 10) -> float:
    """Measure average inference time in seconds."""
    if repeat <= 0:
        raise ValueError("repeat must be positive.")
    start = time.perf_counter()
    for _ in range(repeat):
        predict_fn(x)
    elapsed = time.perf_counter() - start
    return elapsed / repeat
