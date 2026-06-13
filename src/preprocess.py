"""EEG preprocessing interfaces."""

import numpy as np


def select_time_window(x: np.ndarray, start_sample: int, end_sample: int) -> np.ndarray:
    """Select a time window from trial data.

    Expected shape is (..., time). The function does not assume a fixed channel axis.
    """
    if start_sample < 0 or end_sample <= start_sample:
        raise ValueError("Invalid time window.")
    return x[..., start_sample:end_sample]


def preprocess_trials(x: np.ndarray, **kwargs) -> np.ndarray:
    """Preprocess EEG trials.

    Placeholder for filtering, resampling, channel selection, normalization, and artifact checks.
    """
    return x
