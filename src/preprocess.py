"""EEG preprocessing utilities."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly


def select_channels(x: np.ndarray, channel_indices: list[int]) -> np.ndarray:
    """Select EEG channels from data shaped as channels by samples."""
    if not channel_indices:
        return x
    return x[np.asarray(channel_indices, dtype=int), :]


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter."""
    if low <= 0 or high <= low or high >= fs / 2:
        raise ValueError(f"Invalid bandpass range: low={low}, high={high}, fs={fs}")
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="bandpass")
    return filtfilt(b, a, x, axis=-1).astype(np.float32, copy=False)


def resample_eeg(x: np.ndarray, source_fs: int, target_fs: int) -> np.ndarray:
    """Resample EEG data along the time axis."""
    if source_fs == target_fs or target_fs is None:
        return x
    if source_fs <= 0 or target_fs <= 0:
        raise ValueError("Sampling rates must be positive.")
    return resample_poly(x, up=target_fs, down=source_fs, axis=-1).astype(np.float32, copy=False)


def make_fixed_windows(
    x: np.ndarray,
    label: int,
    fs: int,
    window_seconds: float,
    stride_seconds: float,
    max_windows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split continuous EEG into fixed-length windows.

    Parameters
    ----------
    x:
        Array shaped as channels by samples.
    label:
        Integer label assigned to all windows from this recording. This is a
        framework-level fallback. Official event labels should replace this logic
        after the meta/event format is confirmed.
    fs:
        Sampling rate in Hz.
    window_seconds:
        Window length in seconds.
    stride_seconds:
        Window stride in seconds.
    max_windows:
        Optional cap used for fast smoke tests.
    """
    window_samples = int(round(window_seconds * fs))
    stride_samples = int(round(stride_seconds * fs))
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("Window and stride must be positive.")
    if x.shape[-1] < window_samples:
        return np.empty((0, x.shape[0], window_samples), dtype=np.float32), np.empty((0,), dtype=np.int64)

    windows = []
    labels = []
    for start in range(0, x.shape[-1] - window_samples + 1, stride_samples):
        windows.append(x[:, start : start + window_samples])
        labels.append(label)
        if max_windows is not None and len(windows) >= max_windows:
            break

    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.int64)


def standardize_trials(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score each trial and channel independently."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)


def preprocess_continuous_recording(
    x: np.ndarray,
    source_fs: int,
    target_fs: int,
    selected_channel_indices: list[int],
    bandpass_hz: list[float] | tuple[float, float] | None,
) -> tuple[np.ndarray, int]:
    """Preprocess a continuous recording and return data plus effective sampling rate."""
    x = select_channels(x, selected_channel_indices)
    if bandpass_hz is not None:
        x = bandpass_filter(x, fs=source_fs, low=float(bandpass_hz[0]), high=float(bandpass_hz[1]))
    if target_fs is not None and target_fs != source_fs:
        x = resample_eeg(x, source_fs=source_fs, target_fs=target_fs)
        effective_fs = target_fs
    else:
        effective_fs = source_fs
    return x.astype(np.float32, copy=False), int(effective_fs)
