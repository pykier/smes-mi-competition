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
    padlen = min(3 * max(len(a), len(b)), max(1, x.shape[-1] - 1))
    return filtfilt(b, a, x, axis=-1, padlen=padlen).astype(np.float32, copy=False)


def resample_eeg(x: np.ndarray, source_fs: int, target_fs: int | None) -> np.ndarray:
    """Resample EEG data along the time axis."""
    if target_fs is None or source_fs == target_fs:
        return x.astype(np.float32, copy=False)
    if source_fs <= 0 or target_fs <= 0:
        raise ValueError("Sampling rates must be positive.")
    return resample_poly(x, up=int(target_fs), down=int(source_fs), axis=-1).astype(np.float32, copy=False)


def standardize_trials(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score each trial and channel independently."""
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)


def preprocess_epoch(
    epoch: np.ndarray,
    source_fs: int,
    target_fs: int | None,
    bandpass_hz: list[float] | tuple[float, float] | None,
    standardize: bool = True,
) -> np.ndarray:
    """Preprocess one task epoch shaped as channels by samples."""
    x = epoch.astype(np.float32, copy=False)
    if bandpass_hz is not None:
        x = bandpass_filter(x, fs=source_fs, low=float(bandpass_hz[0]), high=float(bandpass_hz[1]))
    x = resample_eeg(x, source_fs=source_fs, target_fs=target_fs)
    if standardize:
        x = standardize_trials(x[None, ...])[0]
    return x.astype(np.float32, copy=False)


def preprocess_batch(
    x: np.ndarray,
    source_fs: int,
    target_fs: int | None,
    bandpass_hz: list[float] | tuple[float, float] | None,
    standardize: bool = True,
) -> np.ndarray:
    """Preprocess a batch of epochs shaped as trials by channels by samples."""
    processed = [
        preprocess_epoch(epoch, source_fs=source_fs, target_fs=target_fs, bandpass_hz=bandpass_hz, standardize=False)
        for epoch in x
    ]
    out = np.stack(processed).astype(np.float32)
    if standardize:
        out = standardize_trials(out)
    return out
