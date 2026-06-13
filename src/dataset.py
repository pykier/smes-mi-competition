"""Dataset construction for EEGNet experiments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .data_io import read_dat_as_channels_by_samples, scan_recordings
from .preprocess import make_fixed_windows, preprocess_continuous_recording, standardize_trials


class EEGWindowDataset(Dataset):
    """Torch dataset for EEG windows shaped as channels by time."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        if x.ndim != 3:
            raise ValueError(f"Expected x with shape trials by channels by time, got {x.shape}")
        if len(x) != len(y):
            raise ValueError("x and y have inconsistent lengths.")
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.y.numel())

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]


def build_manifest(config: dict) -> pd.DataFrame:
    """Build a manifest table for all usable DAT files."""
    data_cfg = config["data"]
    recordings, label_to_id = scan_recordings(
        raw_data_dir=config["paths"]["raw_data_dir"],
        file_glob=data_cfg.get("file_glob", "**/*.dat"),
        label_patterns=data_cfg["label_patterns"],
        meta_suffix=data_cfg.get("meta_suffix", "_meta"),
    )
    rows = []
    for item in recordings:
        row = asdict(item)
        row["dat_path"] = str(item.dat_path)
        row["meta_path"] = str(item.meta_path) if item.meta_path else ""
        row["file_size_mb"] = Path(item.dat_path).stat().st_size / (1024 * 1024)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.attrs["label_to_id"] = label_to_id
    return df


def build_windows(config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load local DAT files and build EEG windows for training.

    This function is intentionally capped by data.max_files and
    data.max_windows_per_file by default, so the first run is fast and low-risk.
    Increase those values only after the pipeline is confirmed.
    """
    data_cfg = config["data"]
    prep_cfg = config["preprocess"]
    manifest = build_manifest(config)

    max_files = data_cfg.get("max_files")
    if max_files is not None:
        manifest = manifest.iloc[: int(max_files)].copy()

    all_x = []
    all_y = []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Loading DAT files"):
        continuous = read_dat_as_channels_by_samples(
            row.dat_path,
            n_channels=int(data_cfg["n_channels"]),
            dtype=str(data_cfg.get("dat_dtype", "float32")),
            layout=str(data_cfg.get("dat_layout", "sample_major")),
        )
        processed, fs = preprocess_continuous_recording(
            continuous,
            source_fs=int(data_cfg["sampling_rate_hz"]),
            target_fs=int(prep_cfg.get("target_sampling_rate_hz") or data_cfg["sampling_rate_hz"]),
            selected_channel_indices=list(data_cfg.get("selected_channel_indices", [])),
            bandpass_hz=prep_cfg.get("bandpass_hz"),
        )
        x_win, y_win = make_fixed_windows(
            processed,
            label=int(row.label_id),
            fs=fs,
            window_seconds=float(prep_cfg["trial_window_seconds"]),
            stride_seconds=float(prep_cfg["stride_seconds"]),
            max_windows=data_cfg.get("max_windows_per_file"),
        )
        if len(y_win) == 0:
            continue
        if bool(prep_cfg.get("standardize", True)):
            x_win = standardize_trials(x_win)
        all_x.append(x_win)
        all_y.append(y_win)

    if not all_x:
        raise RuntimeError("No windows were created. Check data format, sampling rate, and window length.")

    x = np.concatenate(all_x, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)
    metadata = {
        "n_trials": int(x.shape[0]),
        "n_channels": int(x.shape[1]),
        "n_times": int(x.shape[2]),
        "label_to_id": manifest.attrs.get("label_to_id", {}),
        "manifest_rows": int(len(manifest)),
    }
    return x, y, metadata


def train_val_split(x: np.ndarray, y: np.ndarray, validation_ratio: float, seed: int):
    """Create a reproducible random train/validation split."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(indices) * validation_ratio))) if len(indices) > 1 else 0
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]
