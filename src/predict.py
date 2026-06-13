"""Inference utilities for the EEGNet baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .data_io import read_dat_as_channels_by_samples
from .eegnet import EEGNet
from .preprocess import make_fixed_windows, preprocess_continuous_recording, standardize_trials


def resolve_inference_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_eegnet_checkpoint(model_path: str | Path, device: str = "auto"):
    """Load a saved EEGNet checkpoint and rebuild the model."""
    resolved_device = resolve_inference_device(device)
    checkpoint = torch.load(model_path, map_location=resolved_device)
    config = checkpoint["config"]
    model_cfg = config["model"]
    model = EEGNet(
        n_channels=int(checkpoint["n_channels"]),
        n_times=int(checkpoint["n_times"]),
        n_classes=int(checkpoint["n_classes"]),
        F1=int(model_cfg.get("F1", 8)),
        D=int(model_cfg.get("D", 2)),
        F2=int(model_cfg.get("F2", 16)),
        kernel_length=int(model_cfg.get("kernel_length", 64)),
        dropout=float(model_cfg.get("dropout", 0.5)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()
    return model, checkpoint, resolved_device


def predict_windows(model: EEGNet, x: np.ndarray, device: torch.device) -> np.ndarray:
    """Predict labels for windows shaped as trials by channels by time."""
    with torch.no_grad():
        tensor = torch.tensor(x, dtype=torch.float32, device=device)
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
    return pred


def predict_dat_file(model_path: str | Path, dat_path: str | Path, device: str = "auto") -> np.ndarray:
    """Run inference on one local DAT file using the saved configuration."""
    model, checkpoint, resolved_device = load_eegnet_checkpoint(model_path, device=device)
    config = checkpoint["config"]
    data_cfg = config["data"]
    prep_cfg = config["preprocess"]

    continuous = read_dat_as_channels_by_samples(
        dat_path,
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
    x_win, _ = make_fixed_windows(
        processed,
        label=0,
        fs=fs,
        window_seconds=float(prep_cfg["trial_window_seconds"]),
        stride_seconds=float(prep_cfg["stride_seconds"]),
        max_windows=None,
    )
    if len(x_win) == 0:
        raise RuntimeError("No windows were created for inference.")
    if bool(prep_cfg.get("standardize", True)):
        x_win = standardize_trials(x_win)
    return predict_windows(model, x_win, resolved_device)
