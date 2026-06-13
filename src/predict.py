"""Inference utilities for exported four-task EEGNet models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .eegnet import EEGNet
from .preprocess import preprocess_epoch


def resolve_inference_device(device: str) -> torch.device:
    """Resolve inference device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_artifact_config(model_artifacts_dir: str | Path) -> dict:
    """Load exported artifact configuration."""
    path = Path(model_artifacts_dir) / "artifact_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_eegnet_from_checkpoint(checkpoint: dict, device: torch.device) -> EEGNet:
    """Rebuild EEGNet from checkpoint metadata."""
    model_cfg = checkpoint["model_config"]
    model = EEGNet(
        n_channels=int(checkpoint["n_channels"]),
        n_times=int(checkpoint["n_times"]),
        n_classes=int(checkpoint.get("n_classes", 2)),
        F1=int(model_cfg.get("F1", 8)),
        D=int(model_cfg.get("D", 2)),
        F2=int(model_cfg.get("F2", 16)),
        kernel_length=int(model_cfg.get("kernel_length", 64)),
        dropout=float(model_cfg.get("dropout", 0.5)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_task_models(model_artifacts_dir: str | Path, device: str = "auto"):
    """Load all exported task models."""
    resolved_device = resolve_inference_device(device)
    artifact = load_artifact_config(model_artifacts_dir)
    models = {}
    for task_name, file_name in artifact["model_files"].items():
        ckpt = torch.load(Path(model_artifacts_dir) / file_name, map_location=resolved_device)
        models[task_name] = build_eegnet_from_checkpoint(ckpt, resolved_device)
    return models, artifact, resolved_device


def predict_epoch(model: EEGNet, epoch: np.ndarray, device: torch.device) -> int:
    """Predict one preprocessed epoch, returning 0 or 1."""
    with torch.no_grad():
        x = torch.tensor(epoch[None, ...], dtype=torch.float32, device=device)
        logits = model(x)
        return int(torch.argmax(logits, dim=1).detach().cpu().item())


def preprocess_competition_trial(eeg_trial: np.ndarray, artifact: dict) -> np.ndarray:
    """Preprocess one competition trial shaped as selected_channels by samples.

    In the official online stage the framework is expected to pass only the selected
    channel data and only the task window. Therefore this function does not parse
    trigger events; it applies the same filter, resampling and standardization used
    during offline training.
    """
    config = artifact["config"]
    data_cfg = config["data"]
    prep_cfg = config["preprocess"]
    return preprocess_epoch(
        eeg_trial.astype(np.float32, copy=False),
        source_fs=int(data_cfg["sampling_rate_hz"]),
        target_fs=int(prep_cfg.get("target_sampling_rate_hz") or data_cfg["sampling_rate_hz"]),
        bandpass_hz=prep_cfg.get("bandpass_hz"),
        standardize=bool(prep_cfg.get("standardize", True)),
    )


def predict_competition_trial(
    eeg_trial: np.ndarray,
    task_name: str,
    model_artifacts_dir: str | Path = "model_artifacts",
    device: str = "auto",
) -> int:
    """Convenience function for predicting one official trial."""
    models, artifact, resolved_device = load_task_models(model_artifacts_dir, device=device)
    if task_name not in models:
        raise KeyError(f"Unknown task_name={task_name!r}. Available tasks: {list(models)}")
    epoch = preprocess_competition_trial(eeg_trial, artifact)
    return predict_epoch(models[task_name], epoch, resolved_device)
