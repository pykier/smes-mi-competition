"""Competition submission interface for SMES-MI four-task EEGNet models.

This file is designed to be copied into the official competition framework and
used together with the exported ``model_artifacts`` directory.

The official framework may pass task information using different keyword names.
The implementation therefore accepts flexible ``*args`` and ``**kwargs`` in
``calibrate`` and ``predict`` while keeping the required behavior:

- requested EEG channels <= 8;
- no calibration trials requested by default;
- input trial is channels by samples;
- output label is 0 for rest and 1 for target action/imagery.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.eegnet import EEGNet
from src.preprocess import preprocess_epoch


class AlgorithmImplement:
    """Official-style algorithm implementation.

    Four binary task models are supported:

    - vme_left_vs_rest
    - vme_right_vs_rest
    - vmi_left_vs_rest
    - vmi_right_vs_rest
    """

    def __init__(self, task_name: str | None = None, model_artifacts_dir: str | None = None, device: str = "auto", **kwargs):
        self.model_artifacts_dir = Path(model_artifacts_dir or ROOT_DIR / "model_artifacts")
        self.device = self._resolve_device(device)
        self.artifact = self._load_artifact_config()
        self.config = self.artifact["config"]
        self.selected_channel_labels = list(self.artifact["metadata"]["selected_channel_labels"])
        self.selected_channel_indices = list(self.artifact["metadata"].get("selected_channel_indices", []))
        self.default_task_name = task_name or os.environ.get("SMES_TASK_NAME") or self.artifact["task_order"][0]
        self.models = self._load_models()

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_artifact_config(self) -> dict:
        path = self.model_artifacts_dir / "artifact_config.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing model artifact config: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _build_model(self, checkpoint: dict) -> EEGNet:
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
        model.to(self.device)
        model.eval()
        return model

    def _load_models(self) -> dict[str, EEGNet]:
        models = {}
        for task_name, filename in self.artifact["model_files"].items():
            checkpoint = torch.load(self.model_artifacts_dir / filename, map_location=self.device)
            models[task_name] = self._build_model(checkpoint)
        return models

    def get_required_channel_labels(self, *args, **kwargs):
        """Return EEG channel labels requested by the algorithm.

        The list length is 8 by default, satisfying the competition channel limit.
        """
        return self.selected_channel_labels

    def get_calibration_trial_count(self, *args, **kwargs) -> int:
        """Request zero calibration trials by default for maximum calibration score."""
        return 0

    def calibrate(self, *args, **kwargs):
        """Calibration hook required by the competition framework.

        The current baseline does not use online calibration. The method is kept
        intentionally to satisfy the official lifecycle.
        """
        return None

    def _normalize_task_name(self, task_name: str | None = None, **kwargs) -> str:
        if task_name in self.models:
            return task_name
        for key in ("task_name", "task", "model_name"):
            value = kwargs.get(key)
            if value in self.models:
                return value

        condition = str(kwargs.get("condition", kwargs.get("run_type", ""))).lower()
        side = str(kwargs.get("side", kwargs.get("target", ""))).lower()
        if condition in {"vme", "me", "movement", "execution"}:
            prefix = "vme"
        elif condition in {"vmi", "mi", "imagery"}:
            prefix = "vmi"
        else:
            return self.default_task_name

        if side in {"left", "l", "1"}:
            return f"{prefix}_left_vs_rest"
        if side in {"right", "r", "2"}:
            return f"{prefix}_right_vs_rest"
        return self.default_task_name

    def _prepare_input(self, eeg_data) -> np.ndarray:
        x = np.asarray(eeg_data, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Expected a 2D EEG trial array, got shape {x.shape}")

        expected_channels = len(self.selected_channel_labels)
        if x.shape[0] == expected_channels:
            selected = x
        elif x.shape[1] == expected_channels:
            selected = x.T
        elif x.shape[0] >= max(self.selected_channel_indices, default=0) + 1:
            selected = x[np.asarray(self.selected_channel_indices, dtype=int), :]
        elif x.shape[1] >= max(self.selected_channel_indices, default=0) + 1:
            selected = x.T[np.asarray(self.selected_channel_indices, dtype=int), :]
        else:
            raise ValueError(
                f"Input shape {x.shape} is incompatible with selected channels {self.selected_channel_labels}"
            )

        data_cfg = self.config["data"]
        prep_cfg = self.config["preprocess"]
        epoch = preprocess_epoch(
            selected,
            source_fs=int(data_cfg["sampling_rate_hz"]),
            target_fs=int(prep_cfg.get("target_sampling_rate_hz") or data_cfg["sampling_rate_hz"]),
            bandpass_hz=prep_cfg.get("bandpass_hz"),
            standardize=bool(prep_cfg.get("standardize", True)),
        )
        return epoch

    def predict(self, eeg_data, task_name: str | None = None, *args, **kwargs) -> int:
        """Predict one trial.

        Parameters
        ----------
        eeg_data:
            EEG trial data. Official expected shape is channels by samples.
        task_name:
            Optional task selector. If not provided, the default task is used.

        Returns
        -------
        int
            0 means rest; 1 means target movement or imagery.
        """
        resolved_task = self._normalize_task_name(task_name, **kwargs)
        epoch = self._prepare_input(eeg_data)
        model = self.models[resolved_task]
        with torch.no_grad():
            tensor = torch.tensor(epoch[None, ...], dtype=torch.float32, device=self.device)
            logits = model(tensor)
            pred = int(torch.argmax(logits, dim=1).detach().cpu().item())
        return pred

    def run(self, eeg_data, *args, **kwargs) -> int:
        """Alias for frameworks that call run instead of predict."""
        return self.predict(eeg_data, *args, **kwargs)
