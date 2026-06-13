"""Dataset construction for official four-task EEG decoding."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .data_io import (
    extract_trial_infos_from_events,
    get_channel_labels,
    read_dat_as_channels_by_samples,
    read_meta_file,
    resolve_channel_indices,
    scan_recordings,
    split_eeg_and_trigger,
    trigger_to_events,
)
from .preprocess import preprocess_epoch


class EEGWindowDataset(Dataset):
    """Torch dataset for EEG epochs shaped as channels by time."""

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
    recordings = scan_recordings(
        raw_data_dir=config["paths"]["raw_data_dir"],
        file_glob=data_cfg.get("file_glob", "**/*.dat"),
        run_patterns=data_cfg["run_patterns"],
        meta_suffix=data_cfg.get("meta_suffix", "_meta.txt"),
    )
    rows = []
    for item in recordings:
        row = asdict(item)
        row["dat_path"] = str(item.dat_path)
        row["meta_path"] = str(item.meta_path) if item.meta_path else ""
        row["file_size_mb"] = Path(item.dat_path).stat().st_size / (1024 * 1024)
        rows.append(row)
    return pd.DataFrame(rows)


def inspect_one_recording(dat_path: str | Path, config: dict) -> dict:
    """Return shape and trigger diagnostics for one recording."""
    data_cfg = config["data"]
    meta_path = Path(dat_path).with_suffix("").with_name(Path(dat_path).stem + data_cfg.get("meta_suffix", "_meta.txt"))
    meta = read_meta_file(meta_path if meta_path.exists() else None)
    labels = get_channel_labels(meta)
    full = read_dat_as_channels_by_samples(
        dat_path,
        n_channels=int(data_cfg["total_channels"]),
        dtype=str(data_cfg.get("dat_dtype", "float32")),
        layout=str(data_cfg.get("dat_layout", "sample_major")),
    )
    eeg, trigger = split_eeg_and_trigger(
        full,
        eeg_channels=int(data_cfg["eeg_channels"]),
        trigger_position=str(data_cfg.get("trigger_position", "last")),
    )
    events = trigger_to_events(trigger)
    event_values = [value for _, value in events]
    return {
        "dat_path": str(dat_path),
        "meta_path": str(meta_path) if meta_path.exists() else "",
        "shape_full_channels_by_samples": list(full.shape),
        "shape_eeg_channels_by_samples": list(eeg.shape),
        "n_channel_labels": len(labels),
        "first_channel_labels": labels[:10],
        "last_channel_labels": labels[-5:],
        "n_events": len(events),
        "event_value_counts": {str(value): int(event_values.count(value)) for value in sorted(set(event_values))},
        "first_events": events[:20],
    }


def build_all_task_arrays(config: dict) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict]:
    """Build arrays for all official binary tasks.

    Returns
    -------
    task_data:
        Mapping from task name to ``(x, y, subjects)``.
        ``x`` has shape trials by selected_channels by resampled_time.
        ``y`` is binary: 0 means rest, 1 means target movement/imagery.
    metadata:
        Dataset-level metadata used for training reports and model export.
    """
    data_cfg = config["data"]
    prep_cfg = config["preprocess"]
    channel_cfg = config["channels"]
    train_cfg = config["training"]

    manifest = build_manifest(config)
    max_files = train_cfg.get("max_files")
    if max_files is not None:
        manifest = manifest.iloc[: int(max_files)].copy()

    task_buffers: dict[str, dict[str, list]] = {
        task_name: {"x": [], "y": [], "subjects": [], "trial_infos": []}
        for task_name in config["tasks"].keys()
    }
    selected_indices: list[int] | None = None
    selected_labels: list[str] = [item.upper() for item in channel_cfg["selected_labels"]]
    all_trial_rows = []

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Extracting trigger epochs"):
        meta = read_meta_file(row.meta_path if row.meta_path else None)
        channel_labels = get_channel_labels(meta)
        if selected_indices is None:
            selected_indices = resolve_channel_indices(channel_labels, selected_labels)

        full = read_dat_as_channels_by_samples(
            row.dat_path,
            n_channels=int(data_cfg["total_channels"]),
            dtype=str(data_cfg.get("dat_dtype", "float32")),
            layout=str(data_cfg.get("dat_layout", "sample_major")),
        )
        eeg, trigger = split_eeg_and_trigger(
            full,
            eeg_channels=int(data_cfg["eeg_channels"]),
            trigger_position=str(data_cfg.get("trigger_position", "last")),
        )
        events = trigger_to_events(trigger)
        trial_infos = extract_trial_infos_from_events(
            events=events,
            recording=row,
            event_values=data_cfg["event_values"],
            sampling_rate_hz=int(data_cfg["sampling_rate_hz"]),
            task_start_offset_seconds=float(data_cfg["task_start_offset_seconds"]),
            task_window_seconds=float(data_cfg["task_window_seconds"]),
            n_samples=int(eeg.shape[1]),
        )
        for trial_info in trial_infos:
            all_trial_rows.append(asdict(trial_info))
            for task_name, task_cfg in config["tasks"].items():
                if trial_info.run_type != task_cfg["run_type"]:
                    continue
                if trial_info.trial_label_name not in {task_cfg["positive_label"], task_cfg["negative_label"]}:
                    continue
                label = 1 if trial_info.trial_label_name == task_cfg["positive_label"] else 0
                epoch = eeg[selected_indices, trial_info.task_start_sample : trial_info.task_end_sample]
                epoch = preprocess_epoch(
                    epoch,
                    source_fs=int(data_cfg["sampling_rate_hz"]),
                    target_fs=int(prep_cfg.get("target_sampling_rate_hz") or data_cfg["sampling_rate_hz"]),
                    bandpass_hz=prep_cfg.get("bandpass_hz"),
                    standardize=bool(prep_cfg.get("standardize", True)),
                )
                task_buffers[task_name]["x"].append(epoch)
                task_buffers[task_name]["y"].append(label)
                task_buffers[task_name]["subjects"].append(trial_info.subject)
                task_buffers[task_name]["trial_infos"].append(asdict(trial_info))

    task_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    task_summaries = {}
    for task_name, buffer in task_buffers.items():
        if not buffer["x"]:
            raise RuntimeError(f"No trials were created for task {task_name}. Check trigger parsing and task config.")
        x = np.stack(buffer["x"]).astype(np.float32)
        y = np.asarray(buffer["y"], dtype=np.int64)
        subjects = np.asarray(buffer["subjects"])
        max_trials = train_cfg.get("max_trials_per_task")
        if max_trials is not None and len(y) > int(max_trials):
            x = x[: int(max_trials)]
            y = y[: int(max_trials)]
            subjects = subjects[: int(max_trials)]
        task_data[task_name] = (x, y, subjects)
        task_summaries[task_name] = {
            "n_trials": int(len(y)),
            "n_positive": int(np.sum(y == 1)),
            "n_rest": int(np.sum(y == 0)),
            "subjects": sorted(set(subjects.tolist())),
            "shape": list(x.shape),
        }

    metadata = {
        "manifest_rows": int(len(manifest)),
        "selected_channel_labels": selected_labels,
        "selected_channel_indices": selected_indices,
        "sampling_rate_hz": int(data_cfg["sampling_rate_hz"]),
        "target_sampling_rate_hz": int(prep_cfg.get("target_sampling_rate_hz") or data_cfg["sampling_rate_hz"]),
        "task_window_seconds": float(data_cfg["task_window_seconds"]),
        "n_times_after_preprocess": int(next(iter(task_data.values()))[0].shape[-1]),
        "task_summaries": task_summaries,
        "all_trial_count": len(all_trial_rows),
    }
    return task_data, metadata


def subject_split(x: np.ndarray, y: np.ndarray, subjects: np.ndarray, validation_subjects: list[str]):
    """Split by subject to avoid within-subject leakage."""
    validation_subjects = {item.lower() for item in validation_subjects}
    val_mask = np.asarray([str(subject).lower() in validation_subjects for subject in subjects])
    if val_mask.sum() == 0 or (~val_mask).sum() == 0:
        return None
    return x[~val_mask], y[~val_mask], x[val_mask], y[val_mask]


def random_split(x: np.ndarray, y: np.ndarray, validation_ratio: float, seed: int):
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


def split_train_val(x: np.ndarray, y: np.ndarray, subjects: np.ndarray, config: dict):
    """Split training and validation sets according to config."""
    train_cfg = config["training"]
    if train_cfg.get("validation_mode") == "leave_subjects_out":
        split = subject_split(x, y, subjects, train_cfg.get("validation_subjects", []))
        if split is not None:
            return split
    return random_split(
        x,
        y,
        validation_ratio=float(train_cfg.get("validation_ratio", 0.2)),
        seed=int(config["project"].get("random_seed", 42)),
    )
