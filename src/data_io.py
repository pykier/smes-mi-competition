"""Data loading and trigger parsing for the SMES-MI competition.

The provided local files use this format according to ``*_meta.txt``:

- binary float32 little-endian DAT files
- layout: timepoints by channels
- value order: sample major, trigger last
- channels: 69 total = 68 EEG channels + 1 TRIGGER channel
- sampling rate: 1000 Hz

This module keeps the raw-data assumptions explicit and validates them against the
meta file whenever possible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RecordingInfo:
    """Metadata for one EEG recording file."""

    dat_path: Path
    meta_path: Path | None
    subject: str
    session: str
    run_type: str


@dataclass(frozen=True)
class TrialInfo:
    """One extracted trial window."""

    dat_path: Path
    subject: str
    session: str
    run_type: str
    trial_label: int
    trial_label_name: str
    trial_start_sample: int
    task_start_sample: int
    task_end_sample: int


def read_meta_file(meta_path: str | Path | None) -> dict[str, Any]:
    """Read a small text meta file and parse key-value fields."""
    if meta_path is None:
        return {"raw_text": "", "fields": {}}

    path = Path(meta_path)
    if not path.exists():
        return {"raw_text": "", "fields": {}}

    text = path.read_text(encoding="utf-8", errors="ignore")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
        elif "=" in line:
            key, value = line.split("=", 1)
        else:
            continue
        fields[key.strip().lower().replace(" ", "_")] = value.strip()
    return {"raw_text": text, "fields": fields}


def get_channel_labels(meta: dict[str, Any]) -> list[str]:
    """Return channel labels parsed from meta information."""
    text = meta.get("raw_text", "")
    match = re.search(r"channel\s+labels\s*=\s*(.+)", text, flags=re.IGNORECASE)
    if not match:
        value = meta.get("fields", {}).get("channel_labels", "")
    else:
        value = match.group(1)
    if not value:
        return []
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def _infer_subject_session(path: Path) -> tuple[str, str]:
    subject = "unknown_subject"
    session = "unknown_session"
    for part in path.parts:
        part_lower = part.lower()
        if re.fullmatch(r"sub_?\d+", part_lower):
            subject = part_lower.replace("sub", "sub_").replace("__", "_")
        if re.fullmatch(r"session_?\d+", part_lower):
            session = part_lower.replace("session", "session").replace("__", "_")
    return subject, session


def infer_run_type(path: str | Path, run_patterns: dict[str, list[str]]) -> str | None:
    """Infer run type, e.g. vme or vmi, from file name."""
    lower_name = Path(path).name.lower()
    for run_type, patterns in run_patterns.items():
        for pattern in patterns:
            if pattern.lower() in lower_name:
                return run_type
    return None


def find_meta_for_dat(dat_path: str | Path, meta_suffix: str = "_meta.txt") -> Path | None:
    """Find the paired meta file for a DAT file."""
    path = Path(dat_path)
    candidates = [
        path.with_suffix("").with_name(path.stem + meta_suffix),
        path.with_suffix("").with_name(path.stem + "_meta.txt"),
        path.with_suffix("").with_name(path.stem + "_meta"),
        path.with_name(path.name + meta_suffix),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def scan_recordings(
    raw_data_dir: str | Path,
    file_glob: str,
    run_patterns: dict[str, list[str]],
    meta_suffix: str = "_meta.txt",
) -> list[RecordingInfo]:
    """Scan local raw data files and return recording metadata."""
    root = Path(raw_data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {root}. Update paths.raw_data_dir in configs/default.yaml."
        )

    dat_files = sorted(root.glob(file_glob))
    if not dat_files:
        raise FileNotFoundError(f"No DAT files matched {file_glob!r} under {root}")

    recordings: list[RecordingInfo] = []
    for dat_path in dat_files:
        run_type = infer_run_type(dat_path, run_patterns)
        if run_type is None:
            continue
        subject, session = _infer_subject_session(dat_path)
        meta_path = find_meta_for_dat(dat_path, meta_suffix=meta_suffix)
        recordings.append(
            RecordingInfo(
                dat_path=dat_path,
                meta_path=meta_path,
                subject=subject,
                session=session,
                run_type=run_type,
            )
        )

    if not recordings:
        raise ValueError("DAT files were found, but no VME/VMI run type could be inferred from filenames.")
    return recordings


def read_dat_as_channels_by_samples(
    dat_path: str | Path,
    n_channels: int,
    dtype: str = "float32",
    layout: str = "sample_major",
) -> np.ndarray:
    """Read a binary DAT file as an array with shape channels by samples."""
    path = Path(dat_path)
    raw = np.fromfile(path, dtype=np.dtype(dtype))
    if raw.size == 0:
        raise ValueError(f"Empty DAT file: {path}")
    if raw.size % n_channels != 0:
        raise ValueError(
            f"File {path} has {raw.size} values, which cannot be exactly reshaped by n_channels={n_channels}. "
            "Check total_channels and dat_dtype in the config."
        )

    if layout == "sample_major":
        data = raw.reshape(-1, n_channels).T
    elif layout == "channel_major":
        data = raw.reshape(n_channels, -1)
    else:
        raise ValueError(f"Unsupported DAT layout: {layout}")
    return data.astype(np.float32, copy=False)


def split_eeg_and_trigger(data: np.ndarray, eeg_channels: int, trigger_position: str = "last") -> tuple[np.ndarray, np.ndarray]:
    """Split full recording into EEG and trigger arrays."""
    if trigger_position != "last":
        raise NotImplementedError("Only trigger_position='last' is currently supported.")
    if data.shape[0] < eeg_channels + 1:
        raise ValueError(f"Expected at least {eeg_channels + 1} channels, got {data.shape[0]}")
    eeg = data[:eeg_channels]
    trigger = data[-1]
    return eeg, trigger


def trigger_to_events(trigger: np.ndarray, min_abs_value: float = 0.5) -> list[tuple[int, int]]:
    """Convert a trigger channel into transition events.

    Repeated trigger samples are collapsed. Only non-zero values are returned.
    Values are rounded to integers because trigger is stored in float32.
    """
    trig = np.rint(trigger).astype(np.int64)
    nonzero = np.abs(trig) >= min_abs_value
    prev = np.concatenate([[0], trig[:-1]])
    rising_or_change = nonzero & (trig != prev)
    indices = np.flatnonzero(rising_or_change)
    return [(int(idx), int(trig[idx])) for idx in indices if int(trig[idx]) != 0]


def label_name_from_value(value: int, event_values: dict[str, int]) -> str | None:
    """Map numeric trial trigger to a semantic label."""
    for name in ("left", "right", "rest"):
        if int(event_values[name]) == int(value):
            return name
    return None


def extract_trial_infos_from_events(
    events: list[tuple[int, int]],
    recording: RecordingInfo,
    event_values: dict[str, int],
    sampling_rate_hz: int,
    task_start_offset_seconds: float,
    task_window_seconds: float,
    n_samples: int,
) -> list[TrialInfo]:
    """Build trial metadata from trigger events.

    The robust rule used here is:
    1. each trial has one class trigger 1/2/3;
    2. if a nearby 101 trial-start trigger exists, it is used as trial start;
    3. otherwise the class trigger itself is treated as trial start;
    4. the classification window is trial_start + 7 s to trial_start + 11 s.
    """
    class_values = {int(event_values["left"]), int(event_values["right"]), int(event_values["rest"])}
    trial_start_code = int(event_values.get("trial_start", 101))
    task_offset = int(round(task_start_offset_seconds * sampling_rate_hz))
    window_samples = int(round(task_window_seconds * sampling_rate_hz))

    trial_start_indices = [idx for idx, val in events if val == trial_start_code]
    infos: list[TrialInfo] = []
    for event_pos, (label_idx, label_value) in enumerate(events):
        if label_value not in class_values:
            continue
        label_name = label_name_from_value(label_value, event_values)
        if label_name is None:
            continue

        candidate_starts = [idx for idx in trial_start_indices if abs(idx - label_idx) <= sampling_rate_hz]
        if candidate_starts:
            trial_start = min(candidate_starts, key=lambda item: abs(item - label_idx))
        else:
            previous_starts = [idx for idx in trial_start_indices if idx <= label_idx]
            next_event_idx = events[event_pos + 1][0] if event_pos + 1 < len(events) else n_samples
            if previous_starts and label_idx - previous_starts[-1] <= 2 * sampling_rate_hz:
                trial_start = previous_starts[-1]
            elif label_idx + task_offset + window_samples <= n_samples:
                trial_start = label_idx
            elif label_idx >= window_samples:
                trial_start = label_idx - task_offset
            else:
                continue
            if trial_start >= next_event_idx:
                trial_start = label_idx

        task_start = trial_start + task_offset
        task_end = task_start + window_samples
        if task_start < 0 or task_end > n_samples:
            continue
        infos.append(
            TrialInfo(
                dat_path=recording.dat_path,
                subject=recording.subject,
                session=recording.session,
                run_type=recording.run_type,
                trial_label=int(label_value),
                trial_label_name=label_name,
                trial_start_sample=int(trial_start),
                task_start_sample=int(task_start),
                task_end_sample=int(task_end),
            )
        )
    return infos


def resolve_channel_indices(channel_labels: list[str], selected_labels: list[str]) -> list[int]:
    """Resolve requested EEG channel names to zero-based indices."""
    upper_labels = [item.upper() for item in channel_labels]
    resolved = []
    missing = []
    for label in selected_labels:
        target = label.upper()
        if target in upper_labels:
            resolved.append(upper_labels.index(target))
        else:
            missing.append(label)
    if missing:
        raise ValueError(f"Requested channels not found in meta labels: {missing}")
    return resolved
