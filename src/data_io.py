"""Data loading utilities for local EEG competition files.

The current implementation targets the directory style shown by the user:

    data/raw/feel_MI_2026/sub_1/session1/*.dat
    data/raw/feel_MI_2026/sub_1/session1/*_meta

The loader is intentionally conservative. It first scans files, infers labels from
file names, then reads binary DAT files into arrays shaped as channels by samples.
If the official event format is later confirmed, only this module should need
substantial changes.
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
    label_name: str
    label_id: int


def read_meta_file(meta_path: str | Path | None) -> dict[str, Any]:
    """Read a small text meta file.

    The parser keeps both raw text and simple key-value pairs. It is designed to
    avoid failing when the exact competition meta format is unknown.
    """
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
        fields[key.strip().lower()] = value.strip()
    return {"raw_text": text, "fields": fields}


def _infer_subject_session(path: Path) -> tuple[str, str]:
    parts = path.parts
    subject = "unknown_subject"
    session = "unknown_session"
    for part in parts:
        if re.fullmatch(r"sub_?\d+", part.lower()):
            subject = part
        if re.fullmatch(r"session_?\d+", part.lower()):
            session = part
    return subject, session


def infer_label_from_filename(path: str | Path, label_patterns: dict[str, list[str]]) -> str | None:
    """Infer label name from a file path using configured substring patterns."""
    lower_name = Path(path).name.lower()
    for label_name, patterns in label_patterns.items():
        for pattern in patterns:
            if pattern.lower() in lower_name:
                return label_name
    return None


def find_meta_for_dat(dat_path: str | Path, meta_suffix: str = "_meta") -> Path | None:
    """Find the paired meta file for a DAT file."""
    path = Path(dat_path)
    candidates = [
        path.with_suffix("").with_name(path.stem + meta_suffix),
        path.with_name(path.name + meta_suffix),
        path.with_suffix(path.suffix + meta_suffix),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def scan_recordings(
    raw_data_dir: str | Path,
    file_glob: str,
    label_patterns: dict[str, list[str]],
    meta_suffix: str = "_meta",
) -> tuple[list[RecordingInfo], dict[str, int]]:
    """Scan local raw data files and return recording metadata."""
    root = Path(raw_data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {root}. Put data under data/raw/feel_MI_2026 or update configs/default.yaml."
        )

    dat_files = sorted(root.rglob(file_glob.replace("**/", ""))) if "**" not in file_glob else sorted(root.glob(file_glob))
    if not dat_files:
        raise FileNotFoundError(f"No DAT files matched {file_glob!r} under {root}")

    label_names = sorted(label_patterns.keys())
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    recordings: list[RecordingInfo] = []

    for dat_path in dat_files:
        label_name = infer_label_from_filename(dat_path, label_patterns)
        if label_name is None:
            continue
        subject, session = _infer_subject_session(dat_path)
        meta_path = find_meta_for_dat(dat_path, meta_suffix=meta_suffix)
        recordings.append(
            RecordingInfo(
                dat_path=dat_path,
                meta_path=meta_path,
                subject=subject,
                session=session,
                label_name=label_name,
                label_id=label_to_id[label_name],
            )
        )

    if not recordings:
        raise ValueError(
            "DAT files were found, but no labels could be inferred from filenames. "
            "Edit data.label_patterns in configs/default.yaml."
        )

    used_labels = sorted({item.label_name for item in recordings})
    compact_label_to_id = {name: idx for idx, name in enumerate(used_labels)}
    recordings = [
        RecordingInfo(
            dat_path=item.dat_path,
            meta_path=item.meta_path,
            subject=item.subject,
            session=item.session,
            label_name=item.label_name,
            label_id=compact_label_to_id[item.label_name],
        )
        for item in recordings
    ]
    return recordings, compact_label_to_id


def read_dat_as_channels_by_samples(
    dat_path: str | Path,
    n_channels: int,
    dtype: str = "float32",
    layout: str = "sample_major",
) -> np.ndarray:
    """Read a binary DAT file as an array with shape channels by samples.

    Parameters
    ----------
    dat_path:
        Local DAT file path.
    n_channels:
        Number of channels to reshape the binary stream.
    dtype:
        Numpy dtype used by the raw file. If this is wrong, use scripts/inspect_data.py
        to check file size and update configs/default.yaml.
    layout:
        sample_major means data are stored as samples by channels and will be transposed.
        channel_major means data are stored as channels by samples.
    """
    path = Path(dat_path)
    raw = np.fromfile(path, dtype=np.dtype(dtype))
    if raw.size == 0:
        raise ValueError(f"Empty DAT file: {path}")
    usable = raw.size - (raw.size % n_channels)
    if usable <= 0:
        raise ValueError(f"File {path} cannot be reshaped with n_channels={n_channels}")
    if usable != raw.size:
        raw = raw[:usable]

    if layout == "sample_major":
        data = raw.reshape(-1, n_channels).T
    elif layout == "channel_major":
        data = raw.reshape(n_channels, -1)
    else:
        raise ValueError(f"Unsupported DAT layout: {layout}")

    return data.astype(np.float32, copy=False)
