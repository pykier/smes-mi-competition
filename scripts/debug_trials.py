"""Debug trigger parsing and task trial extraction.

Run:
    python scripts/debug_trials.py --config configs/default.yaml --file D:/path/to/file.dat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_io import (
    RecordingInfo,
    extract_trial_infos_from_events,
    find_meta_for_dat,
    read_dat_as_channels_by_samples,
    split_eeg_and_trigger,
    trigger_to_events,
)


def infer_subject_session_from_path(path: Path):
    subject = next((part for part in path.parts if part.lower().startswith("sub_")), "unknown_subject")
    session = next((part for part in path.parts if part.lower().startswith("session")), "unknown_session")
    return subject, session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--file", required=True, help="Path to one DAT file.")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    dat_path = Path(args.file)
    subject, session = infer_subject_session_from_path(dat_path)
    run_type = "vme" if "_vme_" in dat_path.name.lower() else "vmi" if "_vmi_" in dat_path.name.lower() else "unknown"
    recording = RecordingInfo(
        dat_path=dat_path,
        meta_path=find_meta_for_dat(dat_path, data_cfg.get("meta_suffix", "_meta.txt")),
        subject=subject,
        session=session,
        run_type=run_type,
    )
    full = read_dat_as_channels_by_samples(
        dat_path,
        n_channels=int(data_cfg["total_channels"]),
        dtype=str(data_cfg.get("dat_dtype", "float32")),
        layout=str(data_cfg.get("dat_layout", "sample_major")),
    )
    eeg, trigger = split_eeg_and_trigger(full, eeg_channels=int(data_cfg["eeg_channels"]))
    events = trigger_to_events(trigger)
    trials = extract_trial_infos_from_events(
        events=events,
        recording=recording,
        event_values=data_cfg["event_values"],
        sampling_rate_hz=int(data_cfg["sampling_rate_hz"]),
        task_start_offset_seconds=float(data_cfg["task_start_offset_seconds"]),
        task_window_seconds=float(data_cfg["task_window_seconds"]),
        n_samples=int(eeg.shape[1]),
    )
    print("Event counts:")
    values = [value for _, value in events]
    print({value: values.count(value) for value in sorted(set(values))})
    print("First events:")
    print(events[:40])
    print("Extracted trials:")
    print(json.dumps([trial.__dict__ | {"dat_path": str(trial.dat_path)} for trial in trials[:40]], ensure_ascii=False, indent=2))
    print(f"Total extracted trials: {len(trials)}")


if __name__ == "__main__":
    main()
