"""Convert BCI Competition IV 2a GDF files to DH-CAN NPZ files.

Optional helper. Requires:
    pip install mne scipy

Example:
    python prepare_bciciv2a.py --gdf A01T.gdf --out A01T_dhcan.npz --tmin 0 --tmax 4

BCI IV 2a labels:
    769 left hand, 770 right hand, 771 feet, 772 tongue.
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdf", type=str, required=True, help="Path to BCI IV 2a .gdf file.")
    parser.add_argument("--out", type=str, required=True, help="Output .npz path.")
    parser.add_argument("--tmin", type=float, default=0.0, help="Epoch start relative to cue/event.")
    parser.add_argument("--tmax", type=float, default=4.0, help="Epoch end relative to cue/event.")
    parser.add_argument("--l-freq", type=float, default=4.0)
    parser.add_argument("--h-freq", type=float, default=40.0)
    parser.add_argument("--resample", type=float, default=250.0)
    parser.add_argument("--no-filter", action="store_true")
    args = parser.parse_args()

    try:
        import mne
    except ImportError as e:
        raise SystemExit("mne is required: pip install mne scipy") from e

    raw = mne.io.read_raw_gdf(args.gdf, preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, eog=False, stim=False)
    if not args.no_filter:
        raw.filter(args.l_freq, args.h_freq, fir_design="firwin", verbose="ERROR")
    if args.resample:
        raw.resample(args.resample, verbose="ERROR")

    events, event_id = mne.events_from_annotations(raw, verbose="ERROR")
    wanted = {"769": 0, "770": 1, "771": 2, "772": 3}
    reverse_event_id: Dict[int, str] = {v: k for k, v in event_id.items()}
    selected_events = []
    labels = []
    for event in events:
        label_name = reverse_event_id.get(int(event[2]), "")
        if label_name in wanted:
            selected_events.append(event)
            labels.append(wanted[label_name])
    if not selected_events:
        raise RuntimeError(f"No MI cue annotations 769-772 found. Available annotations: {event_id}")

    selected_events = np.asarray(selected_events, dtype=int)
    labels = np.asarray(labels, dtype=np.int64)
    epochs = mne.Epochs(
        raw,
        selected_events,
        event_id=None,
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )
    X = epochs.get_data().astype(np.float32)
    target_samples = int(round((args.tmax - args.tmin) * float(raw.info["sfreq"])))
    if X.shape[-1] == target_samples + 1:
        X = X[..., :-1]

    np.savez_compressed(args.out, X=X, y=labels, sfreq=float(raw.info["sfreq"]), ch_names=np.array(epochs.ch_names))
    print(f"Saved {args.out}: X={X.shape}, y={labels.shape}, sfreq={raw.info['sfreq']}")


if __name__ == "__main__":
    main()
