"""Command-line inference for one already-cut competition trial saved as NPY.

Example:
    python scripts/run_inference.py --npy trial.npy --task vmi_left_vs_rest

The NPY array should be shaped as channels by samples. This mirrors the official
predict-stage input more closely than reading a full raw DAT file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_competition_trial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", required=True, help="Path to one trial array saved as .npy, shape channels by samples.")
    parser.add_argument("--task", default="vmi_left_vs_rest", help="One of the four task names.")
    parser.add_argument("--artifacts", default="model_artifacts", help="Path to model_artifacts directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    args = parser.parse_args()

    trial = np.load(args.npy)
    pred = predict_competition_trial(trial, task_name=args.task, model_artifacts_dir=args.artifacts, device=args.device)
    print(int(pred))


if __name__ == "__main__":
    main()
