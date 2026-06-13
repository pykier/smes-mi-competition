"""Command-line entry for inference on one DAT file.

Run from repository root after training:
    python scripts/run_inference.py --model outputs/eegnet_model.pt --dat path/to/file.dat
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.predict import predict_dat_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/eegnet_model.pt", help="Path to saved EEGNet checkpoint.")
    parser.add_argument("--dat", required=True, help="Path to one local DAT file.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    args = parser.parse_args()

    predictions = predict_dat_file(args.model, args.dat, device=args.device)
    print("Predicted window labels:")
    print(predictions.tolist())


if __name__ == "__main__":
    main()
