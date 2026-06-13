"""Command-line entry for EEGNet baseline training.

Run from repository root:
    python scripts/train_baseline.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config file.")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
