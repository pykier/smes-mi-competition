"""Inspect local EEG DAT files before training.

Run from repository root:
    python scripts/inspect_data.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.dataset import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--head", type=int, default=20)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = build_manifest(config)
    print("\nManifest head:")
    print(manifest.head(args.head).to_string(index=False))
    print("\nLabel mapping:")
    print(manifest.attrs.get("label_to_id", {}))
    print("\nSummary:")
    print(f"files: {len(manifest)}")
    print(manifest.groupby(["label_name"]).size())

    n_channels = int(config["data"]["n_channels"])
    dtype = np.dtype(str(config["data"].get("dat_dtype", "float32")))
    print("\nDAT shape check:")
    for path in manifest["dat_path"].head(args.head):
        size_bytes = Path(path).stat().st_size
        n_values = size_bytes // dtype.itemsize
        remainder = n_values % n_channels
        n_samples = n_values // n_channels
        print(
            f"{path} | {size_bytes / (1024 ** 2):.2f} MB | dtype={dtype} | "
            f"values={n_values} | samples_if_{n_channels}ch={n_samples} | remainder={remainder}"
        )


if __name__ == "__main__":
    main()
