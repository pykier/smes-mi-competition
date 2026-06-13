"""Inspect local EEG DAT and trigger files before training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.dataset import build_manifest, inspect_one_recording


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--head", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = build_manifest(config)
    print("\nManifest head:")
    print(manifest.head(args.head).to_string(index=False))
    print("\nSummary:")
    print(f"files: {len(manifest)}")
    print(manifest.groupby(["run_type"]).size())
    print(f"meta files found: {(manifest['meta_path'] != '').sum()} / {len(manifest)}")

    n_channels = int(config["data"]["total_channels"])
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

    print("\nTrigger and channel check for first file:")
    diagnostic = inspect_one_recording(manifest.iloc[0]["dat_path"], config)
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
