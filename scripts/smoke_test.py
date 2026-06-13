"""Run a tiny synthetic-data smoke test without local competition data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eegnet import EEGNet


def main() -> None:
    rng = np.random.default_rng(42)
    x = torch.tensor(rng.normal(size=(8, 8, 1000)), dtype=torch.float32)
    y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    model = EEGNet(n_channels=8, n_times=1000, n_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    print("Smoke test passed.")
    print(f"logits shape: {tuple(logits.shape)}")
    print(f"loss: {float(loss.detach()):.6f}")


if __name__ == "__main__":
    main()
