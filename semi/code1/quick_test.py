"""Smoke test for DH-CAN.

Run:
    python quick_test.py
It uses a shortened artificial signal so it runs quickly on CPU.
"""

import torch
from torch import nn

from dhcan_model import build_dhcan
from region_config import get_region_preset


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    regions, pairs = get_region_preset("bciciv2a_22")
    model = build_dhcan(
        n_channels=22,
        n_classes=4,
        sfreq=64,
        input_samples=256,
        region_indices=regions,
        symmetric_pairs=pairs,
        feature_time=8,
    )
    x = torch.randn(2, 22, 256)
    y = torch.randint(0, 4, (2,))
    logits, att = model(x, return_attention=True)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    print("logits shape:", tuple(logits.shape))
    print("attention shape:", tuple(att.shape))
    print("loss:", float(loss.detach()))
    print("parameter count:", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    main()
