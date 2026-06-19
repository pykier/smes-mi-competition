"""Train DH-CAN on an EEG NPZ dataset.

Example:
    python train.py --data ./data/A01T_epoch.npz --preset bciciv2a_22 --epochs 500

NPZ format:
    X: (N, C, T), float32/float64
    y: (N,), int labels from 0 to n_classes-1
Optional split arrays:
    train_idx, val_idx, test_idx
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data_utils import compute_metrics, load_npz_data, make_datasets
from dhcan_model import build_dhcan
from region_config import get_region_preset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_epoch(model, loader, criterion, optimizer, device: str, train: bool) -> Tuple[float, Dict[str, float]]:
    model.train(train)
    all_true, all_pred = [], []
    total_loss, total_n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        total_loss += float(loss.item()) * len(y)
        total_n += len(y)
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    metrics = compute_metrics(y_true, y_pred, n_classes=model.config.n_classes)
    return total_loss / max(total_n, 1), metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to NPZ file with X and y arrays.")
    parser.add_argument("--preset", type=str, default="bciciv2a_22", choices=["bciciv2a_22", "meta16"])
    parser.add_argument("--sfreq", type=int, default=250)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--normalize", type=str, default="trial", choices=["trial", "none"])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="runs/dhcan")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-threads", type=int, default=0, help="Set torch CPU threads; 0 keeps PyTorch default.")
    args = parser.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz_data(args.data, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    train_ds, val_ds, test_ds = make_datasets(data, normalize=args.normalize)
    n_channels = int(data.X.shape[1])
    input_samples = int(data.X.shape[2])

    regions, pairs = get_region_preset(args.preset)
    if n_channels == 16 and args.preset != "meta16":
        regions, pairs = get_region_preset("meta16")
    if n_channels == 22 and args.preset != "bciciv2a_22":
        print("Warning: using preset", args.preset, "with 22-channel data.")

    model = build_dhcan(
        n_channels=n_channels,
        n_classes=args.n_classes,
        sfreq=args.sfreq,
        input_samples=input_samples,
        region_indices=regions,
        symmetric_pairs=pairs,
        dropout=args.dropout,
    ).to(args.device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = -1.0
    best_path = out_dir / "best_dhcan.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_m = run_epoch(model, train_loader, criterion, optimizer, args.device, train=True)
        va_loss, va_m = run_epoch(model, val_loader, criterion, optimizer, args.device, train=False)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            **{f"train_{k}": v for k, v in tr_m.items()},
            "val_loss": va_loss,
            **{f"val_{k}": v for k, v in va_m.items()},
        }
        history.append(row)
        if va_m["acc"] > best_val:
            best_val = va_m["acc"]
            torch.save({"model": model.state_dict(), "args": vars(args), "regions": regions, "pairs": pairs}, best_path)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {tr_loss:.4f} acc {tr_m['acc']:.4f} | "
                f"val loss {va_loss:.4f} acc {va_m['acc']:.4f} kappa {va_m['kappa']:.4f}"
            )

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(best_path, map_location=args.device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_m = run_epoch(model, test_loader, criterion, optimizer, args.device, train=False)
    print("Test:", {"loss": te_loss, **te_m})
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"loss": te_loss, **te_m}, f, indent=2)


if __name__ == "__main__":
    main()
