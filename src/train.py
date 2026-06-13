"""Training pipeline for the EEGNet baseline."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .dataset import EEGWindowDataset, build_windows, train_val_split
from .eegnet import EEGNet
from .evaluate import classification_report_dict, measure_inference_time


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    """Resolve configured device name."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def train_one_epoch(model, loader, optimizer, criterion, device) -> dict:
    """Train for one epoch."""
    model.train()
    losses = []
    all_true = []
    all_pred = []
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        all_true.extend(y_batch.detach().cpu().numpy().tolist())
        all_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
    metrics = classification_report_dict(all_true, all_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def evaluate_model(model, loader, criterion, device) -> dict:
    """Evaluate model on a dataloader."""
    model.eval()
    losses = []
    all_true = []
    all_pred = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            losses.append(float(loss.detach().cpu()))
            all_true.extend(y_batch.detach().cpu().numpy().tolist())
            all_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
    metrics = classification_report_dict(all_true, all_pred) if all_true else {"accuracy": 0.0, "confusion_matrix": []}
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def train(config_path: str = "configs/default.yaml") -> dict:
    """Train an EEGNet baseline from local data and save model artifacts."""
    config = load_config(config_path)
    seed = int(config["project"].get("random_seed", 42))
    set_seed(seed)

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, metadata = build_windows(config)
    x_train, y_train, x_val, y_val = train_val_split(
        x,
        y,
        validation_ratio=float(config["training"].get("validation_ratio", 0.2)),
        seed=seed,
    )

    train_dataset = EEGWindowDataset(x_train, y_train)
    val_dataset = EEGWindowDataset(x_val, y_val) if len(y_val) > 0 else EEGWindowDataset(x_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"].get("batch_size", 16)),
        shuffle=True,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(config["training"].get("num_workers", 0)),
    )

    n_classes = int(len(np.unique(y)))
    model_cfg = config["model"]
    model = EEGNet(
        n_channels=int(x.shape[1]),
        n_times=int(x.shape[2]),
        n_classes=n_classes,
        F1=int(model_cfg.get("F1", 8)),
        D=int(model_cfg.get("D", 2)),
        F2=int(model_cfg.get("F2", 16)),
        kernel_length=int(model_cfg.get("kernel_length", 64)),
        dropout=float(model_cfg.get("dropout", 0.5)),
    )

    device = resolve_device(str(config["training"].get("device", "auto")))
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )

    best_val_acc = -1.0
    best_state = None
    history = []
    for epoch in tqdm(range(1, int(config["training"].get("epochs", 5)) + 1), desc="Training EEGNet"):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        if float(val_metrics["accuracy"]) > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"Epoch {epoch:03d} | train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate_model(model, val_loader, criterion, device)

    model_save_path = Path(model_cfg.get("save_path", "outputs/eegnet_model.pt"))
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "metadata": metadata,
        "n_classes": n_classes,
        "n_channels": int(x.shape[1]),
        "n_times": int(x.shape[2]),
        "label_to_id": metadata.get("label_to_id", {}),
    }
    torch.save(checkpoint, model_save_path)

    sample_x = torch.tensor(x_val[:1] if len(x_val) else x_train[:1], dtype=torch.float32).to(device)

    def _predict_once(batch):
        model.eval()
        with torch.no_grad():
            return model(batch)

    inference_time = measure_inference_time(lambda batch=sample_x: _predict_once(batch), sample_x, repeat=20)
    result = {
        "metadata": metadata,
        "final_val": final_val,
        "best_val_accuracy": best_val_acc,
        "single_trial_inference_time_seconds": inference_time,
        "model_path": str(model_save_path),
        "history": history,
    }

    result_path = output_dir / "training_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model to: {model_save_path}")
    print(f"Saved result to: {result_path}")
    print(f"Average single-trial inference time: {inference_time:.6f} s")
    return result
