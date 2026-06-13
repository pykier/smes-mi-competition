"""Training pipeline for four official binary EEGNet tasks."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import load_config
from .dataset import EEGWindowDataset, build_all_task_arrays, split_train_val
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


def train_single_task(task_name: str, x: np.ndarray, y: np.ndarray, subjects: np.ndarray, config: dict, device: torch.device) -> dict:
    """Train one binary EEGNet task and return artifacts and metrics."""
    train_cfg = config["training"]
    model_cfg = config["model"]
    x_train, y_train, x_val, y_val = split_train_val(x, y, subjects, config)

    train_dataset = EEGWindowDataset(x_train, y_train)
    val_dataset = EEGWindowDataset(x_val, y_val) if len(y_val) > 0 else EEGWindowDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    model = EEGNet(
        n_channels=int(x.shape[1]),
        n_times=int(x.shape[2]),
        n_classes=2,
        F1=int(model_cfg.get("F1", 8)),
        D=int(model_cfg.get("D", 2)),
        F2=int(model_cfg.get("F2", 16)),
        kernel_length=int(model_cfg.get("kernel_length", 64)),
        dropout=float(model_cfg.get("dropout", 0.5)),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    best_val_acc = -1.0
    best_state = None
    history = []
    epochs = int(train_cfg.get("epochs", 30))
    for epoch in tqdm(range(1, epochs + 1), desc=f"Training {task_name}"):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if float(val_metrics["accuracy"]) > best_val_acc:
            best_val_acc = float(val_metrics["accuracy"])
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(
            f"[{task_name}] Epoch {epoch:03d} | train_acc={train_metrics['accuracy']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate_model(model, val_loader, criterion, device)
    sample_x = torch.tensor(x_val[:1] if len(x_val) else x_train[:1], dtype=torch.float32).to(device)

    def _predict_once(batch):
        model.eval()
        with torch.no_grad():
            return model(batch)

    inference_time = measure_inference_time(lambda batch=sample_x: _predict_once(batch), sample_x, repeat=50)
    return {
        "task_name": task_name,
        "model": model,
        "history": history,
        "final_val": final_val,
        "best_val_accuracy": best_val_acc,
        "single_trial_inference_time_seconds": inference_time,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_subjects": sorted(set(subjects[np.isin(np.arange(len(subjects)), np.where(np.isin(subjects, subjects))[0])].tolist())),
    }


def _json_safe_config(config: dict) -> dict:
    """Return a JSON-serializable copy of the config."""
    return json.loads(json.dumps(config, ensure_ascii=False))


def train(config_path: str = "configs/default.yaml") -> dict:
    """Train four EEGNet models and export competition model artifacts."""
    config = load_config(config_path)
    seed = int(config["project"].get("random_seed", 42))
    set_seed(seed)

    output_dir = Path(config["paths"]["output_dir"])
    artifacts_dir = Path(config["paths"].get("model_artifacts_dir", "model_artifacts"))
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    task_data, metadata = build_all_task_arrays(config)
    device = resolve_device(str(config["training"].get("device", "auto")))
    print(f"Using device: {device}")
    print(json.dumps(metadata["task_summaries"], ensure_ascii=False, indent=2))

    task_results = {}
    model_files = {}
    for task_name, (x, y, subjects) in task_data.items():
        result = train_single_task(task_name, x, y, subjects, config, device)
        model = result.pop("model")
        model_file = artifacts_dir / f"eegnet_{task_name}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "n_channels": int(x.shape[1]),
                "n_times": int(x.shape[2]),
                "n_classes": 2,
                "task_name": task_name,
                "task_config": config["tasks"][task_name],
                "model_config": config["model"],
            },
            model_file,
        )
        model_files[task_name] = str(model_file)
        task_results[task_name] = result

    artifact_config = {
        "config": _json_safe_config(config),
        "metadata": metadata,
        "model_files": {task: Path(path).name for task, path in model_files.items()},
        "task_order": list(config["tasks"].keys()),
        "predict_label_definition": {"0": "rest", "1": "target_action_or_imagery"},
    }
    artifact_config_path = artifacts_dir / "artifact_config.json"
    artifact_config_path.write_text(json.dumps(artifact_config, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "metadata": metadata,
        "task_results": task_results,
        "model_files": model_files,
        "artifact_config": str(artifact_config_path),
    }
    result_path = output_dir / "training_result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Keep a copy of the final config for reproducibility.
    shutil.copyfile(config_path, output_dir / "used_config.yaml")
    print(f"Saved model artifacts to: {artifacts_dir}")
    print(f"Saved training result to: {result_path}")
    return result
