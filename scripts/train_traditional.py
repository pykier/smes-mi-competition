"""Train traditional EEG models for the four official SMES-MI tasks.

Examples:
    python scripts/train_traditional.py --model fbcsp_lda --config configs/default.yaml
    python scripts/train_traditional.py --model fbcsp_svm --config configs/default.yaml
    python scripts/train_traditional.py --model riemann_lr --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.dataset import build_all_task_arrays, split_train_val
from src.traditional_models import build_traditional_model, evaluate_predictions, save_traditional_model


def make_json_safe(obj):
    """Convert numpy/path objects to JSON-compatible structures."""
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def train_one_task(model_name: str, task_name: str, x: np.ndarray, y: np.ndarray, subjects: np.ndarray, config: dict):
    """Train and validate one task."""
    x_train, y_train, x_val, y_val = split_train_val(x, y, subjects, config)
    fs = int(config["preprocess"].get("target_sampling_rate_hz") or config["data"]["sampling_rate_hz"])
    model = build_traditional_model(model_name, fs=fs)
    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - t0

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    train_metrics = evaluate_predictions(y_train, train_pred)
    val_metrics = evaluate_predictions(y_val, val_pred)

    if len(x_val) > 0:
        sample = x_val[:1]
    else:
        sample = x_train[:1]
    t1 = time.perf_counter()
    repeat = 100
    for _ in range(repeat):
        model.predict(sample)
    inference_time = (time.perf_counter() - t1) / repeat

    result = {
        "task_name": task_name,
        "model_name": model_name,
        "train": train_metrics,
        "val": val_metrics,
        "best_val_accuracy": float(val_metrics["accuracy"]),
        "single_trial_inference_time_seconds": float(inference_time),
        "train_seconds": float(train_seconds),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
        "val_class_counts": {str(k): int(v) for k, v in zip(*np.unique(y_val, return_counts=True))},
    }
    return model, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["fbcsp_lda", "fbcsp_svm", "riemann_lr"], required=True)
    parser.add_argument("--artifacts", default=None, help="Output artifact directory. Defaults to model_artifacts_<model>.")
    parser.add_argument("--output", default=None, help="Output result JSON path. Defaults to outputs/<model>_result.json.")
    parser.add_argument("--disable-broad-bandpass", action="store_true", help="Set preprocess.bandpass_hz to null before feature extraction.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.disable_broad_bandpass:
        config["preprocess"]["bandpass_hz"] = None

    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(args.artifacts or f"model_artifacts_{args.model}")
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    task_data, metadata = build_all_task_arrays(config)
    print(json.dumps(metadata["task_summaries"], ensure_ascii=False, indent=2))

    task_results = {}
    model_files = {}
    for task_name, (x, y, subjects) in task_data.items():
        print(f"\nTraining {args.model} for {task_name}: x={x.shape}, y={np.bincount(y)}")
        model, result = train_one_task(args.model, task_name, x, y, subjects, config)
        model_file = artifact_dir / f"{args.model}_{task_name}.joblib"
        save_traditional_model(model, str(model_file))
        model_files[task_name] = model_file.name
        task_results[task_name] = result
        print(
            f"{task_name}: val_acc={result['val']['accuracy']:.4f}, "
            f"infer={result['single_trial_inference_time_seconds']:.6f}s"
        )

    mean_best = float(np.mean([v["best_val_accuracy"] for v in task_results.values()]))
    artifact_config = {
        "backend": "traditional",
        "traditional_model_name": args.model,
        "config": make_json_safe(config),
        "metadata": make_json_safe(metadata),
        "model_files": model_files,
        "task_order": list(config["tasks"].keys()),
        "predict_label_definition": {"0": "rest", "1": "target_action_or_imagery"},
    }
    (artifact_dir / "artifact_config.json").write_text(
        json.dumps(artifact_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result = {
        "model_name": args.model,
        "mean_best_val_accuracy": mean_best,
        "metadata": make_json_safe(metadata),
        "task_results": make_json_safe(task_results),
        "artifact_dir": str(artifact_dir),
    }
    result_path = Path(args.output or output_dir / f"{args.model}_result.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved artifacts to: {artifact_dir}")
    print(f"Saved result to: {result_path}")
    print(f"Mean best validation accuracy: {mean_best:.4f}")


if __name__ == "__main__":
    main()
