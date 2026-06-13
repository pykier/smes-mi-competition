"""Compare validation results from multiple training runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_result(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        nargs="*",
        default=[
            "outputs/fbcsp_lda_result.json",
            "outputs/fbcsp_svm_result.json",
            "outputs/riemann_lr_result.json",
            "outputs/training_result.json",
        ],
    )
    args = parser.parse_args()

    rows = []
    for file_name in args.files:
        path = Path(file_name)
        if not path.exists():
            continue
        result = load_result(path)
        model_name = result.get("model_name", "eegnet")
        task_results = result.get("task_results", {})
        if not task_results:
            continue
        task_acc = {}
        for task_name, task_result in task_results.items():
            if "best_val_accuracy" in task_result:
                task_acc[task_name] = float(task_result["best_val_accuracy"])
            elif "final_val" in task_result:
                task_acc[task_name] = float(task_result["final_val"]["accuracy"])
        mean_acc = sum(task_acc.values()) / len(task_acc)
        rows.append((model_name, mean_acc, task_acc, str(path)))

    if not rows:
        print("No result files found.")
        return

    rows.sort(key=lambda item: item[1], reverse=True)
    print("\nModel comparison by mean validation accuracy:")
    for model_name, mean_acc, task_acc, path in rows:
        print(f"\n{model_name} | mean={mean_acc:.4f} | file={path}")
        for task_name, acc in task_acc.items():
            print(f"  {task_name}: {acc:.4f}")

    best = rows[0]
    print(f"\nBest local model: {best[0]} with mean validation accuracy {best[1]:.4f}")


if __name__ == "__main__":
    main()
