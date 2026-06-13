"""Training entry functions."""

from pathlib import Path

from .config import load_config


def train(config_path: str = "configs/default.yaml") -> None:
    """Train a model.

    Placeholder entry. Implement after data reading and baseline algorithm are confirmed.
    """
    config = load_config(config_path)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError("Training pipeline has not been implemented yet.")
