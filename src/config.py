"""Configuration utilities."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path = "configs/default.yaml") -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    return config
