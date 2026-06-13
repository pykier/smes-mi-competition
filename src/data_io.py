"""Data loading interfaces."""

from pathlib import Path
from typing import Any


def list_data_files(data_dir: str | Path) -> list[Path]:
    """List files under a data directory."""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory not found: {path}")
    return sorted(item for item in path.rglob("*") if item.is_file())


def load_eeg_file(file_path: str | Path) -> Any:
    """Load one EEG file.

    Placeholder function. Implement this after the competition data format is confirmed.
    """
    raise NotImplementedError("EEG file loader has not been implemented yet.")
