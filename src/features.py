"""Feature extraction interfaces."""

import numpy as np


def extract_features(x: np.ndarray, method: str = "placeholder", **kwargs) -> np.ndarray:
    """Extract features from EEG trials.

    Parameters
    ----------
    x:
        Trial array.
    method:
        Feature extraction method name.

    Returns
    -------
    np.ndarray
        Feature matrix or feature tensor.
    """
    if method == "placeholder":
        return x.reshape(x.shape[0], -1) if x.ndim >= 2 else x
    raise NotImplementedError(f"Unknown feature method: {method}")
