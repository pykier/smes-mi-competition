"""Model interfaces."""

from dataclasses import dataclass
from typing import Any

import joblib


@dataclass
class CompetitionModel:
    """Lightweight wrapper for competition classifiers."""

    estimator: Any
    metadata: dict

    def fit(self, x, y):
        self.estimator.fit(x, y)
        return self

    def predict(self, x):
        return self.estimator.predict(x)


def save_model(model: CompetitionModel, path: str) -> None:
    """Save model object."""
    joblib.dump(model, path)


def load_model(path: str) -> CompetitionModel:
    """Load model object."""
    return joblib.load(path)
