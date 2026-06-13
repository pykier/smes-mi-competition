"""Inference entry functions."""

from .model import load_model


def predict(model_path: str, x):
    """Run prediction with a saved model."""
    model = load_model(model_path)
    return model.predict(x)
