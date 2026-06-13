"""Traditional EEG decoding models for SMES-MI tasks.

Implemented models:

- FBCSP + shrinkage LDA
- FBCSP + linear SVM
- Log-covariance Riemannian features + Logistic Regression

All models use scikit-learn/joblib artifacts and are intentionally lightweight for
competition constraints: small model size, fast inference, and <= 8 channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import joblib
import numpy as np
from scipy.linalg import eigh, logm
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


Array = np.ndarray


def _safe_bandpass(x: Array, fs: int, low: float, high: float, order: int = 4) -> Array:
    """Band-pass filter a batch of EEG epochs shaped trials by channels by time."""
    if low <= 0 or high <= low or high >= fs / 2:
        raise ValueError(f"Invalid bandpass: low={low}, high={high}, fs={fs}")
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="bandpass")
    padlen = min(3 * max(len(a), len(b)), max(1, x.shape[-1] - 1))
    return filtfilt(b, a, x, axis=-1, padlen=padlen).astype(np.float32, copy=False)


def regularized_cov(epoch: Array, reg: float = 1e-6) -> Array:
    """Return trace-normalized regularized covariance for one epoch."""
    x = epoch - epoch.mean(axis=-1, keepdims=True)
    cov = x @ x.T / max(1, x.shape[-1] - 1)
    tr = float(np.trace(cov))
    if tr > 0:
        cov = cov / tr
    cov = cov + reg * np.eye(cov.shape[0], dtype=np.float32)
    return cov.astype(np.float64)


def batch_covariances(x: Array, reg: float = 1e-6) -> Array:
    """Compute covariance matrices for epochs shaped trials by channels by time."""
    return np.stack([regularized_cov(epoch, reg=reg) for epoch in x]).astype(np.float64)


@dataclass
class CSPTransformer:
    """Binary CSP transformer for one frequency band."""

    n_components_per_side: int = 2
    reg: float = 1e-6
    filters_: Array | None = None

    def fit(self, x: Array, y: Array) -> "CSPTransformer":
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires binary labels, got {classes}")
        covs = batch_covariances(x, reg=self.reg)
        c0 = covs[y == classes[0]].mean(axis=0)
        c1 = covs[y == classes[1]].mean(axis=0)
        composite = c0 + c1
        eigenvalues, eigenvectors = eigh(c1, composite)
        order = np.argsort(eigenvalues)
        pick = np.r_[order[: self.n_components_per_side], order[-self.n_components_per_side :]]
        self.filters_ = eigenvectors[:, pick].T.astype(np.float64)
        return self

    def transform(self, x: Array) -> Array:
        if self.filters_ is None:
            raise RuntimeError("CSPTransformer has not been fitted.")
        feats = []
        for epoch in x:
            projected = self.filters_ @ epoch
            var = np.var(projected, axis=-1)
            var = var / max(np.sum(var), 1e-12)
            feats.append(np.log(var + 1e-12))
        return np.asarray(feats, dtype=np.float32)


class FBCSPClassifier:
    """Filter-bank CSP classifier for binary EEG tasks."""

    def __init__(
        self,
        fs: int,
        bands: list[tuple[float, float]] | None = None,
        n_components_per_side: int = 2,
        classifier: Literal["lda", "svm"] = "lda",
        csp_reg: float = 1e-6,
    ):
        self.fs = int(fs)
        self.bands = bands or [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
        self.n_components_per_side = int(n_components_per_side)
        self.classifier = classifier
        self.csp_reg = float(csp_reg)
        self.csp_list_: list[CSPTransformer] = []
        self.clf_: Pipeline | None = None

    def _features(self, x: Array, fit: bool = False, y: Array | None = None) -> Array:
        all_features = []
        if fit:
            self.csp_list_ = []
        for band_index, (low, high) in enumerate(self.bands):
            xb = _safe_bandpass(x, fs=self.fs, low=float(low), high=float(high))
            if fit:
                csp = CSPTransformer(n_components_per_side=self.n_components_per_side, reg=self.csp_reg).fit(xb, y)
                self.csp_list_.append(csp)
            else:
                csp = self.csp_list_[band_index]
            all_features.append(csp.transform(xb))
        return np.concatenate(all_features, axis=1).astype(np.float32)

    def fit(self, x: Array, y: Array) -> "FBCSPClassifier":
        feats = self._features(x, fit=True, y=y)
        if self.classifier == "lda":
            estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        elif self.classifier == "svm":
            estimator = SVC(kernel="linear", C=0.25, class_weight="balanced")
        else:
            raise ValueError(f"Unsupported classifier: {self.classifier}")
        self.clf_ = Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])
        self.clf_.fit(feats, y)
        return self

    def predict(self, x: Array) -> Array:
        if self.clf_ is None:
            raise RuntimeError("FBCSPClassifier has not been fitted.")
        feats = self._features(x, fit=False)
        return self.clf_.predict(feats).astype(np.int64)


class LogCovLRClassifier:
    """Log-covariance Riemannian feature classifier.

    The feature is the upper triangle of the matrix logarithm of a regularized,
    trace-normalized covariance matrix. It is small and robust for low-channel EEG.
    """

    def __init__(self, reg: float = 1e-5, C: float = 0.1):
        self.reg = float(reg)
        self.C = float(C)
        self.triu_indices_: tuple[Array, Array] | None = None
        self.clf_: Pipeline | None = None

    def _features(self, x: Array) -> Array:
        covs = batch_covariances(x, reg=self.reg)
        n_channels = covs.shape[1]
        if self.triu_indices_ is None:
            self.triu_indices_ = np.triu_indices(n_channels)
        feats = []
        for cov in covs:
            log_cov = np.real(logm(cov))
            feats.append(log_cov[self.triu_indices_])
        return np.asarray(feats, dtype=np.float32)

    def fit(self, x: Array, y: Array) -> "LogCovLRClassifier":
        feats = self._features(x)
        estimator = LogisticRegression(
            C=self.C,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        self.clf_ = Pipeline([("scaler", StandardScaler()), ("estimator", estimator)])
        self.clf_.fit(feats, y)
        return self

    def predict(self, x: Array) -> Array:
        if self.clf_ is None:
            raise RuntimeError("LogCovLRClassifier has not been fitted.")
        return self.clf_.predict(self._features(x)).astype(np.int64)


def build_traditional_model(model_name: str, fs: int):
    """Factory for traditional model names used by scripts."""
    if model_name == "fbcsp_lda":
        return FBCSPClassifier(fs=fs, classifier="lda", n_components_per_side=2)
    if model_name == "fbcsp_svm":
        return FBCSPClassifier(fs=fs, classifier="svm", n_components_per_side=2)
    if model_name == "riemann_lr":
        return LogCovLRClassifier(reg=1e-5, C=0.1)
    raise ValueError(f"Unknown traditional model: {model_name}")


def evaluate_predictions(y_true: Array, y_pred: Array) -> dict:
    """Return compact classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_traditional_model(model, path: str) -> None:
    """Save model with joblib."""
    joblib.dump(model, path)


def load_traditional_model(path: str):
    """Load model with joblib."""
    return joblib.load(path)
