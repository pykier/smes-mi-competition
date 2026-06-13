"""PyTorch implementation of a compact EEGNet classifier."""

from __future__ import annotations

import torch
from torch import nn


class EEGNet(nn.Module):
    """Compact EEGNet for EEG window classification.

    Input shape is batch by channels by time.
    """

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_length: int = 64,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.n_classes = int(n_classes)

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_times)
            features = self._forward_features(dummy)
            n_features = features.shape[1]

        self.classifier = nn.Linear(n_features, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_features(x)
        return self.classifier(features)
