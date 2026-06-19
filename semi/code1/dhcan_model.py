"""DH-CAN: Dynamic Hierarchical Convolutional Attention Network.

Self-contained PyTorch reproduction of the paper algorithm:
1) multiscale dynamic temporal convolution;
2) global + local hierarchical spatial convolution;
3) symmetric local-region parameter sharing;
4) dense region-level GAT/GATv2-like attention;
5) high-level global/local feature fusion classifier.

Input:  x with shape (B, C, T) or (B, 1, C, T)
Output: logits with shape (B, n_classes)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class DHCANConfig:
    n_channels: int = 22
    n_classes: int = 4
    sfreq: int = 250
    input_samples: int = 1000
    temporal_filters: int = 8
    temporal_kernel_ratios: Tuple[float, float, float] = (0.25, 0.125, 0.0625)
    temporal_pool_sizes: Tuple[int, int, int] = (4, 2, 1)
    spatial_filters: int = 16
    fusion_filters: int = 16
    sep_kernel: int = 16
    pool1: int = 4
    pool2: int = 8
    dropout: float = 0.5
    feature_time: int = 23
    gat_hidden: int = 16
    gat_heads: int = 1
    gat_dropout: float = 0.1


class SamePadConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int], groups: int = 1, bias: bool = False) -> None:
        super().__init__()
        kh, kw = kernel_size
        ph, pw = kh - 1, kw - 1
        self.pad = nn.ZeroPad2d((pw // 2, pw - pw // 2, ph // 2, ph - ph // 2))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=groups, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(self.pad(x))


class DynamicTemporalConv(nn.Module):
    """Multiscale temporal convolution. At 250 Hz kernels become 64/32/16."""

    def __init__(self, sfreq: int, out_channels: int = 8, kernel_ratios: Sequence[float] = (0.25, 0.125, 0.0625), pool_sizes: Sequence[int] = (4, 2, 1)) -> None:
        super().__init__()
        if len(kernel_ratios) != len(pool_sizes):
            raise ValueError("kernel_ratios and pool_sizes must have the same length")
        self.kernel_sizes = [self._round_kernel(sfreq * r) for r in kernel_ratios]
        self.branches = nn.ModuleList([
            nn.Sequential(
                SamePadConv2d(1, out_channels, kernel_size=(1, k), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.AvgPool2d(kernel_size=(1, p), stride=(1, p)),
            )
            for k, p in zip(self.kernel_sizes, pool_sizes)
        ])

    @staticmethod
    def _round_kernel(value: float) -> int:
        k = int(round(value))
        k = max(k, 3)
        if k % 2 == 1:
            k += 1
        return k

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape (B,C,T) or (B,1,C,T), got {tuple(x.shape)}")
        return torch.cat([branch(x) for branch in self.branches], dim=-1)


class DepthwiseSeparableTemporalConv(nn.Module):
    def __init__(self, channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__()
        self.depthwise = SamePadConv2d(channels, channels, kernel_size=(1, kernel_size), groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class SpatialConvBlock(nn.Module):
    """Global/local spatial block: depthwise spatial conv + separable temporal conv."""

    def __init__(self, in_filters: int, region_channels: int, spatial_filters: int = 16, sep_kernel: int = 16, pool1: int = 4, pool2: int = 8, dropout: float = 0.5, feature_time: int = 23) -> None:
        super().__init__()
        if region_channels < 1:
            raise ValueError("region_channels must be positive")
        groups = in_filters if spatial_filters % in_filters == 0 else 1
        self.depthwise_spatial = nn.Conv2d(in_filters, spatial_filters, kernel_size=(region_channels, 1), groups=groups, bias=False)
        self.bn1 = nn.BatchNorm2d(spatial_filters)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool1), stride=(1, pool1))
        self.drop1 = nn.Dropout(dropout)
        self.sep_temporal = DepthwiseSeparableTemporalConv(spatial_filters, spatial_filters, kernel_size=sep_kernel)
        self.bn2 = nn.BatchNorm2d(spatial_filters)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2), stride=(1, pool2))
        self.drop2 = nn.Dropout(dropout)
        self.final_pool = nn.AdaptiveAvgPool2d((1, feature_time))

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise_spatial(x)
        x = F.elu(self.bn1(x))
        x = self.drop1(self.pool1(x))
        x = self.sep_temporal(x)
        x = F.elu(self.bn2(x))
        x = self.drop2(self.pool2(x))
        return self.final_pool(x)


class DenseRegionGAT(nn.Module):
    """Dense region-level GAT without torch_geometric.

    local_maps: (B, R, F, 1, T)
    returns: optimized maps (B, R, F, 1, T), attention (B, heads, R, R)
    """

    def __init__(self, in_features: int, hidden_features: int = 16, heads: int = 1, dropout: float = 0.1, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.ModuleList([nn.Linear(in_features, hidden_features, bias=False) for _ in range(heads)])
        self.att = nn.ModuleList([nn.Linear(2 * hidden_features, 1, bias=False) for _ in range(heads)])
        self.negative_slope = negative_slope

    def forward(self, local_maps: Tensor) -> Tuple[Tensor, Tensor]:
        if local_maps.ndim != 5:
            raise ValueError("local_maps must have shape (B, R, F, 1, T)")
        b, r, f, h, t = local_maps.shape
        if h != 1:
            raise ValueError("spatial block height must be 1")
        maps = local_maps.squeeze(3)       # (B, R, F, T)
        nodes = maps.mean(dim=-1)          # (B, R, F)
        outs, alphas = [], []
        for proj, att in zip(self.proj, self.att):
            z = proj(nodes)
            zi = z.unsqueeze(2).expand(b, r, r, z.size(-1))
            zj = z.unsqueeze(1).expand(b, r, r, z.size(-1))
            e = att(torch.cat([zi, zj], dim=-1)).squeeze(-1)
            e = F.leaky_relu(e, negative_slope=self.negative_slope)
            alpha = self.dropout(torch.softmax(e, dim=-1))
            out = torch.einsum("bij,bjft->bift", alpha, maps)
            outs.append(out)
            alphas.append(alpha)
        out = torch.stack(outs, dim=0).mean(dim=0)
        alpha_all = torch.stack(alphas, dim=1)
        return out.unsqueeze(3), alpha_all


class RegionSpatialExtractor(nn.Module):
    """Apply local spatial blocks. Symmetric regions share parameters when possible."""

    def __init__(self, region_indices: Sequence[Sequence[int]], symmetric_pairs: Optional[Sequence[Tuple[int, int]]], in_filters: int, spatial_filters: int, sep_kernel: int, pool1: int, pool2: int, dropout: float, feature_time: int) -> None:
        super().__init__()
        self.region_indices = [list(r) for r in region_indices]
        pair_base: Dict[int, int] = {}
        if symmetric_pairs is not None:
            for a, b in symmetric_pairs:
                if a < len(self.region_indices) and b < len(self.region_indices) and len(self.region_indices[a]) == len(self.region_indices[b]):
                    pair_base[b] = a
        self.region_keys: List[str] = []
        self.blocks = nn.ModuleDict()
        for i in range(len(self.region_indices)):
            base = pair_base.get(i, i)
            key = f"region_{base}"
            if key not in self.blocks:
                self.blocks[key] = SpatialConvBlock(
                    in_filters=in_filters,
                    region_channels=len(self.region_indices[base]),
                    spatial_filters=spatial_filters,
                    sep_kernel=sep_kernel,
                    pool1=pool1,
                    pool2=pool2,
                    dropout=dropout,
                    feature_time=feature_time,
                )
            self.region_keys.append(key)

    def forward(self, zt: Tensor) -> Tensor:
        outs = []
        for idx, key in zip(self.region_indices, self.region_keys):
            outs.append(self.blocks[key](zt[:, :, idx, :]))
        return torch.stack(outs, dim=1)


class DHCAN(nn.Module):
    def __init__(self, config: DHCANConfig, region_indices: Sequence[Sequence[int]], symmetric_pairs: Optional[Sequence[Tuple[int, int]]] = None) -> None:
        super().__init__()
        self.config = config
        self.region_indices = [list(r) for r in region_indices]
        self.n_regions = len(self.region_indices)
        self.temporal = DynamicTemporalConv(config.sfreq, config.temporal_filters, config.temporal_kernel_ratios, config.temporal_pool_sizes)
        self.global_spatial = SpatialConvBlock(config.temporal_filters, config.n_channels, config.spatial_filters, config.sep_kernel, config.pool1, config.pool2, config.dropout, config.feature_time)
        self.local_spatial = RegionSpatialExtractor(self.region_indices, symmetric_pairs, config.temporal_filters, config.spatial_filters, config.sep_kernel, config.pool1, config.pool2, config.dropout, config.feature_time)
        self.region_gat = DenseRegionGAT(config.spatial_filters, config.gat_hidden, config.gat_heads, config.gat_dropout)
        self.fusion = nn.Sequential(
            nn.Conv2d(config.spatial_filters, config.fusion_filters, kernel_size=(self.n_regions + 1, 1), bias=False),
            nn.BatchNorm2d(config.fusion_filters),
            nn.ELU(),
        )
        self.classifier = nn.Linear(config.fusion_filters * config.feature_time, config.n_classes)

    def forward(self, x: Tensor, return_attention: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        zt = self.temporal(x)
        global_map = self.global_spatial(zt)              # (B, F, 1, L)
        local_maps = self.local_spatial(zt)               # (B, R, F, 1, L)
        local_maps, att = self.region_gat(local_maps)
        global_f = global_map.squeeze(2).unsqueeze(2)     # (B, F, 1, L)
        local_f = local_maps.squeeze(3).permute(0, 2, 1, 3)  # (B, F, R, L)
        fused = torch.cat([global_f, local_f], dim=2)     # (B, F, R+1, L)
        fused = self.fusion(fused).squeeze(2)             # (B, F, L)
        logits = self.classifier(fused.flatten(start_dim=1))
        return (logits, att) if return_attention else logits


def validate_region_indices(region_indices: Sequence[Sequence[int]], n_channels: int) -> None:
    flat = [i for region in region_indices for i in region]
    if not flat:
        raise ValueError("region_indices is empty")
    if min(flat) < 0 or max(flat) >= n_channels:
        raise ValueError(f"region index out of range for n_channels={n_channels}: {flat}")
    missing = sorted(set(range(n_channels)) - set(flat))
    duplicated = sorted({i for i in flat if flat.count(i) > 1})
    if missing:
        raise ValueError(f"region_indices does not cover channels: {missing}")
    if duplicated:
        raise ValueError(f"region_indices has duplicated channels: {duplicated}")


def make_contiguous_regions(n_channels: int, n_regions: int = 6) -> List[List[int]]:
    chunks: List[List[int]] = [[] for _ in range(n_regions)]
    for i in range(n_channels):
        chunks[int(i * n_regions / n_channels)].append(i)
    return chunks


def build_dhcan(n_channels: int, n_classes: int, sfreq: int, input_samples: int, region_indices: Optional[Sequence[Sequence[int]]] = None, symmetric_pairs: Optional[Sequence[Tuple[int, int]]] = None, **kwargs) -> DHCAN:
    cfg = DHCANConfig(n_channels=n_channels, n_classes=n_classes, sfreq=sfreq, input_samples=input_samples, **kwargs)
    if region_indices is None:
        region_indices = make_contiguous_regions(n_channels, n_regions=6)
        symmetric_pairs = None
    validate_region_indices(region_indices, n_channels)
    return DHCAN(cfg, region_indices=region_indices, symmetric_pairs=symmetric_pairs)


if __name__ == "__main__":
    from region_config import get_region_preset
    regions, pairs = get_region_preset("bciciv2a_22")
    model = build_dhcan(22, 4, 250, 1000, region_indices=regions, symmetric_pairs=pairs)
    x = torch.randn(2, 22, 1000)
    logits, att = model(x, return_attention=True)
    print("logits:", tuple(logits.shape))
    print("attention:", tuple(att.shape))
    print("parameters:", sum(p.numel() for p in model.parameters()))
