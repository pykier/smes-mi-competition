"""Electrode region presets for DH-CAN.

The paper divides BCI Competition IV 2a 22 channels into six motor-imagery
related regions and the 59-channel dataset into seven regions. The exact
figure is visual, so this file provides practical presets that follow the same
principle: left/right lateral sensorimotor regions, central/midline regions,
and posterior regions. For strict reproduction, replace these lists with the
exact region grouping used in the paper or your montage.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


BCICIV2A_CHANNELS_22: List[str] = [
    "Fz",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "P1",
    "Pz",
    "P2",
    "POz",
]

BCICIV2A_REGIONS_22: List[List[int]] = [
    [1, 2, 6, 7],          # left fronto-central / central motor
    [4, 5, 11, 12],        # right fronto-central / central motor
    [3, 8, 9, 10],         # middle sensorimotor strip
    [13, 14, 18],          # left centro-parietal / parietal
    [16, 17, 20],          # right centro-parietal / parietal
    [0, 15, 19, 21],       # midline frontal/posterior
]
BCICIV2A_SYMMETRIC_PAIRS: List[Tuple[int, int]] = [(0, 1), (3, 4)]

META16_CHANNELS: List[str] = [
    "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CP2", "CP4",
]
META16_REGIONS: List[List[int]] = [
    [0, 1, 5, 6, 7],       # left motor-related region
    [3, 4, 9, 10, 11],     # right motor-related region
    [2, 8],                # midline FCz/Cz
    [12, 13],              # left CP region
    [14, 15],              # right CP region
]
META16_SYMMETRIC_PAIRS: List[Tuple[int, int]] = [(0, 1), (3, 4)]


def channel_names_to_indices(
    region_names: Sequence[Sequence[str]],
    channel_names: Sequence[str],
) -> List[List[int]]:
    lookup = {name: i for i, name in enumerate(channel_names)}
    regions: List[List[int]] = []
    for region in region_names:
        missing = [name for name in region if name not in lookup]
        if missing:
            raise ValueError(f"Unknown channels in region {region}: {missing}")
        regions.append([lookup[name] for name in region])
    return regions


def get_region_preset(name: str) -> Tuple[List[List[int]], List[Tuple[int, int]]]:
    name = name.lower()
    if name in {"bciciv2a", "bciciv2a_22", "2a", "22"}:
        return [r.copy() for r in BCICIV2A_REGIONS_22], BCICIV2A_SYMMETRIC_PAIRS.copy()
    if name in {"meta16", "metabci16", "16"}:
        return [r.copy() for r in META16_REGIONS], META16_SYMMETRIC_PAIRS.copy()
    raise ValueError(
        f"Unknown region preset {name!r}. Available: 'bciciv2a_22', 'meta16'."
    )
