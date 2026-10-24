"""Stratified per-class train/valid/test splitting (pure Python, no torch).

The video builds splits per class with a seeded RNG (seed 42) and fixed
per-class targets, e.g. pCT -> train 2800 / valid 600 / test 601 (70/15/15).
We reproduce that behaviour with ratios so it also works on small datasets.
"""
from __future__ import annotations

import random
from collections.abc import Sequence


def stratified_split(
    labels: Sequence[int | str],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[int]]:
    """Return {"train": [...], "valid": [...], "test": [...]} of sample indices.

    Each class is shuffled independently and cut by `ratios`, so every split
    keeps the original class balance. Splits are disjoint and cover all samples.
    Supports both integer and string labels.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")
    rng = random.Random(seed)
    by_class: dict[int | str, list[int]] = {}
    for idx, y in enumerate(labels):
        by_class.setdefault(y, []).append(idx)

    out: dict[str, list[int]] = {"train": [], "valid": [], "test": []}
    for y in sorted(by_class, key=lambda x: str(x)):
        idx = by_class[y][:]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = round(n * ratios[0])
        n_va = round(n * ratios[1])
        out["train"] += idx[:n_tr]
        out["valid"] += idx[n_tr : n_tr + n_va]
        out["test"] += idx[n_tr + n_va :]
    for idx_list in out.values():
        rng.shuffle(idx_list)
    return out


