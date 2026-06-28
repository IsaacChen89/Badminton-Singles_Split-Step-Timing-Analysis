"""Dataset split helpers that do not depend on video/native libraries."""

from __future__ import annotations

import random
from typing import Dict, Sequence, Tuple


def split_train_val_test(
    items: Sequence[str | int],
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> Tuple[set, set, set]:
    """Three-way deterministic random split over ``items``."""
    rng = random.Random(seed)
    indices = list(items)
    rng.shuffle(indices)
    n_train, n_val, n_test = split_sizes(len(indices), val_split, test_split)
    val = set(indices[:n_val])
    test = set(indices[n_val : n_val + n_test])
    train = set(indices[n_val + n_test : n_val + n_test + n_train])
    return train, val, test


def split_sizes(n: int, val_split: float, test_split: float) -> Tuple[int, int, int]:
    if val_split < 0 or test_split < 0:
        raise ValueError(
            f"val_split and test_split must be >= 0 "
            f"(got val={val_split}, test={test_split})."
        )
    if val_split + test_split >= 1.0:
        raise ValueError(
            f"val_split + test_split must be < 1.0 "
            f"(got val={val_split} + test={test_split} = {val_split + test_split})."
        )
    n_val = int(round(val_split * n))
    n_test = int(round(test_split * n))
    if n_val + n_test > n:
        n_test = max(0, n - n_val)
    n_train = max(0, n - n_val - n_test)
    return n_train, n_val, n_test


def assign_group_splits(
    groups: Sequence[str],
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign each group entirely to train, val, or test."""
    unique = sorted(set(groups))
    train, val, test = split_train_val_test(unique, val_split, test_split, seed=seed)
    mapping: Dict[str, str] = {}
    for stem in train:
        mapping[stem] = "train"
    for stem in val:
        mapping[stem] = "val"
    for stem in test:
        mapping[stem] = "test"
    return mapping


def assign_stratified_group_splits(
    group_label_counts: Dict[str, Dict[int, int]],
    val_split: float,
    test_split: float,
    seed: int = 42,
) -> Dict[str, str]:
    """Assign whole videos while prioritizing clip and class-label balance.

    Videos can have very different clip counts, so exact video-count quotas can
    produce poor train/val/test clip ratios. This splitter keeps each video
    whole, but treats video count as a soft constraint and optimizes mostly for
    total clips plus split-step (label 1) balance.
    """
    groups = sorted(group_label_counts)
    if not groups:
        return {}
    totals = {
        label: sum(counts.get(label, 0) for counts in group_label_counts.values())
        for label in (0, 1)
    }
    total_clips = sum(totals.values())
    target_shares = {
        "train": 1.0 - val_split - test_split,
        "val": val_split,
        "test": test_split,
    }
    target_counts = {
        split: {
            "groups": len(groups) * share,
            "clips": total_clips * share,
            0: totals[0] * share,
            1: totals[1] * share,
        }
        for split, share in target_shares.items()
    }

    rng = random.Random(seed)
    jitter = {group: rng.random() for group in groups}
    ordered = sorted(
        groups,
        key=lambda group: (
            -sum(group_label_counts[group].values()),
            -group_label_counts[group].get(1, 0),
            jitter[group],
        ),
    )
    assigned_counts = {
        split: {"groups": 0, "clips": 0, 0: 0, 1: 0}
        for split in ("train", "val", "test")
    }
    mapping: Dict[str, str] = {}

    for group in ordered:
        counts = group_label_counts[group]
        best_split = min(
            ("train", "val", "test"),
            key=lambda split: _stratified_group_cost(
                assigned_counts[split],
                target_counts[split],
                counts,
            ),
        )
        mapping[group] = best_split
        assigned_counts[best_split]["groups"] += 1
        assigned_counts[best_split]["clips"] += sum(counts.values())
        assigned_counts[best_split][0] += counts.get(0, 0)
        assigned_counts[best_split][1] += counts.get(1, 0)
    return mapping


def _stratified_group_cost(
    current: Dict[str | int, int],
    target: Dict[str | int, float],
    addition: Dict[int, int],
) -> float:
    next_groups = current["groups"] + 1
    next_clips = current["clips"] + sum(addition.values())
    next_label0 = current[0] + addition.get(0, 0)
    next_label1 = current[1] + addition.get(1, 0)

    clip_need = max(0.0, target["clips"] - current["clips"])
    label0_need = max(0.0, target[0] - current[0])
    label1_need = max(0.0, target[1] - current[1])
    group_need = max(0.0, target["groups"] - current["groups"])

    clip_over = max(0.0, next_clips - target["clips"])
    label0_over = max(0.0, next_label0 - target[0])
    label1_over = max(0.0, next_label1 - target[1])
    group_over = max(0.0, next_groups - target["groups"])

    # Return negative score because callers use min(). Clip and label-1 needs
    # dominate; group count only nudges ties so large videos do not starve train.
    need_score = (
        (2.0 * clip_need)
        + label0_need
        + (3.0 * label1_need)
        + (0.1 * group_need * max(1.0, target["clips"] / max(1.0, target["groups"])))
    )
    overshoot_penalty = (
        (4.0 * clip_over)
        + (2.0 * label0_over)
        + (6.0 * label1_over)
        + (0.2 * group_over * max(1.0, target["clips"] / max(1.0, target["groups"])))
    )
    return overshoot_penalty - need_score


def _relative_error(value: float, target: float) -> float:
    return abs(value - target) / max(1.0, target)
