"""Utilities for inspecting action dataset manifests."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SPLITS = ("train", "val", "test")
REQUIRED_COLUMNS = {"clip_id", "video", "player_id", "label", "split"}


def summarize_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Return split, label, video, and player counts for an action manifest."""
    path = Path(manifest_path)
    rows = _read_manifest_rows(path)

    summary: dict[str, Any] = {
        "manifest": str(path),
        "total_clips": len(rows),
        "total_videos": len({row["video"] for row in rows}),
        "splits": {},
        "labels": _value_counts(row["label"] for row in rows),
        "players": _value_counts(row["player_id"] for row in rows),
        "videos": {},
        "group_split": _is_group_split(rows),
        "warnings": [],
    }

    for split in SPLITS:
        split_rows = [row for row in rows if row["split"] == split]
        summary["splits"][split] = _split_summary(split_rows)

    rows_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_video[row["video"]].append(row)
    for video in sorted(rows_by_video):
        video_rows = rows_by_video[video]
        summary["videos"][video] = {
            "clips": len(video_rows),
            "splits": _value_counts(row["split"] for row in video_rows),
            "labels": _value_counts(row["label"] for row in video_rows),
            "players": _value_counts(row["player_id"] for row in video_rows),
        }

    summary["warnings"] = manifest_warnings(summary)
    return summary


def manifest_warnings(summary: dict[str, Any]) -> list[str]:
    """Return human-readable risk warnings for a manifest summary."""
    warnings: list[str] = []
    for split, stats in summary.get("splits", {}).items():
        clips = int(stats.get("clips", 0))
        videos = int(stats.get("videos", 0))
        labels = stats.get("labels", {})
        split_step = int(labels.get("1", 0))
        normal = int(labels.get("0", 0))
        if clips == 0:
            warnings.append(f"{split} split has no clips.")
            continue
        if split_step == 0:
            warnings.append(f"{split} split has no split-step clips (label 1).")
        if normal == 0:
            warnings.append(f"{split} split has no normal clips (label 0).")
        minority = min(split_step, normal)
        majority = max(split_step, normal)
        if minority > 0 and majority / minority >= 8:
            warnings.append(
                f"{split} split is highly imbalanced: label0={normal}, label1={split_step}."
            )
        if split in {"val", "test"} and videos == 1:
            warnings.append(f"{split} split contains only one video.")
    return warnings


def _read_manifest_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Manifest {path} missing required columns: {missing}")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            rows.append(
                {
                    "clip_id": str(raw["clip_id"]),
                    "video": str(raw["video"]),
                    "player_id": int(raw["player_id"]),
                    "label": int(raw["label"]),
                    "split": str(raw["split"]),
                }
            )
    return rows


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "clips": len(rows),
        "videos": len({row["video"] for row in rows}),
        "labels": _value_counts(row["label"] for row in rows),
        "players": _value_counts(row["player_id"] for row in rows),
    }


def _value_counts(values: Iterable[Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(values).items())}


def _is_group_split(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    splits_by_video: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        splits_by_video[row["video"]].add(row["split"])
    return all(len(splits) <= 1 for splits in splits_by_video.values())
