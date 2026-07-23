"""Event-level metrics for temporal split-step detections."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dataset import ClipRecord


@dataclass(frozen=True)
class EventMetrics:
    precision: float
    recall: float
    f1: float
    matched_events: int
    predicted_events: int
    ground_truth_events: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "matched_events": self.matched_events,
            "predicted_events": self.predicted_events,
            "ground_truth_events": self.ground_truth_events,
        }


def event_detection_metrics(
    records: Sequence[ClipRecord],
    predictions: Sequence[int] | np.ndarray,
    *,
    event_gap_frames: int,
    tolerance_frames: int = 4,
) -> EventMetrics:
    """Match predicted temporal events to labeled events one-to-one."""
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have the same length.")
    if event_gap_frames <= 0:
        raise ValueError("event_gap_frames must be > 0.")
    if tolerance_frames < 0:
        raise ValueError("tolerance_frames must be >= 0.")

    grouped: dict[tuple[str, int], list[tuple[ClipRecord, int]]] = defaultdict(list)
    for record, prediction in zip(records, predictions):
        grouped[(record.video, record.player_id)].append((record, int(prediction)))

    matched = 0
    predicted_count = 0
    ground_truth_count = 0
    for samples in grouped.values():
        samples.sort(key=lambda sample: sample[0].center_frame)
        ground_truth = _ground_truth_intervals(samples)
        predicted = _predicted_intervals(samples, event_gap_frames)
        ground_truth_count += len(ground_truth)
        predicted_count += len(predicted)
        matched += _match_intervals(ground_truth, predicted, tolerance_frames)

    precision = matched / predicted_count if predicted_count else 0.0
    recall = matched / ground_truth_count if ground_truth_count else 0.0
    denominator = precision + recall
    f1 = 2.0 * precision * recall / denominator if denominator else 0.0
    return EventMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        matched_events=matched,
        predicted_events=predicted_count,
        ground_truth_events=ground_truth_count,
    )


def _ground_truth_intervals(
    samples: Sequence[tuple[ClipRecord, int]],
) -> list[tuple[int, int]]:
    by_event: dict[str, list[int]] = defaultdict(list)
    for record, _prediction in samples:
        if record.label == 1:
            if not record.event_id:
                raise ValueError("Positive clips require event_id for event metrics.")
            by_event[record.event_id].append(record.center_frame)
    return sorted(
        (min(frames), max(frames))
        for frames in by_event.values()
        if frames
    )


def _predicted_intervals(
    samples: Sequence[tuple[ClipRecord, int]],
    event_gap_frames: int,
) -> list[tuple[int, int]]:
    positive_frames = [
        record.center_frame for record, prediction in samples if prediction == 1
    ]
    if not positive_frames:
        return []
    intervals: list[tuple[int, int]] = []
    start = previous = positive_frames[0]
    for frame in positive_frames[1:]:
        if frame - previous > event_gap_frames:
            intervals.append((start, previous))
            start = frame
        previous = frame
    intervals.append((start, previous))
    return intervals


def _match_intervals(
    ground_truth: Sequence[tuple[int, int]],
    predicted: Sequence[tuple[int, int]],
    tolerance_frames: int,
) -> int:
    unmatched_predictions = set(range(len(predicted)))
    matched = 0
    for truth_start, truth_end in ground_truth:
        for index in sorted(unmatched_predictions):
            pred_start, pred_end = predicted[index]
            if (
                pred_start <= truth_end + tolerance_frames
                and truth_start <= pred_end + tolerance_frames
            ):
                unmatched_predictions.remove(index)
                matched += 1
                break
    return matched
