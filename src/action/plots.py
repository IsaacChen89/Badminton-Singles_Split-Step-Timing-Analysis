"""Training result plots for the split-step action model."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from ..utils.logging import get_logger

logger = get_logger("action.plots")

History = Sequence[Mapping[str, Any]]
TestMetrics = Mapping[str, Any]

_CLASS_LABELS = {"0": "normal", "1": "split_step"}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_training_plots(
    history: History,
    output_dir: str | Path,
    *,
    test_metrics: Optional[TestMetrics] = None,
    test_metrics_path: Optional[str | Path] = None,
) -> List[Path]:
    """Write PNG training summaries under ``output_dir``.

    Produces:

    - ``training_curves.png`` — loss, accuracy, val F1, learning rate
    - ``test_class_metrics.png`` — clip/event precision/recall/F1 (when test
      metrics are available)

    Returns paths to the PNG files that were written.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    if not history:
        logger.warning("Empty training history; skipping plot generation.")
        return written

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    val_loss = [_safe_float(row.get("val_loss")) for row in history]
    train_acc = [float(row["train_acc"]) for row in history]
    val_acc = [_safe_float(row.get("val_acc")) for row in history]
    val_f1 = [_safe_float(row.get("val_f1")) for row in history]
    val_split_f1 = [_safe_float(row.get("val_split_step_f1")) for row in history]
    val_event_f1 = [_safe_float(row.get("val_event_f1")) for row in history]
    lr_series = _learning_rate_series(history)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), tight_layout=True)
    fig.suptitle("Split-step model training", fontsize=13, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="train", linewidth=2)
    if _any_finite(val_loss):
        ax.plot(epochs, val_loss, label="val", linewidth=2)
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, train_acc, label="train", linewidth=2)
    if _any_finite(val_acc):
        ax.plot(epochs, val_acc, label="val", linewidth=2)
    ax.set_title("Accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if _any_finite(val_f1):
        ax.plot(epochs, val_f1, color="#2ca02c", linewidth=2, label="macro")
        best_idx = int(np.nanargmax(val_f1))
        ax.scatter(
            [epochs[best_idx]],
            [val_f1[best_idx]],
            color="#d62728",
            zorder=3,
            label=f"best macro (epoch {epochs[best_idx]})",
        )
    if _any_finite(val_split_f1):
        ax.plot(epochs, val_split_f1, linewidth=2, label="split-step clip")
    if _any_finite(val_event_f1):
        ax.plot(epochs, val_event_f1, linewidth=2, label="split-step event")
    if _any_finite(val_f1 + val_split_f1 + val_event_f1):
        ax.legend()
    ax.set_title("Validation F1")
    ax.set_xlabel("epoch")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if lr_series:
        for name, values in lr_series.items():
            if _any_finite(values):
                ax.plot(epochs, values, linewidth=2, label=name)
        ax.legend()
    ax.set_title("Learning rate")
    ax.set_xlabel("epoch")
    ax.grid(True, alpha=0.3)

    curves_path = out_dir / "training_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    written.append(curves_path)
    logger.info(f"Wrote training curves -> {curves_path}")

    metrics = test_metrics
    if metrics is None and test_metrics_path is not None:
        p = Path(test_metrics_path)
        if p.exists():
            metrics = _load_json(p)

    if metrics and isinstance(metrics.get("report"), dict):
        class_path = out_dir / "test_class_metrics.png"
        _save_test_class_plot(metrics, class_path)
        written.append(class_path)
        logger.info(f"Wrote test class metrics -> {class_path}")

    return written


def save_training_plots_from_files(
    history_path: str | Path,
    output_dir: Optional[str | Path] = None,
    *,
    test_metrics_path: Optional[str | Path] = None,
) -> List[Path]:
    """Regenerate PNGs from ``train_history.json`` (and optional test JSON)."""
    history_path = Path(history_path)
    if not history_path.exists():
        raise FileNotFoundError(history_path)
    out_dir = Path(output_dir) if output_dir is not None else history_path.parent
    if test_metrics_path is None:
        candidate = out_dir / "test_metrics.json"
        test_metrics_path = candidate if candidate.exists() else None
    history = _load_json(history_path)
    return save_training_plots(
        history,
        out_dir,
        test_metrics_path=test_metrics_path,
    )


def _save_test_class_plot(metrics: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    report = metrics.get("report", {})
    if not isinstance(report, dict):
        return

    labels: List[str] = []
    rows: List[List[float]] = []
    for class_id in ("0", "1"):
        if class_id not in report:
            continue
        labels.append(_CLASS_LABELS.get(class_id, class_id))
        rows.append(
            [
                float(report[class_id]["precision"]),
                float(report[class_id]["recall"]),
                float(report[class_id]["f1-score"]),
            ]
        )

    # Clip-level "split_step" is clearer once event metrics are shown.
    labels = [
        "split_step_clip" if label == "split_step" else label for label in labels
    ]

    event = metrics.get("event")
    if isinstance(event, Mapping) and event:
        labels.append("split_step_event")
        rows.append(
            [
                float(event.get("precision", float("nan"))),
                float(event.get("recall", float("nan"))),
                float(event.get("f1", event.get("f1-score", float("nan")))),
            ]
        )

    if not labels:
        return

    values = np.asarray(rows, dtype=float)
    metric_names = ("precision", "recall", "f1-score")
    x = np.arange(len(labels))
    width = 0.25
    fig_width = 7.0 if len(labels) <= 2 else 8.5
    fig, ax = plt.subplots(figsize=(fig_width, 4), tight_layout=True)
    for i, metric in enumerate(metric_names):
        ax.bar(x + (i - 1) * width, values[:, i], width=width, label=metric)

    ax.set_title("Held-out test split — clip and event metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _learning_rate_series(history: History) -> dict[str, list[float]]:
    has_backbone = any("backbone_lr" in row for row in history)
    has_head = any("head_lr" in row for row in history)
    if has_backbone and has_head:
        return {
            "backbone": [_safe_float(row.get("backbone_lr")) for row in history],
            "head": [_safe_float(row.get("head_lr")) for row in history],
        }
    if any("lr" in row for row in history):
        return {"lr": [_safe_float(row.get("lr")) for row in history]}
    return {}


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _any_finite(values: Sequence[float]) -> bool:
    return any(math.isfinite(v) for v in values)
