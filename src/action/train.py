"""Training loop for the split-step CNN-LSTM."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from ..utils.config import ActionConfig, TrainActionConfig, device_supports_amp, device_supports_pin_memory
from ..utils.logging import get_logger
from .dataset import (
    SplitStepClipDataset,
    build_eval_transform,
    build_train_transform,
)
from .manifest import summarize_manifest
from .model import build_model, load_checkpoint, save_checkpoint
from .plots import save_training_plots

logger = get_logger("action.train")


@dataclass
class TrainResult:
    best_val_f1: float
    best_val_acc: float
    checkpoint_path: str
    history: list[dict]
    best_epoch: Optional[int] = None
    best_metric: str = "macro_f1"
    best_metric_value: Optional[float] = None
    test_acc: Optional[float] = None
    test_f1: Optional[float] = None
    test_loss: Optional[float] = None
    test_report: Optional[dict] = None


@dataclass
class EvalMetrics:
    loss: float
    acc: float
    f1: float
    report: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "loss": self.loss,
            "acc": self.acc,
            "f1": self.f1,
            "report": self.report,
        }


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> EvalMetrics:
    """Run a single forward pass over ``loader`` and compute eval metrics.

    Returns mean loss, accuracy, macro-F1, and a per-class
    ``classification_report`` dict (keyed by class label as a string).
    """
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for clips, labels in loader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(clips)
        loss = criterion(logits, labels)
        total_loss += loss.item() * clips.size(0)
        total_n += clips.size(0)
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    mean_loss = total_loss / max(1, total_n)
    if total_n == 0:
        return EvalMetrics(loss=float("nan"), acc=float("nan"), f1=float("nan"))
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels, all_preds, output_dict=True, zero_division=0
    )
    return EvalMetrics(loss=mean_loss, acc=acc, f1=float(f1), report=report)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _class_report_metric(report: dict, class_id: str, metric: str) -> float:
    value = report.get(class_id, {}).get(metric, float("nan"))
    return float(value) if value is not None else float("nan")


def _select_metric(metrics: EvalMetrics, best_metric: str) -> float:
    normalized = best_metric.strip().lower()
    if normalized in {"macro_f1", "val_f1", "f1"}:
        return metrics.f1
    if normalized in {"acc", "accuracy", "val_acc"}:
        return metrics.acc
    if normalized in {"split_step_f1", "class1_f1"}:
        return _class_report_metric(metrics.report, "1", "f1-score")
    if normalized in {"split_step_recall", "class1_recall"}:
        return _class_report_metric(metrics.report, "1", "recall")
    raise ValueError(
        "Unsupported train_action.best_metric "
        f"'{best_metric}'. Use macro_f1, accuracy, split_step_f1, or split_step_recall."
    )


def train(
    manifest: str | Path,
    action_cfg: ActionConfig,
    train_cfg: TrainActionConfig,
    device: str = "cpu",
    seed: int = 42,
    pretrained_imagenet: bool = True,
) -> TrainResult:
    """Train the split-step CNN-LSTM and save the best checkpoint."""
    _seed_everything(seed)
    manifest = Path(manifest)
    if not manifest.exists():
        raise FileNotFoundError(manifest)

    train_ds = SplitStepClipDataset(
        manifest_path=manifest,
        split="train",
        clip_length=action_cfg.clip_length,
        transform=build_train_transform(action_cfg.input_size),
    )
    val_ds = SplitStepClipDataset(
        manifest_path=manifest,
        split="val",
        clip_length=action_cfg.clip_length,
        transform=build_eval_transform(action_cfg.input_size),
    )
    test_ds = SplitStepClipDataset(
        manifest_path=manifest,
        split="test",
        clip_length=action_cfg.clip_length,
        transform=build_eval_transform(action_cfg.input_size),
    )
    if len(train_ds) == 0:
        raise RuntimeError("Empty train split in manifest.")
    logger.info(
        f"Train clips: {len(train_ds)}  |  Val clips: {len(val_ds)}  "
        f"|  Test clips: {len(test_ds)}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=train_cfg.num_workers,
        pin_memory=device_supports_pin_memory(device),
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=device_supports_pin_memory(device),
        )
        if len(val_ds) > 0
        else None
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=train_cfg.num_workers,
            pin_memory=device_supports_pin_memory(device),
        )
        if len(test_ds) > 0
        else None
    )

    model = build_model(action_cfg).to(device)
    if pretrained_imagenet:
        model.try_load_imagenet_weights()

    label_smoothing = max(0.0, min(1.0, float(train_cfg.label_smoothing)))
    if train_cfg.class_weight_balance:
        weights = _class_weights(train_ds.labels, action_cfg.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        logger.info(f"Class weights: {weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        logger.info(f"Label smoothing: {label_smoothing:.3f}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.epochs * steps_per_epoch
    )

    use_amp = bool(train_cfg.amp and device_supports_amp(device))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    logger.info(f"Training on device={device}  amp={use_amp}")

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "action_best.pt"
    last_path = out_dir / "action_last.pt"
    split_summary = summarize_manifest(manifest)
    (out_dir / "split_summary.json").write_text(json.dumps(split_summary, indent=2))
    for warning in split_summary.get("warnings", []):
        logger.warning(f"Manifest warning: {warning}")

    history: list[dict] = []
    best_metric_name = train_cfg.best_metric.strip().lower()
    best_score = -math.inf
    best_f1 = -math.inf
    best_acc = 0.0
    best_epoch: Optional[int] = None
    epochs_without_improvement = 0

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        running_correct = 0
        for clips, labels in train_loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                assert scaler is not None
                with torch.amp.autocast("cuda"):
                    logits = model(clips)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(clips)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            scheduler.step()
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                running_correct += (preds == labels).sum().item()
            running_loss += loss.item() * clips.size(0)
            running_n += clips.size(0)
        train_loss = running_loss / max(1, running_n)
        train_acc = running_correct / max(1, running_n)

        val_loss = float("nan")
        val_acc = float("nan")
        val_f1 = float("nan")
        val_split_precision = float("nan")
        val_split_recall = float("nan")
        val_split_f1 = float("nan")
        val_split_support = 0.0
        val_score = float("nan")
        if val_loader is not None:
            val_metrics = evaluate_model(model, val_loader, criterion, device)
            val_loss = val_metrics.loss
            val_acc = val_metrics.acc
            val_f1 = val_metrics.f1
            val_split_precision = _class_report_metric(val_metrics.report, "1", "precision")
            val_split_recall = _class_report_metric(val_metrics.report, "1", "recall")
            val_split_f1 = _class_report_metric(val_metrics.report, "1", "f1-score")
            val_split_support = _class_report_metric(val_metrics.report, "1", "support")
            val_score = _select_metric(val_metrics, best_metric_name)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_split_step_precision": val_split_precision,
            "val_split_step_recall": val_split_recall,
            "val_split_step_f1": val_split_f1,
            "val_split_step_support": val_split_support,
            "val_best_metric": best_metric_name,
            "val_best_metric_value": val_score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_log)
        logger.info(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
            f"| val_loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f} "
            f"| split_f1 {val_split_f1:.3f} split_recall {val_split_recall:.3f}"
        )

        save_checkpoint(model, last_path, extra={"epoch": epoch})
        improved = not math.isnan(val_score) and val_score > best_score
        if improved:
            best_score = val_score
            best_f1 = val_f1
            best_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                model,
                best_path,
                extra={
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "best_metric": best_metric_name,
                    "best_metric_value": val_score,
                },
            )
            epochs_without_improvement = 0
            logger.info(
                f"  -> new best {best_metric_name} {best_score:.3f} saved to {best_path}"
            )
        elif val_loader is not None:
            epochs_without_improvement += 1
            patience = max(0, int(train_cfg.early_stopping_patience))
            if patience and epochs_without_improvement >= patience:
                logger.info(
                    f"Early stopping after {epoch} epochs: no {best_metric_name} "
                    f"improvement for {patience} epoch(s)."
                )
                break

    if math.isinf(best_score):
        # No validation set — keep "last" as best.
        save_checkpoint(model, best_path, extra={"epoch": train_cfg.epochs})
        best_f1 = float("nan")
        best_acc = float("nan")
        best_score = float("nan")

    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    logger.info(
        f"Training done. Best {best_metric_name}: {best_score:.3f}  "
        f"| best F1: {best_f1:.3f}  | best ckpt: {best_path}"
    )

    test_acc: Optional[float] = None
    test_f1: Optional[float] = None
    test_loss: Optional[float] = None
    test_report: Optional[dict] = None
    if test_loader is not None:
        logger.info(f"Evaluating best checkpoint on test split ({len(test_ds)} clips)...")
        try:
            best_model = load_checkpoint(best_path, map_location=device).to(device)
        except Exception as exc:
            logger.warning(
                f"Could not reload best checkpoint for test eval ({exc}); "
                f"falling back to in-memory model."
            )
            best_model = model
        test_metrics = evaluate_model(best_model, test_loader, criterion, device)
        test_loss = test_metrics.loss
        test_acc = test_metrics.acc
        test_f1 = test_metrics.f1
        test_report = test_metrics.report
        logger.info(
            f"Test  | loss {test_loss:.4f}  acc {test_acc:.3f}  f1 {test_f1:.3f}"
        )
        (out_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics.to_dict(), indent=2)
        )
    else:
        logger.info("No test split in manifest — skipping post-training test eval.")

    plot_paths = save_training_plots(
        history,
        out_dir,
        test_metrics_path=out_dir / "test_metrics.json",
    )
    if plot_paths:
        logger.info(f"Training plots: {', '.join(str(p) for p in plot_paths)}")

    return TrainResult(
        best_val_f1=best_f1,
        best_val_acc=best_acc,
        checkpoint_path=str(best_path),
        history=history,
        best_epoch=best_epoch,
        best_metric=best_metric_name,
        best_metric_value=None if math.isnan(best_score) else best_score,
        test_acc=test_acc,
        test_f1=test_f1,
        test_loss=test_loss,
        test_report=test_report,
    )
