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
    best_threshold: Optional[float] = None
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
    threshold: float = 0.5

    def to_dict(self) -> dict:
        return {
            "loss": self.loss,
            "acc": self.acc,
            "f1": self.f1,
            "threshold": self.threshold,
            "report": self.report,
        }


@dataclass
class EvalOutputs:
    loss: float
    probs: np.ndarray
    labels: np.ndarray


@torch.inference_mode()
def collect_eval_outputs(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_bce: bool = False,
) -> EvalOutputs:
    """Forward once and collect per-clip probabilities plus hard labels."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    all_probs: list[float] = []
    all_labels: list[int] = []
    for clips, targets, labels in loader:
        clips = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(clips)
        if use_bce:
            binary_logits = _binary_logits(logits)
            loss = criterion(binary_logits, targets)
            probs = torch.sigmoid(binary_logits)
        else:
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1)[:, 1]
        total_loss += loss.item() * clips.size(0)
        total_n += clips.size(0)
        all_probs.extend(probs.detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    mean_loss = total_loss / max(1, total_n)
    if total_n == 0:
        return EvalOutputs(
            loss=float("nan"),
            probs=np.array([], dtype=np.float32),
            labels=np.array([], dtype=np.int64),
        )
    return EvalOutputs(
        loss=mean_loss,
        probs=np.asarray(all_probs, dtype=np.float32),
        labels=np.asarray(all_labels, dtype=np.int64),
    )


def _metrics_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    loss: float = float("nan"),
) -> EvalMetrics:
    if labels.size == 0:
        return EvalMetrics(loss=loss, acc=float("nan"), f1=float("nan"), threshold=threshold)
    preds = (probs >= threshold).astype(np.int64)
    acc = float(np.mean(preds == labels))
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return EvalMetrics(loss=loss, acc=acc, f1=float(f1), report=report, threshold=threshold)


def _threshold_grid(train_cfg: TrainActionConfig) -> np.ndarray:
    low = float(train_cfg.threshold_sweep_min)
    high = float(train_cfg.threshold_sweep_max)
    step = float(train_cfg.threshold_sweep_step)
    if step <= 0:
        raise ValueError("train_action.threshold_sweep_step must be > 0.")
    if low > high:
        raise ValueError("train_action.threshold_sweep_min must be <= threshold_sweep_max.")
    count = int(round((high - low) / step)) + 1
    grid = low + step * np.arange(count, dtype=np.float64)
    return np.clip(grid, 0.0, 1.0)


def _sweep_threshold(
    outputs: EvalOutputs,
    train_cfg: TrainActionConfig,
    best_metric_name: str,
) -> EvalMetrics:
    grid = _threshold_grid(train_cfg)
    best_metrics = _metrics_from_probs(
        outputs.probs,
        outputs.labels,
        threshold=float(grid[0]),
        loss=outputs.loss,
    )
    best_score = _select_metric(best_metrics, best_metric_name)
    for threshold in grid[1:]:
        metrics = _metrics_from_probs(
            outputs.probs,
            outputs.labels,
            threshold=float(threshold),
            loss=outputs.loss,
        )
        score = _select_metric(metrics, best_metric_name)
        if score > best_score:
            best_score = score
            best_metrics = metrics
    return best_metrics


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_bce: bool = False,
    threshold: float = 0.5,
) -> EvalMetrics:
    """Run a single forward pass over ``loader`` and compute eval metrics."""
    outputs = collect_eval_outputs(model, loader, criterion, device, use_bce=use_bce)
    if outputs.labels.size == 0:
        return EvalMetrics(loss=float("nan"), acc=float("nan"), f1=float("nan"))
    return _metrics_from_probs(outputs.probs, outputs.labels, threshold, loss=outputs.loss)


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


def _binary_pos_weight(labels: np.ndarray, max_weight: float = 4.0) -> torch.Tensor:
    counts = np.bincount(labels.astype(np.int64), minlength=2).astype(np.float64)
    positives = max(1.0, counts[1])
    negatives = max(1.0, counts[0])
    weight = min(float(max_weight), negatives / positives)
    return torch.tensor([weight], dtype=torch.float32)


def _binary_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError(f"Expected logits with shape (B,C); got {tuple(logits.shape)}")
    if logits.size(1) == 1:
        return logits[:, 0]
    if logits.size(1) == 2:
        return logits[:, 1] - logits[:, 0]
    raise ValueError(
        "BCEWithLogitsLoss requires action.num_classes to be 1, or 2 for "
        "backward-compatible two-logit binary training."
    )


def _uses_bce_loss(train_cfg: TrainActionConfig) -> bool:
    return train_cfg.loss.strip().lower() in {"bce", "bce_with_logits", "binary_bce"}


def _class_report_metric(report: dict, class_id: str, metric: str) -> float:
    value = report.get(class_id, {}).get(metric, float("nan"))
    return float(value) if value is not None else float("nan")


def _normalize_metric_name(metric: str) -> str:
    return metric.strip().lower()


def _metric_higher_is_better(metric_name: str) -> bool:
    return _normalize_metric_name(metric_name) not in {"val_loss", "loss"}


def _select_metric(metrics: EvalMetrics, best_metric: str) -> float:
    normalized = _normalize_metric_name(best_metric)
    if normalized in {"val_loss", "loss"}:
        return metrics.loss
    if normalized in {"macro_f1", "val_f1", "f1"}:
        return metrics.f1
    if normalized in {"acc", "accuracy", "val_acc"}:
        return metrics.acc
    if normalized in {"split_step_f1", "class1_f1"}:
        return _class_report_metric(metrics.report, "1", "f1-score")
    raise ValueError(
        "Unsupported train_action metric "
        f"'{best_metric}'. Use val_loss, macro_f1, accuracy, or split_step_f1."
    )


def _metric_improved(current: float, best: float, metric_name: str) -> bool:
    if math.isnan(current):
        return False
    if math.isinf(best) and best < 0:
        return True
    if math.isnan(best):
        return True
    if _metric_higher_is_better(metric_name):
        return current > best
    return current < best


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
    use_bce = _uses_bce_loss(train_cfg)
    if use_bce and action_cfg.num_classes not in {1, 2}:
        raise ValueError("BCE action training requires action.num_classes to be 1 or 2.")
    if not use_bce and action_cfg.num_classes < 2:
        raise ValueError("Cross-entropy action training requires action.num_classes >= 2.")

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
    if use_bce:
        if train_cfg.class_weight_balance:
            pos_weight = _binary_pos_weight(
                train_ds.labels,
                max_weight=train_cfg.max_pos_weight,
            ).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            logger.info(
                f"BCE positive class weight: {pos_weight.item():.4f} "
                f"(cap {train_cfg.max_pos_weight:.1f})"
            )
        else:
            criterion = nn.BCEWithLogitsLoss()
        logger.info("Loss: BCEWithLogitsLoss")
    elif train_cfg.class_weight_balance:
        weights = _class_weights(train_ds.labels, action_cfg.num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        logger.info(f"Class weights: {weights.tolist()}")
        if label_smoothing > 0:
            logger.info(f"Label smoothing: {label_smoothing:.3f}")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if label_smoothing > 0:
            logger.info(f"Label smoothing: {label_smoothing:.3f}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found.")
    optimizer = torch.optim.AdamW(
        trainable, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    min_lr = max(0.0, float(train_cfg.lr) * float(train_cfg.min_lr_ratio))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg.epochs, eta_min=min_lr
    )

    use_amp = bool(train_cfg.amp and device_supports_amp(device))
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    logger.info(f"Training on device={device}  amp={use_amp}")

    out_dir = Path(train_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "action_best.pt"
    last_path = out_dir / "action_last.pt"
    history: list[dict] = []
    best_metric_name = _normalize_metric_name(train_cfg.best_metric)
    stop_metric_name = _normalize_metric_name(train_cfg.early_stopping_metric)
    best_score = math.inf if not _metric_higher_is_better(best_metric_name) else -math.inf
    best_stop_score = (
        math.inf if not _metric_higher_is_better(stop_metric_name) else -math.inf
    )
    best_f1 = -math.inf
    best_acc = 0.0
    best_epoch: Optional[int] = None
    best_threshold = float(train_cfg.classification_threshold)
    fixed_threshold = float(train_cfg.classification_threshold)
    epochs_without_improvement = 0
    grad_clip_norm = max(0.0, float(train_cfg.grad_clip_norm))
    logger.info(
        "Validation during training uses fixed threshold "
        f"{fixed_threshold:.2f}; threshold sweep runs once on val after training."
    )
    logger.info(
        f"Checkpoint metric: {best_metric_name}  |  "
        f"Early stopping metric: {stop_metric_name}  |  "
        f"patience={train_cfg.early_stopping_patience}"
    )

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        running_correct = 0
        for clips, targets, labels in train_loader:
            clips = clips.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                assert scaler is not None
                with torch.amp.autocast("cuda"):
                    logits = model(clips)
                    if use_bce:
                        loss = criterion(_binary_logits(logits), targets)
                    else:
                        loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                if grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(clips)
                if use_bce:
                    loss = criterion(_binary_logits(logits), targets)
                else:
                    loss = criterion(logits, labels)
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
            with torch.no_grad():
                if use_bce:
                    preds = (torch.sigmoid(_binary_logits(logits)) >= 0.5).long()
                else:
                    preds = logits.argmax(dim=-1)
                running_correct += (preds == labels).sum().item()
            running_loss += loss.item() * clips.size(0)
            running_n += clips.size(0)
        train_loss = running_loss / max(1, running_n)
        train_acc = running_correct / max(1, running_n)
        scheduler.step()

        val_loss = float("nan")
        val_acc = float("nan")
        val_f1 = float("nan")
        val_split_precision = float("nan")
        val_split_recall = float("nan")
        val_split_f1 = float("nan")
        val_split_support = 0.0
        val_score = float("nan")
        stop_score = float("nan")
        val_threshold = fixed_threshold
        if val_loader is not None:
            val_metrics = evaluate_model(
                model,
                val_loader,
                criterion,
                device,
                use_bce=use_bce,
                threshold=fixed_threshold,
            )
            val_loss = val_metrics.loss
            val_acc = val_metrics.acc
            val_f1 = val_metrics.f1
            val_threshold = val_metrics.threshold
            val_split_precision = _class_report_metric(val_metrics.report, "1", "precision")
            val_split_recall = _class_report_metric(val_metrics.report, "1", "recall")
            val_split_f1 = _class_report_metric(val_metrics.report, "1", "f1-score")
            val_split_support = _class_report_metric(val_metrics.report, "1", "support")
            val_score = _select_metric(val_metrics, best_metric_name)
            stop_score = _select_metric(val_metrics, stop_metric_name)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_threshold": val_threshold,
            "val_split_step_precision": val_split_precision,
            "val_split_step_recall": val_split_recall,
            "val_split_step_f1": val_split_f1,
            "val_split_step_support": val_split_support,
            "val_best_metric": best_metric_name,
            "val_best_metric_value": val_score,
            "val_stop_metric": stop_metric_name,
            "val_stop_metric_value": stop_score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_log)
        logger.info(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
            f"| val_loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f} "
            f"| split_f1 {val_split_f1:.3f}"
        )

        save_checkpoint(model, last_path, extra={"epoch": epoch})
        checkpoint_improved = _metric_improved(val_score, best_score, best_metric_name)
        if checkpoint_improved:
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
                    "val_loss": val_loss,
                    "best_metric": best_metric_name,
                    "best_metric_value": val_score,
                    "classification_threshold": fixed_threshold,
                },
            )
            logger.info(
                f"  -> new best {best_metric_name} {best_score:.3f} "
                f"saved to {best_path}"
            )

        stop_improved = _metric_improved(stop_score, best_stop_score, stop_metric_name)
        if stop_improved:
            best_stop_score = stop_score
            epochs_without_improvement = 0
        elif val_loader is not None:
            epochs_without_improvement += 1
            patience = max(0, int(train_cfg.early_stopping_patience))
            if patience and epochs_without_improvement >= patience:
                logger.info(
                    f"Early stopping after {epoch} epochs: no {stop_metric_name} "
                    f"improvement for {patience} epoch(s)."
                )
                break

    if best_epoch is None:
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

    best_model: nn.Module = model
    if best_path.exists():
        try:
            best_model = load_checkpoint(best_path, map_location=device).to(device)
        except Exception as exc:
            logger.warning(
                f"Could not reload best checkpoint ({exc}); "
                f"falling back to in-memory model."
            )

    if val_loader is not None:
        logger.info(
            "Sweeping split-step threshold on validation: "
            f"{train_cfg.threshold_sweep_min:.2f}..{train_cfg.threshold_sweep_max:.2f} "
            f"step {train_cfg.threshold_sweep_step:.2f}"
        )
        val_outputs = collect_eval_outputs(
            best_model, val_loader, criterion, device, use_bce=use_bce
        )
        if val_outputs.labels.size > 0:
            swept_val = _sweep_threshold(val_outputs, train_cfg, best_metric_name)
            best_threshold = swept_val.threshold
            swept_split_f1 = _class_report_metric(swept_val.report, "1", "f1-score")
            logger.info(
                f"Post-train val sweep | {best_metric_name} {swept_split_f1:.3f} "
                f"| thr {best_threshold:.2f}"
            )
            save_checkpoint(
                best_model,
                best_path,
                extra={
                    "epoch": best_epoch,
                    "val_f1": best_f1,
                    "val_acc": best_acc,
                    "best_metric": best_metric_name,
                    "best_metric_value": best_score,
                    "classification_threshold": best_threshold,
                },
            )
        else:
            logger.warning("Empty validation split; keeping fixed classification threshold.")

    test_acc: Optional[float] = None
    test_f1: Optional[float] = None
    test_loss: Optional[float] = None
    test_report: Optional[dict] = None
    if test_loader is not None:
        logger.info(f"Evaluating best checkpoint on test split ({len(test_ds)} clips)...")
        test_metrics = evaluate_model(
            best_model,
            test_loader,
            criterion,
            device,
            use_bce=use_bce,
            threshold=best_threshold,
        )
        test_loss = test_metrics.loss
        test_acc = test_metrics.acc
        test_f1 = test_metrics.f1
        test_report = test_metrics.report
        test_split_f1 = _class_report_metric(test_metrics.report, "1", "f1-score")
        logger.info(
            f"Test  | loss {test_loss:.4f}  acc {test_acc:.3f}  f1 {test_f1:.3f} "
            f"| split_f1 {test_split_f1:.3f}  thr {best_threshold:.2f}"
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
        best_threshold=best_threshold,
        test_acc=test_acc,
        test_f1=test_f1,
        test_loss=test_loss,
        test_report=test_report,
    )
