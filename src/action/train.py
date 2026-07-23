"""Training loop for the split-step CNN-LSTM."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from ..utils.config import ActionConfig, TrainActionConfig, device_supports_amp, device_supports_pin_memory
from ..utils.logging import get_logger
from .dataset import (
    ClipRecord,
    EventBalancedBatchSampler,
    SplitStepClipDataset,
    build_eval_transform,
    build_train_transform,
)
from .event_metrics import EventMetrics, event_detection_metrics
from .model import build_model, load_checkpoint, save_checkpoint
from .plots import save_training_plots

logger = get_logger("action.train")


def _ema_multi_avg_fn(
    configured_decay: float,
):
    """Return EMA update with a short bias-reducing warmup."""

    def average(
        averaged: list[torch.Tensor],
        current: list[torch.Tensor],
        num_averaged: torch.Tensor | int,
    ) -> None:
        updates = int(num_averaged)
        warmup_decay = (updates + 1.0) / (updates + 10.0)
        decay = min(configured_decay, warmup_decay)
        torch._foreach_lerp_(averaged, current, 1.0 - decay)

    return average


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
    best_split_step_f1: Optional[float] = None
    best_event_f1: Optional[float] = None
    test_acc: Optional[float] = None
    test_f1: Optional[float] = None
    test_split_step_f1: Optional[float] = None
    test_event_f1: Optional[float] = None
    test_loss: Optional[float] = None
    test_report: Optional[dict] = None
    resumed_from: Optional[str] = None
    ema_decay: Optional[float] = None
    temperature: Optional[float] = None
    calibration_loss_before: Optional[float] = None
    calibration_loss_after: Optional[float] = None


@dataclass
class EvalMetrics:
    loss: float
    acc: float
    f1: float
    report: dict = field(default_factory=dict)
    threshold: float = 0.5
    event: Optional[EventMetrics] = None

    def to_dict(self) -> dict:
        return {
            "loss": self.loss,
            "acc": self.acc,
            "f1": self.f1,
            "threshold": self.threshold,
            "report": self.report,
            "event": self.event.to_dict() if self.event is not None else None,
        }


@dataclass
class EvalOutputs:
    loss: float
    probs: np.ndarray
    labels: np.ndarray
    records: Sequence[ClipRecord] = field(default_factory=list)


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
    dataset_records = list(getattr(loader.dataset, "records", []))
    if dataset_records and len(dataset_records) != total_n:
        raise ValueError("Event metrics require evaluation in dataset record order.")
    if total_n == 0:
        return EvalOutputs(
            loss=float("nan"),
            probs=np.array([], dtype=np.float32),
            labels=np.array([], dtype=np.int64),
            records=[],
        )
    return EvalOutputs(
        loss=mean_loss,
        probs=np.asarray(all_probs, dtype=np.float32),
        labels=np.asarray(all_labels, dtype=np.int64),
        records=dataset_records,
    )


@torch.inference_mode()
def _collect_calibration_logits(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect uncalibrated logits and hard labels on CPU."""
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for clips, _targets, labels in loader:
        clips = clips.to(device, non_blocking=True)
        all_logits.append(model(clips).detach().cpu())
        all_labels.append(labels.detach().cpu())
    if not all_logits:
        return torch.empty((0, 1)), torch.empty(0, dtype=torch.long)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    *,
    use_bce: bool,
    max_steps: int = 200,
) -> tuple[float, float, float]:
    """Fit one positive temperature on hard validation labels."""
    set_temperature = getattr(model, "set_temperature", None)
    if not callable(set_temperature):
        raise TypeError("Temperature calibration requires model.set_temperature().")
    set_temperature(1.0)
    logits, labels = _collect_calibration_logits(model, loader, device)
    if labels.numel() == 0:
        return 1.0, float("nan"), float("nan")
    # Tensors created under inference_mode cannot participate in the scalar
    # temperature optimization graph; clone them into ordinary CPU tensors.
    logits = logits.clone()
    labels = labels.clone()

    if use_bce:
        calibration_logits = _binary_logits(logits)
        calibration_labels = labels.float()

        def calibration_loss(scaled_logits: torch.Tensor) -> torch.Tensor:
            return nn.functional.binary_cross_entropy_with_logits(
                scaled_logits,
                calibration_labels,
            )

    else:
        calibration_logits = logits
        calibration_labels = labels

        def calibration_loss(scaled_logits: torch.Tensor) -> torch.Tensor:
            return nn.functional.cross_entropy(scaled_logits, calibration_labels)

    with torch.no_grad():
        loss_before = float(calibration_loss(calibration_logits).item())

    log_temperature = nn.Parameter(torch.zeros((), dtype=calibration_logits.dtype))
    optimizer = torch.optim.Adam([log_temperature], lr=0.05)
    for _ in range(max(1, int(max_steps))):
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = calibration_loss(calibration_logits / temperature)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_temperature.clamp_(math.log(0.05), math.log(10.0))

    temperature_value = float(log_temperature.detach().exp().item())
    with torch.no_grad():
        loss_after = float(
            calibration_loss(calibration_logits / temperature_value).item()
        )
    set_temperature(temperature_value)
    return temperature_value, loss_before, loss_after


def _metrics_from_probs(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    loss: float = float("nan"),
    *,
    records: Sequence[ClipRecord] = (),
    event_gap_frames: int = 4,
    event_tolerance_frames: int = 4,
) -> EvalMetrics:
    if labels.size == 0:
        return EvalMetrics(loss=loss, acc=float("nan"), f1=float("nan"), threshold=threshold)
    preds = (probs >= threshold).astype(np.int64)
    acc = float(np.mean(preds == labels))
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    event = (
        event_detection_metrics(
            records,
            preds,
            event_gap_frames=event_gap_frames,
            tolerance_frames=event_tolerance_frames,
        )
        if records
        else None
    )
    return EvalMetrics(
        loss=loss,
        acc=acc,
        f1=float(f1),
        report=report,
        threshold=threshold,
        event=event,
    )


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


def _metric_uses_threshold(metric_name: str) -> bool:
    return _normalize_metric_name(metric_name) not in {"val_loss", "loss"}


def _sweep_threshold(
    outputs: EvalOutputs,
    train_cfg: TrainActionConfig,
    best_metric_name: str,
    *,
    event_gap_frames: int,
) -> EvalMetrics:
    grid = _threshold_grid(train_cfg)
    best_metrics = _metrics_from_probs(
        outputs.probs,
        outputs.labels,
        threshold=float(grid[0]),
        loss=outputs.loss,
        records=outputs.records,
        event_gap_frames=event_gap_frames,
        event_tolerance_frames=train_cfg.event_match_tolerance_frames,
    )
    best_score = _select_metric(best_metrics, best_metric_name)
    for threshold in grid[1:]:
        metrics = _metrics_from_probs(
            outputs.probs,
            outputs.labels,
            threshold=float(threshold),
            loss=outputs.loss,
            records=outputs.records,
            event_gap_frames=event_gap_frames,
            event_tolerance_frames=train_cfg.event_match_tolerance_frames,
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
    event_gap_frames: int = 4,
    event_tolerance_frames: int = 4,
) -> EvalMetrics:
    """Run a single forward pass over ``loader`` and compute eval metrics."""
    outputs = collect_eval_outputs(model, loader, criterion, device, use_bce=use_bce)
    if outputs.labels.size == 0:
        return EvalMetrics(loss=float("nan"), acc=float("nan"), f1=float("nan"))
    return _metrics_from_probs(
        outputs.probs,
        outputs.labels,
        threshold,
        loss=outputs.loss,
        records=outputs.records,
        event_gap_frames=event_gap_frames,
        event_tolerance_frames=event_tolerance_frames,
    )


@torch.inference_mode()
def evaluate_with_threshold_sweep(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    train_cfg: TrainActionConfig,
    use_bce: bool = False,
    sweep_metric_name: str = "split_step_f1",
    fallback_threshold: float = 0.5,
    event_gap_frames: int = 4,
) -> EvalMetrics:
    """Forward once, then pick the best classification threshold on val."""
    outputs = collect_eval_outputs(model, loader, criterion, device, use_bce=use_bce)
    if outputs.labels.size == 0:
        return EvalMetrics(loss=float("nan"), acc=float("nan"), f1=float("nan"))
    if _metric_uses_threshold(sweep_metric_name):
        return _sweep_threshold(
            outputs,
            train_cfg,
            sweep_metric_name,
            event_gap_frames=event_gap_frames,
        )
    return _metrics_from_probs(
        outputs.probs,
        outputs.labels,
        fallback_threshold,
        loss=outputs.loss,
        records=outputs.records,
        event_gap_frames=event_gap_frames,
        event_tolerance_frames=train_cfg.event_match_tolerance_frames,
    )


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
    if normalized in {"split_clip_f1", "split_step_f1", "class1_f1"}:
        return _class_report_metric(metrics.report, "1", "f1-score")
    if normalized in {
        "split_event_f1",
        "event_f1",
        "split_step_event_f1",
    }:
        return metrics.event.f1 if metrics.event is not None else float("nan")
    raise ValueError(
        "Unsupported train_action metric "
        f"'{best_metric}'. Use val_loss, macro_f1, accuracy, split_clip_f1, "
        "or split_event_f1."
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


def _effective_head_lr(train_cfg: TrainActionConfig) -> float:
    if train_cfg.head_lr is not None:
        return float(train_cfg.head_lr)
    return float(train_cfg.lr)


def _effective_backbone_lr(train_cfg: TrainActionConfig) -> float:
    if train_cfg.backbone_lr is not None:
        return float(train_cfg.backbone_lr)
    return float(train_cfg.lr)


def _uses_differential_lr(action_cfg: ActionConfig, train_cfg: TrainActionConfig) -> bool:
    if action_cfg.freeze_backbone:
        return False
    if train_cfg.backbone_lr is None and train_cfg.head_lr is None:
        return False
    return True


def _head_parameters(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in (getattr(model, "lstm", None), getattr(model, "head", None)):
        if module is None:
            continue
        params.extend(p for p in module.parameters() if p.requires_grad)
    return params


def _build_optimizer(
    model: nn.Module,
    action_cfg: ActionConfig,
    train_cfg: TrainActionConfig,
) -> torch.optim.AdamW:
    weight_decay = float(train_cfg.weight_decay)
    if _uses_differential_lr(action_cfg, train_cfg):
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = _head_parameters(model)
        if not backbone_params or not head_params:
            raise RuntimeError(
                "Differential LR requires both backbone and head to be trainable."
            )
        backbone_lr = _effective_backbone_lr(train_cfg)
        head_lr = _effective_head_lr(train_cfg)
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )
        logger.info(
            f"Optimizer: AdamW differential LR  "
            f"backbone={backbone_lr:g}  head={head_lr:g}  wd={weight_decay:g}"
        )
        return optimizer

    lr = _effective_head_lr(train_cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters found.")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    logger.info(f"Optimizer: AdamW lr={lr:g}  wd={weight_decay:g}")
    return optimizer


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainActionConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Cosine decay to ``min_lr_ratio`` of each group's initial LR.

    Uses ``LambdaLR`` so multi-group optimizers (differential LR) work on all
    PyTorch versions; ``CosineAnnealingLR`` only accepts a scalar ``eta_min``.
    """
    min_lr_ratio = max(0.0, float(train_cfg.min_lr_ratio))
    t_max = max(1, int(train_cfg.epochs))

    def _cosine_multiplier(epoch: int) -> float:
        # LambdaLR passes last_epoch (0 on first step); align with CosineAnnealingLR
        # which uses last_epoch + 1 after the same training epoch.
        progress = min(epoch + 1, t_max) / t_max
        cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[_cosine_multiplier for _ in optimizer.param_groups],
    )


def _optimizer_lr_snapshot(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    if len(optimizer.param_groups) == 1:
        return {"lr": float(optimizer.param_groups[0]["lr"])}
    labels = ("backbone_lr", "head_lr")
    snapshot: dict[str, float] = {}
    for idx, group in enumerate(optimizer.param_groups):
        key = labels[idx] if idx < len(labels) else f"group_{idx}_lr"
        snapshot[key] = float(group["lr"])
    return snapshot


def _warn_checkpoint_config_mismatch(
    model: nn.Module,
    action_cfg: ActionConfig,
) -> None:
    """Log when resume checkpoint architecture differs from the current config."""
    ckpt_cfg = getattr(model, "config", {})
    checks = {
        "backbone": action_cfg.backbone,
        "num_classes": action_cfg.num_classes,
        "lstm_hidden": action_cfg.lstm_hidden,
        "lstm_layers": action_cfg.lstm_layers,
        "bidirectional": action_cfg.bidirectional,
    }
    for key, expected in checks.items():
        saved = ckpt_cfg.get(key)
        if saved is not None and saved != expected:
            logger.warning(
                f"Resume checkpoint {key}={saved!r} differs from config {expected!r}; "
                "using checkpoint weights as-is."
            )


def train(
    manifest: str | Path,
    action_cfg: ActionConfig,
    train_cfg: TrainActionConfig,
    device: str = "cpu",
    seed: int = 42,
    pretrained_imagenet: bool = True,
    resume_checkpoint: str | Path | None = None,
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
    if train_cfg.event_match_tolerance_frames < 0:
        raise ValueError("event_match_tolerance_frames must be >= 0.")
    if train_cfg.boundary_soft_label_radius_frames > 0 and not use_bce:
        raise ValueError("Boundary soft labels currently require BCE action training.")

    train_ds = SplitStepClipDataset(
        manifest_path=manifest,
        split="train",
        clip_length=action_cfg.clip_length,
        event_gap_frames=max(1, int(action_cfg.clip_stride)),
        transform=build_train_transform(
            action_cfg.input_size,
            random_crop_margin=train_cfg.augmentation_random_crop_margin,
            horizontal_flip_probability=(
                train_cfg.augmentation_horizontal_flip_probability
            ),
            brightness=train_cfg.augmentation_brightness,
            contrast=train_cfg.augmentation_contrast,
            saturation=train_cfg.augmentation_saturation,
            bbox_translate=train_cfg.augmentation_bbox_translate,
            bbox_scale_min=train_cfg.augmentation_bbox_scale_min,
            bbox_scale_max=train_cfg.augmentation_bbox_scale_max,
            blur_probability=train_cfg.augmentation_blur_probability,
            blur_sigma_min=train_cfg.augmentation_blur_sigma_min,
            blur_sigma_max=train_cfg.augmentation_blur_sigma_max,
            jpeg_probability=train_cfg.augmentation_jpeg_probability,
            jpeg_quality_min=train_cfg.augmentation_jpeg_quality_min,
            jpeg_quality_max=train_cfg.augmentation_jpeg_quality_max,
            frame_drop_probability=train_cfg.augmentation_frame_drop_probability,
        ),
        boundary_soft_label_radius_frames=(
            train_cfg.boundary_soft_label_radius_frames
        ),
    )
    val_ds = SplitStepClipDataset(
        manifest_path=manifest,
        split="val",
        clip_length=action_cfg.clip_length,
        event_gap_frames=max(1, int(action_cfg.clip_stride)),
        transform=build_eval_transform(action_cfg.input_size),
    )
    test_ds = SplitStepClipDataset(
        manifest_path=manifest,
        split="test",
        clip_length=action_cfg.clip_length,
        event_gap_frames=max(1, int(action_cfg.clip_stride)),
        transform=build_eval_transform(action_cfg.input_size),
    )
    if len(train_ds) == 0:
        raise RuntimeError("Empty train split in manifest.")
    logger.info(
        f"Train clips: {len(train_ds)}  |  Val clips: {len(val_ds)}  "
        f"|  Test clips: {len(test_ds)}"
    )
    if train_ds.soft_target_count:
        logger.info(
            f"Boundary soft labels: {train_ds.soft_target_count} train clips  |  "
            f"radius={train_cfg.boundary_soft_label_radius_frames} frames"
        )

    if train_cfg.event_balanced_sampling:
        batch_sampler = EventBalancedBatchSampler(
            train_ds.records,
            train_cfg.batch_size,
            positive_fraction=train_cfg.event_positive_fraction,
            boundary_negative_fraction=train_cfg.event_boundary_negative_fraction,
            event_gap_frames=max(1, int(action_cfg.clip_stride)),
            boundary_radius_frames=train_cfg.event_boundary_radius_frames,
            seed=seed,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=batch_sampler,
            num_workers=train_cfg.num_workers,
            pin_memory=device_supports_pin_memory(device),
        )
        logger.info(
            f"Event-balanced sampling: {len(batch_sampler.positive_events)} events  |  "
            f"positive={train_cfg.event_positive_fraction:.0%}  |  "
            f"boundary-negative={train_cfg.event_boundary_negative_fraction:.0%}  |  "
            f"radius={train_cfg.event_boundary_radius_frames} frames"
        )
        if train_cfg.class_weight_balance:
            logger.warning(
                "Event-balanced sampling and class_weight_balance are both enabled; "
                "this can over-weight split-step clips and reduce precision."
            )
    else:
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

    resumed_from: Optional[str] = None
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)
        model = load_checkpoint(resume_path, map_location=device).to(device)
        _warn_checkpoint_config_mismatch(model, action_cfg)
        model.apply_freeze_backbone(action_cfg.freeze_backbone)
        model.apply_freeze_batchnorm_stats(action_cfg.freeze_batchnorm_stats)
        model.set_temperature(1.0)
        resumed_from = str(resume_path)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Resumed from {resume_path}  |  freeze_backbone={action_cfg.freeze_backbone}  "
            f"|  freeze_batchnorm_stats={action_cfg.freeze_batchnorm_stats}  "
            f"|  trainable params={n_trainable:,}"
        )
    else:
        model = build_model(action_cfg).to(device)
        imagenet_loaded = False
        if pretrained_imagenet:
            imagenet_loaded = model.try_load_imagenet_weights()
        if action_cfg.freeze_backbone and not imagenet_loaded:
            raise RuntimeError(
                "action.freeze_backbone=true requires ImageNet backbone weights. "
                "Either allow pretrained loading or set action.freeze_backbone=false."
            )

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

    optimizer = _build_optimizer(model, action_cfg, train_cfg)
    scheduler = _build_scheduler(optimizer, train_cfg)
    ema_decay = float(train_cfg.ema_decay)
    if not 0.0 <= ema_decay < 1.0:
        raise ValueError("train_action.ema_decay must satisfy 0 <= decay < 1.")
    ema_model: Optional[AveragedModel] = None
    if ema_decay > 0:
        ema_model = AveragedModel(
            model,
            multi_avg_fn=_ema_multi_avg_fn(ema_decay),
            use_buffers=False,
        )
        logger.info(
            f"EMA model enabled: target decay={ema_decay:.6f} with warmup"
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
    best_split_step_f1 = float("nan")
    best_event_f1 = float("nan")
    best_epoch: Optional[int] = None
    best_threshold = float(train_cfg.classification_threshold)
    fallback_threshold = float(train_cfg.classification_threshold)
    epochs_without_improvement = 0
    grad_clip_norm = max(0.0, float(train_cfg.grad_clip_norm))
    logger.info(
        "Validation sweeps threshold each epoch: "
        f"{train_cfg.threshold_sweep_min:.2f}..{train_cfg.threshold_sweep_max:.2f} "
        f"step {train_cfg.threshold_sweep_step:.2f} "
        f"(optimizing {best_metric_name}; fallback {fallback_threshold:.2f})"
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
            if ema_model is not None:
                ema_model.update_parameters(model)
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
        val_event_precision = float("nan")
        val_event_recall = float("nan")
        val_event_f1 = float("nan")
        val_event_support = 0
        val_score = float("nan")
        stop_score = float("nan")
        val_threshold = fallback_threshold
        if val_loader is not None:
            eval_model = ema_model.module if ema_model is not None else model
            val_metrics = evaluate_with_threshold_sweep(
                eval_model,
                val_loader,
                criterion,
                device,
                train_cfg,
                use_bce=use_bce,
                sweep_metric_name=best_metric_name,
                fallback_threshold=fallback_threshold,
                event_gap_frames=max(1, int(action_cfg.clip_stride)),
            )
            val_loss = val_metrics.loss
            val_acc = val_metrics.acc
            val_f1 = val_metrics.f1
            val_threshold = val_metrics.threshold
            val_split_precision = _class_report_metric(val_metrics.report, "1", "precision")
            val_split_recall = _class_report_metric(val_metrics.report, "1", "recall")
            val_split_f1 = _class_report_metric(val_metrics.report, "1", "f1-score")
            val_split_support = _class_report_metric(val_metrics.report, "1", "support")
            if val_metrics.event is not None:
                val_event_precision = val_metrics.event.precision
                val_event_recall = val_metrics.event.recall
                val_event_f1 = val_metrics.event.f1
                val_event_support = val_metrics.event.ground_truth_events
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
            "val_event_precision": val_event_precision,
            "val_event_recall": val_event_recall,
            "val_event_f1": val_event_f1,
            "val_event_support": val_event_support,
            "val_best_metric": best_metric_name,
            "val_best_metric_value": val_score,
            "val_stop_metric": stop_metric_name,
            "val_stop_metric_value": stop_score,
            **_optimizer_lr_snapshot(optimizer),
        }
        history.append(epoch_log)
        logger.info(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.3f} "
            f"| val_loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f} "
            f"| split_clip_f1 {val_split_f1:.3f} "
            f"split_event_f1 {val_event_f1:.3f} "
            f"thr {val_threshold:.2f}"
        )

        checkpoint_model = ema_model.module if ema_model is not None else model
        save_checkpoint(
            checkpoint_model,
            last_path,
            extra={"epoch": epoch, "ema_decay": ema_decay},
        )
        checkpoint_improved = _metric_improved(val_score, best_score, best_metric_name)
        if checkpoint_improved:
            best_score = val_score
            best_f1 = val_f1
            best_acc = val_acc
            best_split_step_f1 = val_split_f1
            best_event_f1 = val_event_f1
            best_threshold = val_threshold
            best_epoch = epoch
            save_checkpoint(
                checkpoint_model,
                best_path,
                extra={
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "val_split_step_f1": val_split_f1,
                    "val_event_f1": val_event_f1,
                    "best_metric": best_metric_name,
                    "best_metric_value": val_score,
                    "classification_threshold": best_threshold,
                    "ema_decay": ema_decay,
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
        checkpoint_model = ema_model.module if ema_model is not None else model
        save_checkpoint(
            checkpoint_model,
            best_path,
            extra={"epoch": train_cfg.epochs, "ema_decay": ema_decay},
        )
        best_f1 = float("nan")
        best_acc = float("nan")
        best_split_step_f1 = float("nan")
        best_event_f1 = float("nan")
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

    temperature: Optional[float] = None
    calibration_loss_before: Optional[float] = None
    calibration_loss_after: Optional[float] = None
    if train_cfg.temperature_calibration and val_loader is not None:
        temperature, calibration_loss_before, calibration_loss_after = fit_temperature(
            best_model,
            val_loader,
            device,
            use_bce=use_bce,
        )
        calibrated_val_metrics = evaluate_with_threshold_sweep(
            best_model,
            val_loader,
            criterion,
            device,
            train_cfg,
            use_bce=use_bce,
            sweep_metric_name=best_metric_name,
            fallback_threshold=fallback_threshold,
            event_gap_frames=max(1, int(action_cfg.clip_stride)),
        )
        best_threshold = calibrated_val_metrics.threshold
        best_f1 = calibrated_val_metrics.f1
        best_acc = calibrated_val_metrics.acc
        best_split_step_f1 = _class_report_metric(
            calibrated_val_metrics.report,
            "1",
            "f1-score",
        )
        best_event_f1 = (
            calibrated_val_metrics.event.f1
            if calibrated_val_metrics.event is not None
            else float("nan")
        )
        best_score = _select_metric(calibrated_val_metrics, best_metric_name)
        save_checkpoint(
            best_model,
            best_path,
            extra={
                "epoch": best_epoch,
                "val_f1": best_f1,
                "val_acc": best_acc,
                "val_loss": calibrated_val_metrics.loss,
                "val_split_step_f1": best_split_step_f1,
                "val_event_f1": best_event_f1,
                "best_metric": best_metric_name,
                "best_metric_value": best_score,
                "classification_threshold": best_threshold,
                "ema_decay": ema_decay,
                "temperature": temperature,
                "calibration_loss_before": calibration_loss_before,
                "calibration_loss_after": calibration_loss_after,
            },
        )
        logger.info(
            f"Temperature calibration: T={temperature:.4f}  "
            f"unweighted NLL {calibration_loss_before:.4f} -> "
            f"{calibration_loss_after:.4f}  |  threshold={best_threshold:.2f}"
        )

    test_acc: Optional[float] = None
    test_f1: Optional[float] = None
    test_split_step_f1: Optional[float] = None
    test_event_f1: Optional[float] = None
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
            event_gap_frames=max(1, int(action_cfg.clip_stride)),
            event_tolerance_frames=train_cfg.event_match_tolerance_frames,
        )
        test_loss = test_metrics.loss
        test_acc = test_metrics.acc
        test_f1 = test_metrics.f1
        test_report = test_metrics.report
        test_split_f1 = _class_report_metric(test_metrics.report, "1", "f1-score")
        test_split_step_f1 = test_split_f1
        test_event_f1 = (
            test_metrics.event.f1 if test_metrics.event is not None else None
        )
        logger.info(
            f"Test  | loss {test_loss:.4f}  acc {test_acc:.3f}  f1 {test_f1:.3f} "
            f"| split_clip_f1 {test_split_f1:.3f}  "
            f"split_event_f1 "
            f"{test_event_f1 if test_event_f1 is not None else float('nan'):.3f} "
            f"thr {best_threshold:.2f}"
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
        best_split_step_f1=None if math.isnan(best_split_step_f1) else best_split_step_f1,
        best_event_f1=None if math.isnan(best_event_f1) else best_event_f1,
        test_acc=test_acc,
        test_f1=test_f1,
        test_split_step_f1=test_split_step_f1,
        test_event_f1=test_event_f1,
        test_loss=test_loss,
        test_report=test_report,
        resumed_from=resumed_from,
        ema_decay=ema_decay if ema_model is not None else None,
        temperature=temperature,
        calibration_loss_before=calibration_loss_before,
        calibration_loss_after=calibration_loss_after,
    )
