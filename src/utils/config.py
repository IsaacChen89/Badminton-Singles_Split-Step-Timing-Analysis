"""Strongly-typed configuration objects backed by ``config.yaml``.

The runtime always loads ``config.yaml`` (or any user-supplied path) into the
``AppConfig`` dataclass. Sub-systems (detection, tracking, action, etc.)
receive only the slice they care about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import yaml


Device = Literal["auto", "cpu", "cuda", "mps"]


@dataclass
class PipelineConfig:
    frame_skip: int = 1
    # Resample the input video to this FPS before running the pipeline so the
    # action model always sees the cadence it was trained on (30 FPS). Set to
    # ``None`` or 0 to disable and keep the source rate.
    target_fps: Optional[float] = 30.0
    output_fps: Optional[float] = None
    draw_hud: bool = True


@dataclass
class DetectionConfig:
    model_path: str = "models/yolo26n.pt"
    finetuned_model_path: str = "models/yolo_player/yolo_player_best.pt"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5
    imgsz: int = 640
    classes: List[int] = field(default_factory=lambda: [0])
    max_det: int = 8


@dataclass
class TrackingConfig:
    """Low-level tracker (BoT-SORT/ByteTrack) configuration.

    ``mode`` selects the high-level behavior and also picks an appropriate
    tracker YAML when ``tracker_yaml`` is left at its default of ``auto``:

    - ``strong``   -> ``botsort_strong.yaml`` (Re-ID on, long buffer, tighter thresholds)
    - ``normal``   -> ``botsort_normal.yaml`` (Re-ID off, defaults)
    - ``court_side`` -> ``botsort_normal.yaml`` (assigner ignores tracker IDs anyway)
    """

    mode: str = "strong"  # strong | normal | court_side
    tracker_yaml: str = "auto"  # 'auto' | 'botsort.yaml' | 'bytetrack.yaml' | <path/to/custom.yaml>
    persist: bool = True
    show_tracker_ids: bool = False


@dataclass
class AssignmentConfig:
    """How tracker IDs are mapped onto stable Player 1 / Player 2 slots."""

    # Where the upper-court / left-court / etc. player lives in the frame.
    # One of: top | bottom | left | right
    player1_position: str = "top"
    # Back-compat: if True and ``player1_position`` is at its default,
    # keep behaving as the old top-half rule.
    top_is_player1: bool = True

    reassign_after_lost_frames: int = 60
    bbox_smoothing_alpha: float = 0.5     # EMA on rendered bbox; 1.0 disables smoothing
    velocity_alpha: float = 0.4           # EMA on per-frame bbox velocity for prediction
    predict_max_frames: int = 12          # render predicted bbox for up to N missing frames
    min_confidence: float = 0.25          # drop detections below this score
    iou_recovery_threshold: float = 0.2   # min IoU to recover a slot via continuity


@dataclass
class ActionConfig:
    model_checkpoint: str = "models/action_player/action_best.pt"
    clip_length: int = 16
    clip_stride: int = 1
    input_size: int = 224
    num_classes: int = 2
    backbone: str = "resnet18"
    freeze_backbone: bool = True
    freeze_batchnorm_stats: bool = False
    lstm_hidden: int = 128
    lstm_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.2
    feature_dropout: float = 0.25  # between backbone and BiLSTM


@dataclass
class SmoothingConfig:
    ema_alpha: float = 0.4
    prob_on: float = 0.6
    prob_off: float = 0.4
    min_on_frames: int = 3
    cooldown_frames: int = 6


@dataclass
class TrainActionConfig:
    epochs: int = 30
    batch_size: int = 16
    lr: float = 3e-4
    backbone_lr: Optional[float] = None
    head_lr: Optional[float] = None
    weight_decay: float = 1e-4
    num_workers: int = 2
    val_split: float = 0.2
    loss: str = "cross_entropy"
    class_weight_balance: bool = True
    label_smoothing: float = 0.0
    early_stopping_patience: int = 0
    early_stopping_metric: str = "val_loss"
    best_metric: str = "macro_f1"
    classification_threshold: float = 0.5
    min_lr_ratio: float = 0.05
    max_pos_weight: float = 4.0
    grad_clip_norm: float = 1.0
    threshold_sweep_min: float = 0.10
    threshold_sweep_max: float = 0.90
    threshold_sweep_step: float = 0.05
    ema_decay: float = 0.0
    temperature_calibration: bool = False
    event_balanced_sampling: bool = False
    event_positive_fraction: float = 0.35
    event_boundary_negative_fraction: float = 0.25
    event_boundary_radius_frames: int = 8
    augmentation_random_crop_margin: int = 16
    augmentation_horizontal_flip_probability: float = 0.5
    augmentation_brightness: float = 0.2
    augmentation_contrast: float = 0.2
    augmentation_saturation: float = 0.2
    augmentation_bbox_translate: float = 0.05
    augmentation_bbox_scale_min: float = 0.9
    augmentation_bbox_scale_max: float = 1.1
    augmentation_blur_probability: float = 0.15
    augmentation_blur_sigma_min: float = 0.1
    augmentation_blur_sigma_max: float = 1.5
    augmentation_jpeg_probability: float = 0.15
    augmentation_jpeg_quality_min: int = 60
    augmentation_jpeg_quality_max: int = 95
    augmentation_frame_drop_probability: float = 0.05
    augmentation_frame_shift_max: int = 0
    amp: bool = True
    output_dir: str = "models/action_player"


@dataclass
class TrainYoloConfig:
    base_model: str = "models/yolo26n.pt"
    epochs: int = 50
    imgsz: int = 640
    batch: int = 16
    project: str = "models"
    name: str = "yolo_player"


@dataclass
class CvatConfig:
    player1_label: str = "player1"
    player2_label: str = "player2"
    split_attribute: str = "split_step"
    yolo_class_name: str = "player"
    centered_action_clips: bool = False
    # Three-way split: train share is implicit (1 - val_split - test_split).
    val_split: float = 0.2
    test_split: float = 0.2
    every_n_frames: int = 1
    group_split: bool = True


@dataclass
class AppConfig:
    device: Device = "auto"
    seed: int = 42
    log_level: str = "INFO"
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    train_action: TrainActionConfig = field(default_factory=TrainActionConfig)
    train_yolo: TrainYoloConfig = field(default_factory=TrainYoloConfig)
    cvat: CvatConfig = field(default_factory=CvatConfig)


def _coerce(section_cls, raw: Optional[dict]):
    """Build a dataclass instance from a (possibly partial) dict."""
    if raw is None:
        return section_cls()
    valid = {f for f in section_cls.__dataclass_fields__}
    filtered = {k: v for k, v in raw.items() if k in valid}
    return section_cls(**filtered)


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load an :class:`AppConfig` from a YAML file.

    Missing keys fall back to the dataclass defaults so partial configs are
    safe.
    """
    p = Path(path)
    if not p.exists():
        return AppConfig()

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return AppConfig(
        device=data.get("device", "auto"),
        seed=int(data.get("seed", 42)),
        log_level=data.get("log_level", "INFO"),
        pipeline=_coerce(PipelineConfig, data.get("pipeline")),
        detection=_coerce(DetectionConfig, data.get("detection")),
        tracking=_coerce(TrackingConfig, data.get("tracking")),
        assignment=_coerce(AssignmentConfig, data.get("assignment")),
        action=_coerce(ActionConfig, data.get("action")),
        smoothing=_coerce(SmoothingConfig, data.get("smoothing")),
        train_action=_coerce(TrainActionConfig, data.get("train_action")),
        train_yolo=_coerce(TrainYoloConfig, data.get("train_yolo")),
        cvat=_coerce(CvatConfig, data.get("cvat")),
    )


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is not installed in the active Python environment. "
            "Activate the project virtualenv (`source .venv/bin/activate`) "
            "or run with `.venv/bin/python`, then `pip install -r requirements.txt`."
        ) from exc
    return torch


def mps_available() -> bool:
    """Return True when PyTorch can use Apple Metal (MPS)."""
    torch = _require_torch()

    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def resolve_device(requested: Device) -> str:
    """Map ``auto|cpu|cuda|mps`` to a concrete torch device string.

    ``auto`` prefers CUDA, then Apple MPS, then CPU.
    """
    torch = _require_torch()

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        return "cuda"
    if requested == "mps":
        if not mps_available():
            raise RuntimeError(
                "MPS requested but torch.backends.mps.is_available() is False."
            )
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if mps_available():
        return "mps"
    return "cpu"


def device_supports_pin_memory(device: str) -> bool:
    """DataLoader pin_memory is only beneficial for CUDA transfers."""
    return device == "cuda"


def device_supports_amp(device: str) -> bool:
    """Mixed precision is enabled for CUDA only (stable GradScaler path)."""
    return device == "cuda"
