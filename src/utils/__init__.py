"""Utility helpers (config loading, logging, geometry)."""

__all__ = [
    "AppConfig",
    "ActionConfig",
    "AssignmentConfig",
    "DetectionConfig",
    "PipelineConfig",
    "SmoothingConfig",
    "TrackingConfig",
    "TrainActionConfig",
    "TrainYoloConfig",
    "CvatConfig",
    "load_config",
    "resolve_device",
    "setup_logging",
    "get_logger",
    "bbox_center",
    "clip_bbox",
    "crop_with_padding",
    "iou_xyxy",
    "xyxy_to_yolo",
]


def __getattr__(name: str):
    """Lazily import utility groups so pure helpers avoid native dependencies."""
    if name in {
        "AppConfig",
        "ActionConfig",
        "AssignmentConfig",
        "DetectionConfig",
        "PipelineConfig",
        "SmoothingConfig",
        "TrackingConfig",
        "TrainActionConfig",
        "TrainYoloConfig",
        "CvatConfig",
        "load_config",
        "resolve_device",
    }:
        from . import config

        return getattr(config, name)
    if name in {"setup_logging", "get_logger"}:
        from . import logging

        return getattr(logging, name)
    if name in {"bbox_center", "clip_bbox", "crop_with_padding", "iou_xyxy", "xyxy_to_yolo"}:
        from . import geometry

        return getattr(geometry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
