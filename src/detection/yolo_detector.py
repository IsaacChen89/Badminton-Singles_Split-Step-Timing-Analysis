"""Thin wrapper around an ultralytics YOLO model for player detection.

Why a wrapper? It (a) hides the ultralytics-specific result schema behind a
simple :class:`Detection` namedtuple, (b) automatically chooses between a
fine-tuned single-class (``player``) checkpoint and a stock COCO model, and
(c) plays nicely with our :class:`PlayerTracker` which calls ``model.track``
directly on the underlying ``YOLO`` instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ..utils.logging import get_logger

logger = get_logger("detection")


@dataclass
class Detection:
    """A single player detection in pixel coordinates."""

    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    score: float
    class_id: int
    tracker_id: Optional[int] = None


def _select_model_path(
    primary: str,
    fallback: str,
) -> str:
    """Pick the fine-tuned weights when present, else fall back to stock."""
    if primary and Path(primary).exists():
        return primary
    if fallback and Path(fallback).exists():
        return fallback
    # Last resort: ask ultralytics to download the stock checkpoint by name.
    return fallback or primary


class YOLODetector:
    """Lazy-loaded ultralytics YOLO detector with a stable output type.

    Parameters
    ----------
    model_path:
        Path to a fine-tuned single-class ``player`` checkpoint, or ``None`` to
        use ``stock_model_path``.
    stock_model_path:
        Path or name (e.g. ``yolo26n.pt``) used when ``model_path`` is missing.
    device:
        ``"cpu"`` or ``"cuda"``.
    conf:
        Detection confidence threshold.
    iou:
        NMS IoU threshold.
    classes:
        Filter to these class indices (e.g. ``[0]`` for COCO ``person``).
        Ignored for fine-tuned single-class models.
    imgsz:
        Inference image size.
    max_det:
        Per-frame max detections.
    """

    def __init__(
        self,
        model_path: Optional[str],
        stock_model_path: str = "yolo26n.pt",
        device: str = "cpu",
        conf: float = 0.35,
        iou: float = 0.5,
        classes: Optional[Sequence[int]] = (0,),
        imgsz: int = 640,
        max_det: int = 8,
    ) -> None:
        from ultralytics import YOLO  # imported lazily so import-time stays cheap

        chosen = _select_model_path(model_path or "", stock_model_path)
        logger.info(f"Loading YOLO weights: {chosen}")
        self.model = YOLO(chosen)
        self.device = device
        self.conf = conf
        self.iou = iou
        self.classes = list(classes) if classes is not None else None
        self.imgsz = imgsz
        self.max_det = max_det
        self.is_finetuned = bool(model_path) and Path(model_path).exists()
        # Single-class fine-tuned model -> ignore class filter (it's just 'player').
        if self.is_finetuned:
            self.classes = None

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR frame.

        Returns a list of :class:`Detection` objects sorted by descending
        score.
        """
        results = self.model.predict(
            source=frame,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            classes=self.classes,
            max_det=self.max_det,
            verbose=False,
        )
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for i in range(xyxy.shape[0]):
            out.append(
                Detection(
                    bbox=(
                        float(xyxy[i, 0]),
                        float(xyxy[i, 1]),
                        float(xyxy[i, 2]),
                        float(xyxy[i, 3]),
                    ),
                    score=float(conf[i]),
                    class_id=int(cls[i]),
                )
            )
        out.sort(key=lambda d: d.score, reverse=True)
        return out
