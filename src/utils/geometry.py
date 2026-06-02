"""Geometry helpers for bounding boxes and crops."""

from __future__ import annotations

from typing import Tuple

import numpy as np


BBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


def clip_bbox(bbox: BBox, width: int, height: int) -> BBox:
    """Clamp a bbox to image bounds while preserving its order."""
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(x1), width - 1))
    y1 = max(0.0, min(float(y1), height - 1))
    x2 = max(0.0, min(float(x2), width - 1))
    y2 = max(0.0, min(float(y2), height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def iou_xyxy(a: BBox, b: BBox) -> float:
    """Standard IoU on (x1,y1,x2,y2) pairs."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def xyxy_to_yolo(
    bbox: BBox,
    img_w: int,
    img_h: int,
) -> Tuple[float, float, float, float]:
    """Convert ``(x1,y1,x2,y2)`` pixels to YOLO ``(cx,cy,w,h)`` normalized."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) * 0.5 / img_w
    cy = (y1 + y2) * 0.5 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return (cx, cy, w, h)


def crop_with_padding(
    image: np.ndarray,
    bbox: BBox,
    pad_ratio: float = 0.1,
) -> np.ndarray:
    """Crop ``image`` to ``bbox`` with a fractional padding around the box.

    The crop is clipped to the frame bounds. Returns an ``H x W x 3`` array
    in the input image's color order. If the resulting crop is empty for any
    reason (e.g. zero-area bbox), a 1x1 black pixel is returned to keep
    downstream tensor shapes valid.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, w, h)
    bw = x2 - x1
    bh = y2 - y1
    px = bw * pad_ratio
    py = bh * pad_ratio
    px1 = int(max(0, np.floor(x1 - px)))
    py1 = int(max(0, np.floor(y1 - py)))
    px2 = int(min(w, np.ceil(x2 + px)))
    py2 = int(min(h, np.ceil(y2 + py)))
    if px2 <= px1 or py2 <= py1:
        return np.zeros((1, 1, 3), dtype=image.dtype)
    return image[py1:py2, px1:px2].copy()
