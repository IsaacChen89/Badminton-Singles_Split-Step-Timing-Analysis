"""Video I/O wrappers around OpenCV.

These exist so the rest of the codebase doesn't have to think about OpenCV's
VideoCapture quirks (random-seek behavior, FOURCC selection, color order).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


class VideoReader(AbstractContextManager):
    """Sequential + random-access video reader.

    Always returns BGR frames (the OpenCV default). Use as a context manager
    or call ``release()`` manually.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {self.path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ok, frame = self.cap.read()
        return ok, frame if ok else None

    def read_at(self, frame_idx: int) -> Optional[np.ndarray]:
        """Random-access read of frame ``frame_idx``.

        Note: random seeks rely on the codec's keyframe layout and may be
        approximate for some containers. For sequential workloads, prefer
        :meth:`iter_frames`.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = self.cap.read()
        return frame if ok else None

    def iter_frames(self, start: int = 0, end: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        """Yield ``(frame_idx, frame)`` from ``start`` to ``end`` exclusive."""
        if start > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
        i = start
        while True:
            if end is not None and i >= end:
                return
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return
            yield i, frame
            i += 1


def effective_fps(source_fps: float, target_fps: Optional[float]) -> float:
    """Pick the FPS the inference pipeline should treat the stream as running at.

    Returns ``source_fps`` when resampling is disabled (``target_fps`` is
    ``None`` or non-positive) or when the source rate is already within
    0.5 FPS of the target (the typical 29.97 vs. 30 case). Otherwise
    returns ``float(target_fps)``.
    """
    if not target_fps or target_fps <= 0:
        return source_fps
    if source_fps <= 0:
        return float(target_fps)
    if abs(source_fps - target_fps) < 0.5:
        return source_fps
    return float(target_fps)


def iter_frames_at_fps(
    reader: "VideoReader",
    target_fps: Optional[float],
    start: int = 0,
    end: Optional[int] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(out_idx, frame)`` from ``reader`` at approximately ``target_fps``.

    Drops or duplicates source frames so the long-run output rate matches
    ``target_fps`` exactly. When :func:`effective_fps` decides no resampling
    is needed (target is ``None``, non-positive, or close enough to the
    source), this iterates ``reader.iter_frames(start, end)`` directly.

    ``out_idx`` always counts the emitted frames so downstream code (clip
    windows, smoothing, the on-screen clock) can treat the stream as if it
    were captured at ``target_fps`` natively.
    """
    eff = effective_fps(reader.fps, target_fps)
    src_iter = reader.iter_frames(start=start, end=end)
    if eff == reader.fps:
        yield from src_iter
        return

    source_fps = reader.fps if reader.fps > 0 else eff
    ratio = eff / source_fps
    out_idx = 0
    src_consumed = 0
    # Recompute the desired output count from a single multiplication each
    # iteration. This is numerically stable for clean ratios like 30/25 or
    # 30/60 where naive accumulation would drift by one frame per second.
    for _src_idx, frame in src_iter:
        src_consumed += 1
        desired = int(src_consumed * ratio)
        while out_idx < desired:
            yield out_idx, frame
            out_idx += 1


class VideoWriter(AbstractContextManager):
    """Wrapper around ``cv2.VideoWriter`` that picks a portable codec.

    Tries ``avc1`` first (H.264 in MP4) and falls back to ``mp4v``.
    """

    _CANDIDATES = ("avc1", "mp4v")

    def __init__(
        self,
        path: str | Path,
        fps: float,
        size: Tuple[int, int],
    ) -> None:
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.size = size
        self.writer: Optional[cv2.VideoWriter] = None
        self._fourcc_used: Optional[str] = None
        for code in self._CANDIDATES:
            fourcc = cv2.VideoWriter_fourcc(*code)
            writer = cv2.VideoWriter(self.path, fourcc, fps, size)
            if writer.isOpened():
                self.writer = writer
                self._fourcc_used = code
                break
            writer.release()
        if self.writer is None:
            raise IOError(
                f"Could not open VideoWriter for {self.path} at fps={fps} size={size}"
            )

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    @property
    def fourcc(self) -> Optional[str]:
        return self._fourcc_used

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            raise RuntimeError("VideoWriter is closed.")
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None
