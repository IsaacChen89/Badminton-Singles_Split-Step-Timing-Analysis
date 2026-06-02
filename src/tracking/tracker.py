"""Multi-object tracker built on ultralytics' built-in BoT-SORT/ByteTrack.

We use ``model.track(persist=True)`` which keeps a stateful tracker across
calls. The tracker YAML is selected from the ``mode``:

- ``strong``     -> ``configs/botsort_strong.yaml`` (Re-ID on, long buffer)
- ``normal``     -> ``configs/botsort_normal.yaml`` (defaults)
- ``court_side`` -> ``configs/botsort_normal.yaml`` (assigner ignores tracker IDs)

A user-supplied YAML path (or the stock ultralytics names ``botsort.yaml`` /
``bytetrack.yaml``) takes priority when ``tracker_yaml`` is not ``"auto"``.
If the chosen YAML fails to load (e.g. older ultralytics that doesn't know
``with_reid``), we transparently fall back to ``botsort.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from ..detection.yolo_detector import Detection, YOLODetector
from ..utils.logging import get_logger

logger = get_logger("tracking")


_CONFIG_DIR = Path(__file__).resolve().parent / "configs"
_MODE_TO_YAML = {
    "strong": _CONFIG_DIR / "botsort_strong.yaml",
    "normal": _CONFIG_DIR / "botsort_normal.yaml",
    "court_side": _CONFIG_DIR / "botsort_normal.yaml",
}


def resolve_tracker_yaml(mode: str, tracker_yaml: str = "auto") -> str:
    """Map ``(mode, tracker_yaml)`` to a concrete YAML reference for ultralytics.

    - If ``tracker_yaml`` is anything other than ``"auto"``, it is returned
      verbatim (path or ultralytics-known name).
    - Else we use the mode-specific bundled config; if it doesn't exist we
      fall back to the stock ``botsort.yaml`` reference.
    """
    if tracker_yaml and tracker_yaml.lower() != "auto":
        return tracker_yaml
    candidate = _MODE_TO_YAML.get(mode.lower())
    if candidate is not None and candidate.exists():
        return str(candidate)
    return "botsort.yaml"


class PlayerTracker:
    """Stateful tracker that wraps :class:`YOLODetector`.

    Parameters
    ----------
    detector:
        Loaded :class:`YOLODetector`.
    mode:
        Tracking mode: ``"strong" | "normal" | "court_side"``.
    tracker_yaml:
        ``"auto"`` (default; chosen from ``mode``) or an explicit path / known
        ultralytics name.
    persist:
        Forwarded to ultralytics; should be ``True`` for video streams.
    """

    def __init__(
        self,
        detector: YOLODetector,
        mode: str = "strong",
        tracker_yaml: str = "auto",
        persist: bool = True,
    ) -> None:
        self.detector = detector
        self.mode = mode
        self.tracker_yaml = resolve_tracker_yaml(mode, tracker_yaml)
        self.persist = persist
        self._frames_seen = 0
        self._fallback_yaml: Optional[str] = None
        logger.info(
            f"PlayerTracker mode='{mode}' tracker_yaml='{self.tracker_yaml}'"
        )

    def reset(self) -> None:
        """Reset internal tracker state (call between independent videos)."""
        try:
            predictor = getattr(self.detector.model, "predictor", None)
            if predictor is not None and hasattr(predictor, "trackers"):
                predictor.trackers = []
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Tracker reset best-effort failed: {exc}")
        self._frames_seen = 0

    def _track_call(self, frame: np.ndarray, tracker_yaml: str):
        return self.detector.model.track(
            source=frame,
            device=self.detector.device,
            conf=self.detector.conf,
            iou=self.detector.iou,
            imgsz=self.detector.imgsz,
            classes=self.detector.classes,
            max_det=self.detector.max_det,
            tracker=tracker_yaml,
            persist=self.persist,
            verbose=False,
        )

    def update(self, frame: np.ndarray) -> List[Detection]:
        """Run detection + tracking on ``frame``.

        Returns a list of :class:`Detection` objects with ``tracker_id``
        populated (un-IDed detections are dropped to keep player assignment
        clean).
        """
        try:
            results = self._track_call(frame, self.tracker_yaml)
        except Exception as exc:
            # Older ultralytics may not understand ``with_reid``. Fall back
            # once to the stock botsort yaml and remember the choice.
            if self._fallback_yaml is None and self.tracker_yaml != "botsort.yaml":
                logger.warning(
                    f"Tracker yaml '{self.tracker_yaml}' rejected by ultralytics "
                    f"({exc}); falling back to 'botsort.yaml'."
                )
                self._fallback_yaml = "botsort.yaml"
                self.tracker_yaml = "botsort.yaml"
                results = self._track_call(frame, self.tracker_yaml)
            else:
                raise
        self._frames_seen += 1
        out: List[Detection] = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        ids: Optional[np.ndarray] = (
            r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None
        )
        if ids is None:
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
                    tracker_id=int(ids[i]),
                )
            )
        return out
