"""Map raw tracker IDs to stable Player 1 / Player 2 identities.

Goals
-----
Keep two stable identities (Player 1 = red, Player 2 = blue) for the entire
length of a match, even when:

* Players cross sides briefly (or the umpire/coach walks across).
* Motion blur causes the detector to merge / drop a player for a few frames.
* Players wear similar uniforms (Re-ID alone can't tell them apart).
* The tracker swaps IDs when bounding boxes overlap.

Strategy
--------
Per frame we maintain two *slots* (player_id ∈ {1, 2}). Each slot keeps:

* The latest **raw** detection bbox + its tracker id and score.
* An EMA-**smoothed** bbox for rendering (no jitter).
* A first-order **velocity** so we can predict where a missing player should
  be in the next frame and still match a re-appearing detection.
* ``frames_since_seen`` so we can render the predicted bbox for a short
  bridging window and recover the slot if a detection comes back close to
  the prediction.

The slot ↔ tracker_id mapping is **sticky**: once a slot owns a tracker_id
we keep it until either (a) the tracker_id has been gone for
``reassign_after_lost_frames`` frames, or (b) the tracker drops it and a
better candidate (by IoU vs. the predicted bbox) shows up.

When the sticky map can't claim both slots from the new detections, we fall
back to a **court-side rule** (``player1_position``: top / bottom / left /
right). For badminton singles the camera is almost always behind one
baseline, so the upper-court player is reliably Player 1 by default.

Three high-level modes
----------------------
``strong`` (default)
    Full pipeline: sticky map + IoU recovery + trajectory prediction +
    bbox smoothing + court-side fallback when a slot has been gone long
    enough.
``normal``
    Sticky map only. No trajectory, no court-side fallback. Lighter and
    matches the very first version of the assigner.
``court_side``
    Tracker IDs are ignored entirely. Every frame we sort detections by
    court half and assign accordingly. Bullet-proof against ID swaps,
    but won't follow players if they actually cross sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from ..detection.yolo_detector import Detection
from ..utils.geometry import bbox_center, clip_bbox, iou_xyxy
from ..utils.logging import get_logger

logger = get_logger("assignment")


PLAYER1_ID = 1
PLAYER2_ID = 2

PlayerPosition = Literal["top", "bottom", "left", "right"]
TrackingMode = Literal["strong", "normal", "court_side"]


# -------------------------------------------------------------------- #
# Public dataclasses
# -------------------------------------------------------------------- #
@dataclass
class AssignedPlayer:
    """A detection annotated with a stable player identity + smoothed bbox.

    Attributes
    ----------
    player_id:
        ``1`` (red) or ``2`` (blue).
    detection:
        The underlying raw :class:`Detection` (bbox is the *raw* detector
        output). May be ``None`` when the slot is being held over with a
        predicted bbox (see ``predicted``).
    bbox:
        The bbox that should be rendered (EMA-smoothed; falls back to the
        prediction when no fresh detection is present).
    confidence:
        Smoothed detection confidence.
    tracker_id:
        Latest tracker id owning the slot, when known.
    predicted:
        ``True`` when this assignment is being held over from a previous
        frame's prediction (no detection on this frame). The renderer can
        choose to draw such boxes with a dashed style if desired.
    """

    player_id: int
    detection: Optional[Detection]
    bbox: Tuple[float, float, float, float]
    confidence: float
    tracker_id: Optional[int] = None
    predicted: bool = False


@dataclass
class _Slot:
    """Internal per-player state."""

    player_id: int
    tracker_id: Optional[int] = None
    bbox_raw: Optional[Tuple[float, float, float, float]] = None
    bbox_smoothed: Optional[Tuple[float, float, float, float]] = None
    velocity: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # cx, cy, w, h
    confidence: float = 0.0
    frames_since_seen: int = 10**6
    last_seen_frame: int = -10**6


# -------------------------------------------------------------------- #
# PlayerAssigner
# -------------------------------------------------------------------- #
class PlayerAssigner:
    """Convert tracked detections into stable Player 1 / Player 2 slots."""

    def __init__(
        self,
        mode: TrackingMode = "strong",
        player1_position: PlayerPosition = "top",
        # Back-compat: if caller still uses the legacy bool we honor it.
        top_is_player1: Optional[bool] = None,
        reassign_after_lost_frames: int = 60,
        bbox_smoothing_alpha: float = 0.5,
        velocity_alpha: float = 0.4,
        predict_max_frames: int = 12,
        min_confidence: float = 0.25,
        iou_recovery_threshold: float = 0.2,
    ) -> None:
        if mode not in {"strong", "normal", "court_side"}:
            raise ValueError(f"unknown mode: {mode!r}")
        if player1_position not in {"top", "bottom", "left", "right"}:
            raise ValueError(f"unknown player1_position: {player1_position!r}")

        # Legacy bool wins if the caller is still passing the old kwarg AND
        # left player1_position at its default.
        if top_is_player1 is not None and player1_position == "top":
            player1_position = "top" if top_is_player1 else "bottom"

        self.mode: TrackingMode = mode
        self.player1_position: PlayerPosition = player1_position
        self.reassign_after_lost_frames = reassign_after_lost_frames
        self.bbox_smoothing_alpha = max(0.0, min(1.0, bbox_smoothing_alpha))
        self.velocity_alpha = max(0.0, min(1.0, velocity_alpha))
        self.predict_max_frames = max(0, int(predict_max_frames))
        self.min_confidence = float(min_confidence)
        self.iou_recovery_threshold = float(iou_recovery_threshold)

        self._slots: Dict[int, _Slot] = {
            PLAYER1_ID: _Slot(player_id=PLAYER1_ID),
            PLAYER2_ID: _Slot(player_id=PLAYER2_ID),
        }
        self._frame_w: int = 0
        self._frame_h: int = 0
        self._frame_idx: int = -1

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def set_frame_size(self, width: int, height: int) -> None:
        """Tell the assigner how big the video frames are.

        Required for the court-side rule. Call once per video.
        """
        self._frame_w = int(width)
        self._frame_h = int(height)

    def reset(self) -> None:
        for s in self._slots.values():
            self._slots[s.player_id] = _Slot(player_id=s.player_id)
        self._frame_idx = -1

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _is_player1_half(self, bbox: Tuple[float, float, float, float]) -> bool:
        cx, cy = bbox_center(bbox)
        if self._frame_w <= 0 or self._frame_h <= 0:
            # Without frame size, fall back to a relative comparison vs. the
            # other detection (handled by the caller).
            return cy < 0  # never used in that case
        if self.player1_position == "top":
            return cy < self._frame_h * 0.5
        if self.player1_position == "bottom":
            return cy > self._frame_h * 0.5
        if self.player1_position == "left":
            return cx < self._frame_w * 0.5
        return cx > self._frame_w * 0.5  # "right"

    def _other_pid(self, pid: int) -> int:
        return PLAYER2_ID if pid == PLAYER1_ID else PLAYER1_ID

    @staticmethod
    def _to_cxcywh(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1)

    @staticmethod
    def _from_cxcywh(c: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        cx, cy, w, h = c
        return (cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5)

    def _predicted_bbox(self, slot: _Slot) -> Optional[Tuple[float, float, float, float]]:
        if slot.bbox_smoothed is None:
            return None
        cx, cy, w, h = self._to_cxcywh(slot.bbox_smoothed)
        vcx, vcy, vw, vh = slot.velocity
        # Velocity is "per processed frame"; apply once to bridge the missing frame.
        cx += vcx
        cy += vcy
        w = max(1.0, w + vw)
        h = max(1.0, h + vh)
        bbox = self._from_cxcywh((cx, cy, w, h))
        if self._frame_w > 0 and self._frame_h > 0:
            bbox = clip_bbox(bbox, self._frame_w, self._frame_h)
        return bbox

    def _update_slot_with_detection(
        self,
        slot: _Slot,
        det: Detection,
    ) -> None:
        new_raw = det.bbox
        # Velocity update from previous smoothed bbox (more stable than raw).
        if slot.bbox_smoothed is not None and slot.frames_since_seen <= self.predict_max_frames:
            prev = self._to_cxcywh(slot.bbox_smoothed)
            curr = self._to_cxcywh(new_raw)
            gap = max(1, slot.frames_since_seen)  # frames between observations
            inst_v = tuple((c - p) / gap for c, p in zip(curr, prev))
            slot.velocity = tuple(  # type: ignore[assignment]
                self.velocity_alpha * iv + (1 - self.velocity_alpha) * sv
                for iv, sv in zip(inst_v, slot.velocity)
            )
        else:
            slot.velocity = (0.0, 0.0, 0.0, 0.0)

        # Smoothed bbox (EMA in cx/cy/w/h space).
        a = self.bbox_smoothing_alpha
        if slot.bbox_smoothed is None or a >= 1.0:
            slot.bbox_smoothed = new_raw
        else:
            prev = self._to_cxcywh(slot.bbox_smoothed)
            curr = self._to_cxcywh(new_raw)
            blended = tuple(a * c + (1 - a) * p for c, p in zip(curr, prev))
            slot.bbox_smoothed = self._from_cxcywh(blended)
            if self._frame_w > 0 and self._frame_h > 0:
                slot.bbox_smoothed = clip_bbox(
                    slot.bbox_smoothed, self._frame_w, self._frame_h
                )

        slot.bbox_raw = new_raw
        slot.confidence = (
            0.4 * slot.confidence + 0.6 * det.score if slot.confidence > 0 else det.score
        )
        slot.tracker_id = det.tracker_id
        slot.frames_since_seen = 0
        slot.last_seen_frame = self._frame_idx

    def _hold_slot(self, slot: _Slot) -> None:
        """Advance a slot one frame without a fresh detection."""
        slot.frames_since_seen += 1
        if slot.frames_since_seen <= self.predict_max_frames and slot.bbox_smoothed is not None:
            predicted = self._predicted_bbox(slot)
            if predicted is not None:
                slot.bbox_smoothed = predicted

    # ------------------------------------------------------------------ #
    # Mode-specific assignment
    # ------------------------------------------------------------------ #
    def _assign_court_side_only(
        self, detections: List[Detection]
    ) -> List[Tuple[int, Detection]]:
        """Assign every detection by court half; ignore tracker IDs entirely."""
        if not detections:
            return []
        ordered = sorted(detections, key=lambda d: d.score, reverse=True)[:2]
        out: List[Tuple[int, Detection]] = []
        # If both detections are present, use the rule directly; if there's
        # ambiguity (both fall in the same half), keep the higher-scoring one
        # in its half and put the other in the opposite slot.
        if len(ordered) == 2:
            a, b = ordered
            a_is_p1 = self._is_player1_half(a.bbox)
            b_is_p1 = self._is_player1_half(b.bbox)
            if a_is_p1 != b_is_p1:
                out.append((PLAYER1_ID if a_is_p1 else PLAYER2_ID, a))
                out.append((PLAYER1_ID if b_is_p1 else PLAYER2_ID, b))
            else:
                # Both in same half: keep a where the rule puts it; force b
                # into the opposite slot.
                pid_a = PLAYER1_ID if a_is_p1 else PLAYER2_ID
                out.append((pid_a, a))
                out.append((self._other_pid(pid_a), b))
        else:
            d = ordered[0]
            out.append((PLAYER1_ID if self._is_player1_half(d.bbox) else PLAYER2_ID, d))
        return out

    def _match_sticky_then_iou(
        self,
        detections: List[Detection],
    ) -> Tuple[Dict[int, Detection], List[Detection]]:
        """Try to match detections to slots via sticky tracker_id then IoU vs. prediction.

        Returns ``(matched, unmatched_dets)`` where ``matched`` maps
        ``player_id -> Detection``.
        """
        matched: Dict[int, Detection] = {}
        used_tids: set[int] = set()

        # 1. Sticky tracker_id -> slot mapping.
        for det in detections:
            tid = det.tracker_id
            if tid is None or tid in used_tids:
                continue
            for pid, slot in self._slots.items():
                if pid in matched:
                    continue
                if slot.tracker_id == tid and slot.frames_since_seen <= self.reassign_after_lost_frames:
                    matched[pid] = det
                    used_tids.add(tid)
                    break

        # 2. IoU recovery vs. predicted bbox for any remaining slot.
        unmatched_dets = [d for d in detections if d.tracker_id not in used_tids]
        if unmatched_dets and len(matched) < 2:
            pairs: List[Tuple[float, int, int]] = []  # (iou, det_idx, pid)
            for i, det in enumerate(unmatched_dets):
                for pid, slot in self._slots.items():
                    if pid in matched:
                        continue
                    pred = self._predicted_bbox(slot) or slot.bbox_smoothed
                    if pred is None:
                        continue
                    pairs.append((iou_xyxy(det.bbox, pred), i, pid))
            pairs.sort(reverse=True)
            taken_dets: set[int] = set()
            for iou, i, pid in pairs:
                if iou < self.iou_recovery_threshold:
                    break
                if pid in matched or i in taken_dets:
                    continue
                det = unmatched_dets[i]
                matched[pid] = det
                taken_dets.add(i)
                if det.tracker_id is not None:
                    used_tids.add(det.tracker_id)

        residual = [
            d
            for d in detections
            if d.tracker_id is None or d.tracker_id not in used_tids
        ]
        return matched, residual

    def _court_side_fallback(
        self,
        residual: List[Detection],
        already_taken: Dict[int, Detection],
    ) -> Dict[int, Detection]:
        """Fill missing slots from residual detections using the court-side rule."""
        out = dict(already_taken)
        free_pids = [pid for pid in (PLAYER1_ID, PLAYER2_ID) if pid not in out]
        if not free_pids or not residual:
            return out

        # Pick the two best (by score) from residuals to consider.
        ordered = sorted(residual, key=lambda d: d.score, reverse=True)
        for det in ordered:
            if not free_pids:
                break
            pid_by_rule = PLAYER1_ID if self._is_player1_half(det.bbox) else PLAYER2_ID
            # If the rule's preferred pid is free, take it. Else if any pid is
            # free, take that (the rule is ambiguous when both are in the same
            # half).
            if pid_by_rule in free_pids:
                out[pid_by_rule] = det
                free_pids.remove(pid_by_rule)
            else:
                pid = free_pids.pop(0)
                out[pid] = det
        return out

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #
    def assign(
        self,
        detections: List[Detection],
        frame_idx: Optional[int] = None,
    ) -> List[AssignedPlayer]:
        """Convert tracked detections to player-tagged detections.

        At most one Player 1 and one Player 2 are returned per call. The
        returned ``bbox`` is EMA-smoothed (or predicted, when ``predicted``
        is True).
        """
        if frame_idx is not None:
            self._frame_idx = frame_idx
        else:
            self._frame_idx += 1

        # Confidence filter.
        dets = [d for d in detections if d.score >= self.min_confidence]

        # Court-side mode: short circuit.
        if self.mode == "court_side":
            assignments = self._assign_court_side_only(dets)
            slot_to_det: Dict[int, Detection] = {pid: d for pid, d in assignments}
        else:
            matched, residual = self._match_sticky_then_iou(dets)
            if self.mode == "strong":
                # Only fall back to court-side once the slot has been missing
                # long enough OR we have nothing better. This avoids reacting
                # to single-frame drops.
                missing_pids = [
                    pid for pid in (PLAYER1_ID, PLAYER2_ID) if pid not in matched
                ]
                slot_lost_long = any(
                    self._slots[pid].frames_since_seen > self.predict_max_frames
                    for pid in missing_pids
                )
                if missing_pids and (slot_lost_long or not self._slots[missing_pids[0]].bbox_smoothed):
                    matched = self._court_side_fallback(residual, matched)
            slot_to_det = matched

        # Update slot state (with detection where present, hold otherwise).
        for pid, slot in self._slots.items():
            det = slot_to_det.get(pid)
            if det is not None:
                self._update_slot_with_detection(slot, det)
            else:
                self._hold_slot(slot)

        # Build output.
        out: List[AssignedPlayer] = []
        for pid in (PLAYER1_ID, PLAYER2_ID):
            slot = self._slots[pid]
            det = slot_to_det.get(pid)
            if det is not None:
                out.append(
                    AssignedPlayer(
                        player_id=pid,
                        detection=det,
                        bbox=slot.bbox_smoothed or det.bbox,
                        confidence=slot.confidence,
                        tracker_id=slot.tracker_id,
                        predicted=False,
                    )
                )
            elif (
                slot.bbox_smoothed is not None
                and slot.frames_since_seen <= self.predict_max_frames
            ):
                out.append(
                    AssignedPlayer(
                        player_id=pid,
                        detection=None,
                        bbox=slot.bbox_smoothed,
                        confidence=slot.confidence,
                        tracker_id=slot.tracker_id,
                        predicted=True,
                    )
                )
            # else: slot has been gone too long; emit nothing.
        return out

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def slot_summary(self) -> Dict[int, Dict[str, float]]:
        """Return a tidy dict for logging/debugging."""
        return {
            pid: {
                "tracker_id": float(slot.tracker_id) if slot.tracker_id is not None else -1.0,
                "frames_since_seen": float(slot.frames_since_seen),
                "confidence": float(slot.confidence),
            }
            for pid, slot in self._slots.items()
        }
