"""Drawing helpers: colored player boxes, ``SPLIT STEP`` labels, HUD.

All drawing is in BGR pixel space (OpenCV convention). The two players have
fixed colors so the output is consistent across runs:

- Player 1 -> RED  ``(0, 0, 255)``
- Player 2 -> BLUE ``(255, 0, 0)``
"""

from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from ..tracking.player_assigner import AssignedPlayer, PLAYER1_ID, PLAYER2_ID


# BGR
PLAYER_COLORS = {
    PLAYER1_ID: (0, 0, 255),    # red
    PLAYER2_ID: (255, 0, 0),    # blue
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _player_label(player_id: int) -> str:
    return f"Player {player_id}"


def draw_player_box(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    player_id: int,
    score: float | None = None,
    tracker_id: Optional[int] = None,
    show_tracker_id: bool = False,
    predicted: bool = False,
    thickness: int = 2,
) -> None:
    """Draw a colored bounding box + ``Player N`` tag.

    When ``predicted`` is True (no fresh detection on this frame, the box is
    being held over from a trajectory prediction), the box is drawn dashed
    so it's visually distinguishable from a real detection.
    """
    color = PLAYER_COLORS.get(player_id, (255, 255, 255))
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    if predicted:
        _draw_dashed_rect(frame, (x1, y1), (x2, y2), color, thickness)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    label = _player_label(player_id)
    if show_tracker_id and tracker_id is not None and tracker_id >= 0:
        label = f"{label} #{tracker_id}"
    elif predicted:
        label = f"{label} (pred)"
    _draw_text(
        frame,
        label,
        org=(x1, max(0, y1 - 6)),
        color=color,
        scale=0.9,
        thickness=2,
    )


def draw_split_step_label(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    player_id: int,
    probability: float | None = None,
) -> None:
    """Draw a prominent ``SPLIT STEP`` label next to a player's bbox."""
    color = PLAYER_COLORS.get(player_id, (0, 255, 255))
    x1, y1, x2, _y2 = (int(round(v)) for v in bbox)
    text = "SPLIT STEP"

    # Place to the right of the bbox by default; fall back to inside if it
    # would clip beyond the frame.
    org_x = x2 + 8
    org_y = max(0, y1 + 28)
    if org_x + 320 > frame.shape[1]:
        org_x = max(0, x1 + 4)
        org_y = max(28, y1 + 28)

    _draw_text(
        frame,
        text,
        org=(org_x, org_y),
        color=color,
        scale=1.2,
        thickness=3,
    )


def draw_hud(
    frame: np.ndarray,
    *,
    frame_idx: int,
    time_seconds: float,
    fps: float | None = None,
    tracking_mode: Optional[str] = None,
) -> None:
    """Top-left HUD: frame, time, optional FPS, optional tracking mode."""
    parts = [f"frame {frame_idx}", f"t {time_seconds:6.2f}s"]
    if fps is not None and fps > 0:
        parts.append(f"{fps:5.1f} FPS")
    if tracking_mode:
        parts.append(f"track:{tracking_mode}")
    text = "   ".join(parts)
    _draw_text(
        frame, text, org=(10, 32), color=(255, 255, 255), scale=0.85, thickness=2
    )


def render_annotated_frame(
    frame: np.ndarray,
    assigned: Iterable[AssignedPlayer],
    split_step_flags: dict[int, bool],
    split_probs: dict[int, float] | None = None,
    *,
    frame_idx: int = 0,
    time_seconds: float = 0.0,
    runtime_fps: float | None = None,
    show_hud: bool = True,
    show_tracker_ids: bool = False,
    tracking_mode: Optional[str] = None,
) -> np.ndarray:
    """One-shot draw of all overlays for a frame.

    The input frame is *modified in place* and returned for convenience.
    """
    split_probs = split_probs or {}
    for ap in assigned:
        draw_player_box(
            frame,
            ap.bbox,
            ap.player_id,
            tracker_id=ap.tracker_id,
            show_tracker_id=show_tracker_ids,
            predicted=ap.predicted,
        )
        if split_step_flags.get(ap.player_id, False):
            draw_split_step_label(frame, ap.bbox, ap.player_id)
    if show_hud:
        draw_hud(
            frame,
            frame_idx=frame_idx,
            time_seconds=time_seconds,
            fps=runtime_fps,
            tracking_mode=tracking_mode,
        )
    return frame


def _draw_dashed_rect(
    frame: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_length: int = 8,
    gap_length: int = 6,
) -> None:
    """Draw a rectangle with dashed edges."""
    x1, y1 = pt1
    x2, y2 = pt2

    def _dash_line(p1: tuple[int, int], p2: tuple[int, int]) -> None:
        x_a, y_a = p1
        x_b, y_b = p2
        dx, dy = x_b - x_a, y_b - y_a
        dist = max(1.0, (dx * dx + dy * dy) ** 0.5)
        step = dash_length + gap_length
        n = int(dist // step) + 1
        for k in range(n):
            t0 = (k * step) / dist
            t1 = min(1.0, (k * step + dash_length) / dist)
            sx = int(round(x_a + dx * t0))
            sy = int(round(y_a + dy * t0))
            ex = int(round(x_a + dx * t1))
            ey = int(round(y_a + dy * t1))
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)

    _dash_line((x1, y1), (x2, y1))
    _dash_line((x2, y1), (x2, y2))
    _dash_line((x2, y2), (x1, y2))
    _dash_line((x1, y2), (x1, y1))


def _draw_text(
    frame: np.ndarray,
    text: str,
    *,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.6,
    thickness: int = 1,
) -> None:
    """Draw text with a dark outline so it stays readable on any background."""
    x, y = org
    outline_thickness = thickness + 3
    cv2.putText(
        frame, text, (x, y), _FONT, scale, (0, 0, 0), outline_thickness, cv2.LINE_AA
    )
    cv2.putText(frame, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)
