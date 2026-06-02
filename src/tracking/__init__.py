"""Multi-object tracking + per-player ID assignment."""

from .tracker import PlayerTracker, resolve_tracker_yaml
from .player_assigner import (
    AssignedPlayer,
    PlayerAssigner,
    PlayerPosition,
    TrackingMode,
    PLAYER1_ID,
    PLAYER2_ID,
)

__all__ = [
    "PlayerTracker",
    "resolve_tracker_yaml",
    "PlayerAssigner",
    "AssignedPlayer",
    "PlayerPosition",
    "TrackingMode",
    "PLAYER1_ID",
    "PLAYER2_ID",
]
