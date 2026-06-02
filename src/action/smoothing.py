"""Per-player label smoother.

Combines an exponential moving average over the raw split-step probability
with hysteresis (separate ON/OFF thresholds) and minimum dwell times for
both the ON and OFF states. The result is a stable boolean label without
the typical sub-second flicker around the actual split step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class _PlayerState:
    ema: float = 0.0
    on: bool = False
    on_streak: int = 0
    off_streak: int = 0


class LabelSmoother:
    """Smoother for per-frame split-step probabilities.

    Parameters
    ----------
    ema_alpha:
        Weight for the new sample in the EMA, in ``(0, 1]``.
    prob_on:
        Probability that triggers the ``ON`` state once sustained.
    prob_off:
        Probability that triggers the ``OFF`` state once sustained.
    min_on_frames:
        Number of consecutive high-probability frames required to flip ON.
    cooldown_frames:
        Number of consecutive low-probability frames required to flip OFF.
    """

    def __init__(
        self,
        ema_alpha: float = 0.4,
        prob_on: float = 0.6,
        prob_off: float = 0.4,
        min_on_frames: int = 3,
        cooldown_frames: int = 6,
    ) -> None:
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be in (0, 1]")
        if prob_on < prob_off:
            raise ValueError("prob_on must be >= prob_off (hysteresis)")
        self.ema_alpha = ema_alpha
        self.prob_on = prob_on
        self.prob_off = prob_off
        self.min_on_frames = max(1, int(min_on_frames))
        self.cooldown_frames = max(1, int(cooldown_frames))
        self._state: Dict[int, _PlayerState] = {}

    def reset(self, player_id: int | None = None) -> None:
        if player_id is None:
            self._state.clear()
        else:
            self._state.pop(player_id, None)

    def update(self, player_id: int, raw_prob: float) -> tuple[bool, float]:
        """Update with the raw probability for ``player_id``.

        Returns ``(is_split_step, smoothed_probability)``.
        """
        st = self._state.setdefault(player_id, _PlayerState())
        st.ema = self.ema_alpha * raw_prob + (1.0 - self.ema_alpha) * st.ema

        if st.ema >= self.prob_on:
            st.on_streak += 1
            st.off_streak = 0
        elif st.ema <= self.prob_off:
            st.off_streak += 1
            st.on_streak = 0
        else:
            # In the dead-band: don't accumulate either streak.
            st.on_streak = max(0, st.on_streak - 1)
            st.off_streak = max(0, st.off_streak - 1)

        if not st.on and st.on_streak >= self.min_on_frames:
            st.on = True
        elif st.on and st.off_streak >= self.cooldown_frames:
            st.on = False

        return st.on, st.ema

    def state(self, player_id: int) -> _PlayerState:
        return self._state.setdefault(player_id, _PlayerState())
