"""Rolling-window inference for the split-step CNN-LSTM at video time.

Each player owns its own deque of cropped frames. As soon as the deque is
full (``clip_length``) we forward the stack through the model and return a
fresh probability for that player. Until then we return ``0.0``. Hand the
returned probability to :class:`LabelSmoother` for a stable boolean label.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np
import torch
from PIL import Image

from .dataset import build_eval_transform
from .model import SplitStepCNNLSTM
from ..utils.logging import get_logger

logger = get_logger("action.inference")


class RollingActionInference:
    """Stateful per-player rolling inference.

    Parameters
    ----------
    model:
        Trained :class:`SplitStepCNNLSTM` instance (or untrained — see
        ``warn_if_untrained``).
    clip_length:
        Window size ``T``.
    input_size:
        Square crop size used by the model's transform.
    device:
        ``"cpu"`` or ``"cuda"`` or ``"mps"``.
    stride:
        Run the forward pass every ``stride`` updates (1 = every frame).
    """

    def __init__(
        self,
        model: Optional[SplitStepCNNLSTM],
        clip_length: int = 16,
        input_size: int = 224,
        device: str = "cpu",
        stride: int = 1,
    ) -> None:
        self.model = model.eval().to(device) if model is not None else None
        self.clip_length = clip_length
        self.transform = build_eval_transform(input_size)
        self.device = device
        self.stride = max(1, int(stride))
        self._buffers: Dict[int, Deque[torch.Tensor]] = {}
        self._steps: Dict[int, int] = {}
        self._last_prob: Dict[int, float] = {}

    def reset(self, player_id: int | None = None) -> None:
        if player_id is None:
            self._buffers.clear()
            self._steps.clear()
            self._last_prob.clear()
        else:
            self._buffers.pop(player_id, None)
            self._steps.pop(player_id, None)
            self._last_prob.pop(player_id, None)

    def _ensure(self, player_id: int) -> Deque[torch.Tensor]:
        buf = self._buffers.get(player_id)
        if buf is None:
            buf = deque(maxlen=self.clip_length)
            self._buffers[player_id] = buf
            self._steps[player_id] = 0
            self._last_prob[player_id] = 0.0
        return buf

    @torch.inference_mode()
    def update(self, player_id: int, crop_bgr: np.ndarray) -> float:
        """Push a new BGR crop for ``player_id`` and return the latest split-step probability."""
        if crop_bgr.size == 0:
            return self._last_prob.get(player_id, 0.0)
        rgb = crop_bgr[:, :, ::-1]
        pil = Image.fromarray(rgb)
        tensor = self.transform([pil])[0]  # (3, H, W) on CPU
        buf = self._ensure(player_id)
        buf.append(tensor)
        self._steps[player_id] = self._steps.get(player_id, 0) + 1

        if self.model is None:
            return 0.0
        if len(buf) < self.clip_length:
            return self._last_prob.get(player_id, 0.0)
        if (self._steps[player_id] % self.stride) != 0:
            return self._last_prob.get(player_id, 0.0)

        clip = torch.stack(list(buf), dim=0).unsqueeze(0).to(self.device)
        logits = self.model(clip)
        if logits.size(-1) == 1:
            prob = torch.sigmoid(logits)[0, 0].item()
        else:
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
        self._last_prob[player_id] = float(prob)
        return float(prob)

    def last_probability(self, player_id: int) -> float:
        return self._last_prob.get(player_id, 0.0)
