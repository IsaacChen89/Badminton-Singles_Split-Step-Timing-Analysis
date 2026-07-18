"""Lightweight CNN-LSTM split-step classifier.

The model takes a clip of shape ``(B, T, 3, H, W)`` of cropped player ROIs,
runs a frozen-by-default ResNet18 backbone per frame, then a (Bi)LSTM over
the time axis, and predicts logits for the clip. Current BCE training uses a
single split-step logit; older checkpoints may use a 2-class logits vector
(``normal`` vs ``split_step``). At inference we treat the prediction as the
label of the *last* frame in the rolling window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torchvision import models as tvm

from ..utils.config import ActionConfig
from ..utils.logging import get_logger

logger = get_logger("action.model")


_BACKBONE_FACTORIES = {
    "resnet18": (tvm.resnet18, 512),
    "resnet34": (tvm.resnet34, 512),
    "mobilenet_v3_small": (tvm.mobilenet_v3_small, 576),
}


def _build_backbone(name: str) -> tuple[nn.Module, int]:
    if name not in _BACKBONE_FACTORIES:
        raise ValueError(f"Unsupported backbone '{name}'. Choices: {list(_BACKBONE_FACTORIES)}")
    factory, feat_dim = _BACKBONE_FACTORIES[name]
    # ``weights=None`` keeps offline-friendliness; users can call
    # ``model.load_imagenet_weights()`` after the fact if they want to.
    backbone = factory(weights=None)
    if name.startswith("resnet"):
        backbone.fc = nn.Identity()
    elif name.startswith("mobilenet"):
        backbone.classifier = nn.Identity()
    return backbone, feat_dim


class SplitStepCNNLSTM(nn.Module):
    """ResNet18 + (Bi)LSTM split-step classifier."""

    def __init__(
        self,
        backbone_name: str = "resnet18",
        num_classes: int = 2,
        lstm_hidden: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        feature_dropout: float = 0.25,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = _build_backbone(backbone_name)
        self.feat_dim = feat_dim
        # Regularize CNN features before the temporal model (head dropout is separate).
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

        self.config = {
            "backbone": backbone_name,
            "num_classes": num_classes,
            "lstm_hidden": lstm_hidden,
            "lstm_layers": lstm_layers,
            "bidirectional": bidirectional,
            "dropout": dropout,
            "feature_dropout": feature_dropout,
            "freeze_backbone": freeze_backbone,
        }

        self.apply_freeze_backbone(freeze_backbone)

    def apply_freeze_backbone(self, freeze: bool) -> None:
        """Toggle backbone gradient flow and persist the choice in ``config``."""
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        self.config["freeze_backbone"] = bool(freeze)

    def try_load_imagenet_weights(self) -> bool:
        """Best-effort load of ImageNet weights for the backbone.

        Returns ``True`` when successful, ``False`` if torchvision can't reach
        its weight cache (e.g. offline). Existing parameters are left
        untouched on failure.
        """
        name = self.config["backbone"]
        try:
            if name == "resnet18":
                weights = tvm.ResNet18_Weights.DEFAULT
                ref = tvm.resnet18(weights=weights)
                ref.fc = nn.Identity()
            elif name == "resnet34":
                weights = tvm.ResNet34_Weights.DEFAULT
                ref = tvm.resnet34(weights=weights)
                ref.fc = nn.Identity()
            elif name == "mobilenet_v3_small":
                weights = tvm.MobileNet_V3_Small_Weights.DEFAULT
                ref = tvm.mobilenet_v3_small(weights=weights)
                ref.classifier = nn.Identity()
            else:
                return False
            self.backbone.load_state_dict(ref.state_dict())
            logger.info(f"Loaded ImageNet weights for backbone '{name}'.")
            return True
        except Exception as exc:  # pragma: no cover - depends on internet
            logger.warning(f"Could not load ImageNet weights for '{name}': {exc}")
            return False

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        clips: ``(B, T, 3, H, W)``

        Returns
        -------
        logits: ``(B, num_classes)`` for the *last* timestep.
        """
        if clips.ndim != 5:
            raise ValueError(f"Expected (B,T,3,H,W); got shape {tuple(clips.shape)}")
        b, t, c, h, w = clips.shape
        feats = self.backbone(clips.view(b * t, c, h, w))
        feats = feats.view(b, t, -1)
        feats = self.feature_dropout(feats)
        out, _ = self.lstm(feats)
        # Predict from the final timestep -> labels the most recent frame.
        last = out[:, -1, :]
        return self.head(last)


def build_model(cfg: ActionConfig) -> SplitStepCNNLSTM:
    """Construct a model from an :class:`ActionConfig`."""
    return SplitStepCNNLSTM(
        backbone_name=cfg.backbone,
        num_classes=cfg.num_classes,
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
        bidirectional=cfg.bidirectional,
        dropout=cfg.dropout,
        feature_dropout=cfg.feature_dropout,
        freeze_backbone=cfg.freeze_backbone,
    )


def save_checkpoint(
    model: SplitStepCNNLSTM,
    path: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "config": dict(model.config),
        "extra": extra or {},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path,
    map_location: str = "cpu",
) -> SplitStepCNNLSTM:
    """Reconstruct a model from a checkpoint produced by :func:`save_checkpoint`."""
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    cfg = payload["config"]
    model = SplitStepCNNLSTM(
        backbone_name=cfg.get("backbone", "resnet18"),
        num_classes=cfg.get("num_classes", 2),
        lstm_hidden=cfg.get("lstm_hidden", 128),
        lstm_layers=cfg.get("lstm_layers", 1),
        bidirectional=cfg.get("bidirectional", True),
        dropout=cfg.get("dropout", 0.2),
        feature_dropout=cfg.get("feature_dropout", 0.0),
        freeze_backbone=cfg.get("freeze_backbone", True),
    )
    model.load_state_dict(payload["model_state"])
    return model


def model_summary(model: SplitStepCNNLSTM) -> str:
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"SplitStepCNNLSTM(params={n_params:,}, trainable={n_trainable:,}, "
        f"cfg={model.config})"
    )
