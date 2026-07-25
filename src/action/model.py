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
from typing import Any, Dict, Iterable, List, Optional, Sequence

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

_RESNET_STAGE_NAMES = ("conv1", "layer1", "layer2", "layer3", "layer4")


def _normalize_backbone_stages(stages: Optional[Iterable[str]]) -> List[str]:
    """Validate and de-duplicate ResNet stage names."""
    if stages is None:
        return []
    normalized: List[str] = []
    for stage in stages:
        name = str(stage).strip().lower()
        if not name:
            continue
        if name not in _RESNET_STAGE_NAMES:
            raise ValueError(
                f"Unknown backbone stage '{stage}'. "
                f"Supported: {list(_RESNET_STAGE_NAMES)}"
            )
        if name not in normalized:
            normalized.append(name)
    return normalized


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
        freeze_backbone_stages: Optional[Sequence[str]] = None,
        freeze_batchnorm_stats: bool = False,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.backbone, feat_dim = _build_backbone(backbone_name)
        self.feat_dim = feat_dim
        self.freeze_batchnorm_stats = bool(freeze_batchnorm_stats)
        self.temperature = float(temperature)
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

        stages = _normalize_backbone_stages(freeze_backbone_stages)
        self.config = {
            "backbone": backbone_name,
            "num_classes": num_classes,
            "lstm_hidden": lstm_hidden,
            "lstm_layers": lstm_layers,
            "bidirectional": bidirectional,
            "dropout": dropout,
            "feature_dropout": feature_dropout,
            "freeze_backbone": freeze_backbone,
            "freeze_backbone_stages": stages,
            "freeze_batchnorm_stats": self.freeze_batchnorm_stats,
            "temperature": self.temperature,
        }

        self.apply_freeze_backbone(freeze_backbone)
        if not freeze_backbone:
            self.apply_freeze_backbone_stages(stages)

    def train(self, mode: bool = True) -> "SplitStepCNNLSTM":
        """Set training mode while optionally keeping backbone BN stats fixed."""
        super().train(mode)
        if mode and self.freeze_batchnorm_stats:
            for module in self.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def apply_freeze_backbone(self, freeze: bool) -> None:
        """Toggle full backbone gradient flow and persist the choice in ``config``."""
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        self.config["freeze_backbone"] = bool(freeze)
        if not freeze:
            # Re-apply any configured partial stage freeze after unfreezing.
            self._set_stage_requires_grad(
                _normalize_backbone_stages(self.config.get("freeze_backbone_stages"))
            )

    def apply_freeze_backbone_stages(
        self,
        stages: Optional[Sequence[str]] = None,
    ) -> None:
        """Freeze selected ResNet stages while leaving later stages trainable.

        Ignored when ``freeze_backbone`` is already true (entire backbone frozen).
        ``conv1`` also freezes ``bn1``. Only supported for ResNet backbones.
        """
        normalized = _normalize_backbone_stages(stages)
        self.config["freeze_backbone_stages"] = list(normalized)
        if self.config.get("freeze_backbone"):
            return
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._set_stage_requires_grad(normalized)

    def _set_stage_requires_grad(self, stages: Sequence[str]) -> None:
        if not stages:
            return
        backbone_name = str(self.config.get("backbone", ""))
        if not backbone_name.startswith("resnet"):
            raise ValueError(
                "freeze_backbone_stages is only supported for ResNet backbones "
                f"(got '{backbone_name}')."
            )
        for stage in stages:
            for module in self._stage_modules(stage):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def _stage_modules(self, stage: str) -> List[nn.Module]:
        if stage == "conv1":
            modules: List[nn.Module] = [self.backbone.conv1]
            if hasattr(self.backbone, "bn1"):
                modules.append(self.backbone.bn1)
            return modules
        return [getattr(self.backbone, stage)]

    def apply_freeze_batchnorm_stats(self, freeze: bool) -> None:
        """Toggle automatic freezing of backbone BatchNorm running statistics."""
        self.freeze_batchnorm_stats = bool(freeze)
        self.config["freeze_batchnorm_stats"] = self.freeze_batchnorm_stats
        if self.training:
            self.train(True)

    def set_temperature(self, temperature: float) -> None:
        """Set positive post-training logit temperature."""
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.temperature = float(temperature)
        self.config["temperature"] = self.temperature

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
        # EMA/deep-copied LSTM weights may not retain cuDNN's packed layout.
        # This is a no-op once contiguous and avoids repacking on every call.
        self.lstm.flatten_parameters()
        _, (h_n, _) = self.lstm(feats)
        if self.lstm.bidirectional:
            # The last two states are the final forward/backward states of the
            # last LSTM layer, so both summarize the complete trailing clip.
            sequence_summary = torch.cat((h_n[-2], h_n[-1]), dim=-1)
        else:
            sequence_summary = h_n[-1]
        return self.head(sequence_summary) / self.temperature


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
        freeze_backbone_stages=cfg.freeze_backbone_stages,
        freeze_batchnorm_stats=cfg.freeze_batchnorm_stats,
        temperature=1.0,
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
        freeze_backbone_stages=cfg.get("freeze_backbone_stages", []),
        freeze_batchnorm_stats=cfg.get("freeze_batchnorm_stats", False),
        temperature=cfg.get("temperature", 1.0),
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
