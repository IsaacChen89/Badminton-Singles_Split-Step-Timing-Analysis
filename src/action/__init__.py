"""Per-player split-step action classifier (CNN-LSTM)."""

__all__ = [
    "SplitStepCNNLSTM",
    "build_model",
    "load_checkpoint",
    "save_checkpoint",
    "SplitStepClipDataset",
    "build_train_transform",
    "build_eval_transform",
    "LabelSmoother",
    "RollingActionInference",
]


def __getattr__(name: str):
    """Lazily import torch-dependent action helpers on first access."""
    if name in {"SplitStepCNNLSTM", "build_model", "load_checkpoint", "save_checkpoint"}:
        from . import model

        return getattr(model, name)
    if name in {"SplitStepClipDataset", "build_train_transform", "build_eval_transform"}:
        from . import dataset

        return getattr(dataset, name)
    if name == "LabelSmoother":
        from .smoothing import LabelSmoother

        return LabelSmoother
    if name == "RollingActionInference":
        from .inference import RollingActionInference

        return RollingActionInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
