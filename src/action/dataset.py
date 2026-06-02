"""Action-clip dataset reading manifests produced by ``cvat_converter``.

Manifest schema (CSV with header):

    clip_id, video, player_id, center_frame, label, split

``label`` is ``0`` (normal) or ``1`` (split_step). ``split`` is ``train``,
``val``, or ``test``. The matching frame images live at::

    <root>/clips/<clip_id>/<frame_idx>.jpg

with exactly ``clip_length`` frames in lexicographic order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


SplitName = Literal["train", "val", "test"]


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(input_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((input_size + 16, input_size + 16)),
            T.RandomCrop(input_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def build_eval_transform(input_size: int = 224) -> T.Compose:
    return T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


@dataclass
class ClipRecord:
    clip_id: str
    video: str
    player_id: int
    center_frame: int
    label: int
    split: SplitName


class SplitStepClipDataset(Dataset):
    """PyTorch dataset for split-step clips.

    Parameters
    ----------
    manifest_path:
        Path to ``manifest.csv``.
    split:
        ``"train"``, ``"val"``, or ``"test"``. Filters rows by the ``split``
        column of the manifest.
    root:
        Root dataset directory; defaults to the manifest's parent.
    clip_length:
        Number of frames per clip. Frames are zero-padded if missing.
    transform:
        Per-frame transform (``PIL.Image -> torch.Tensor``).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: SplitName = "train",
        root: Optional[str | Path] = None,
        clip_length: int = 16,
        transform: Optional[T.Compose] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        self.root = Path(root) if root is not None else self.manifest_path.parent
        self.clip_length = clip_length
        self.transform = transform or build_eval_transform()

        df = pd.read_csv(self.manifest_path)
        df = df[df["split"] == split].reset_index(drop=True)
        self.records: List[ClipRecord] = [
            ClipRecord(
                clip_id=str(r["clip_id"]),
                video=str(r["video"]),
                player_id=int(r["player_id"]),
                center_frame=int(r["center_frame"]),
                label=int(r["label"]),
                split=split,
            )
            for _, r in df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> np.ndarray:
        return np.array([r.label for r in self.records], dtype=np.int64)

    def _load_clip(self, clip_id: str) -> torch.Tensor:
        clip_dir = self.root / "clips" / clip_id
        if not clip_dir.exists():
            raise FileNotFoundError(f"Clip directory missing: {clip_dir}")
        frame_paths = sorted(clip_dir.glob("*.jpg"))
        if not frame_paths:
            raise RuntimeError(f"No frames in {clip_dir}")
        # If we have more than needed, take the trailing window so the label
        # corresponds to the most recent frame.
        if len(frame_paths) > self.clip_length:
            frame_paths = frame_paths[-self.clip_length:]
        tensors: List[torch.Tensor] = []
        for fp in frame_paths:
            with Image.open(fp) as im:
                im = im.convert("RGB")
                tensors.append(self.transform(im))
        # Pad at the front by repeating the first frame if too short.
        while len(tensors) < self.clip_length:
            tensors.insert(0, tensors[0].clone())
        return torch.stack(tensors, dim=0)  # (T, 3, H, W)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        clip = self._load_clip(rec.clip_id)
        return clip, rec.label
