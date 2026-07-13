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
from typing import List, Literal, Optional, Protocol, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F


SplitName = Literal["train", "val", "test"]


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class FrameTransform(Protocol):
    def __call__(self, image: Image.Image) -> torch.Tensor:
        ...


class ClipTransform(Protocol):
    def __call__(self, images: List[Image.Image]) -> torch.Tensor:
        ...


class ConsistentClipTrainTransform:
    """Apply one sampled augmentation consistently to every frame in a clip."""

    def __init__(
        self,
        input_size: int = 224,
        resize_margin: int = 16,
        horizontal_flip_p: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
    ) -> None:
        self.input_size = input_size
        self.resize_size = input_size + resize_margin
        self.horizontal_flip_p = horizontal_flip_p
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
        )
        self.brightness = self.color_jitter.brightness
        self.contrast = self.color_jitter.contrast
        self.saturation = self.color_jitter.saturation

    def __call__(self, images: List[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("Cannot transform an empty clip.")

        resized = [F.resize(im, [self.resize_size, self.resize_size]) for im in images]
        crop_params = T.RandomCrop.get_params(
            resized[0], output_size=(self.input_size, self.input_size)
        )
        do_flip = bool(torch.rand(()) < self.horizontal_flip_p)
        brightness_factor = _sample_jitter_factor(self.brightness)
        contrast_factor = _sample_jitter_factor(self.contrast)
        saturation_factor = _sample_jitter_factor(self.saturation)
        jitter_order = torch.randperm(3).tolist()

        tensors: List[torch.Tensor] = []
        for im in resized:
            im = F.crop(im, *crop_params)
            if do_flip:
                im = F.hflip(im)
            im = _apply_color_jitter(
                im,
                jitter_order,
                brightness_factor,
                contrast_factor,
                saturation_factor,
            )
            tensor = F.to_tensor(im)
            tensor = F.normalize(tensor, _IMAGENET_MEAN, _IMAGENET_STD)
            tensors.append(tensor)
        return torch.stack(tensors, dim=0)


class PerFrameClipTransform:
    """Wrap a per-frame transform so the dataset returns a clip tensor."""

    def __init__(self, frame_transform: FrameTransform) -> None:
        self.frame_transform = frame_transform

    def __call__(self, images: List[Image.Image]) -> torch.Tensor:
        return torch.stack([self.frame_transform(im) for im in images], dim=0)


def build_train_transform(input_size: int = 224) -> ClipTransform:
    return ConsistentClipTrainTransform(input_size=input_size)


def _sample_jitter_factor(bounds: Optional[Tuple[float, float]]) -> Optional[float]:
    if bounds is None:
        return None
    low, high = bounds
    return float(torch.empty(1).uniform_(float(low), float(high)).item())


def _apply_color_jitter(
    image: Image.Image,
    order: List[int],
    brightness: Optional[float],
    contrast: Optional[float],
    saturation: Optional[float],
) -> Image.Image:
    for op in order:
        if op == 0 and brightness is not None:
            image = F.adjust_brightness(image, brightness)
        elif op == 1 and contrast is not None:
            image = F.adjust_contrast(image, contrast)
        elif op == 2 and saturation is not None:
            image = F.adjust_saturation(image, saturation)
    return image


def build_eval_transform(input_size: int = 224) -> ClipTransform:
    frame_transform = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )
    return PerFrameClipTransform(frame_transform)


@dataclass
class ClipRecord:
    clip_id: str
    video: str
    player_id: int
    center_frame: int
    label: int
    target: float
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
        Clip transform (``List[PIL.Image] -> torch.Tensor``). The training
        transform samples one crop/flip/color jitter and applies it to every
        frame in the clip.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: SplitName = "train",
        root: Optional[str | Path] = None,
        clip_length: int = 16,
        transform: Optional[ClipTransform] = None,
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
                target=float(r["target"]) if "target" in df.columns else float(r["label"]),
                split=split,
            )
            for _, r in df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.records)

    @property
    def labels(self) -> np.ndarray:
        return np.array([r.label for r in self.records], dtype=np.int64)

    @property
    def targets(self) -> np.ndarray:
        return np.array([r.target for r in self.records], dtype=np.float32)

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
        images: List[Image.Image] = []
        for fp in frame_paths:
            with Image.open(fp) as im:
                im = im.convert("RGB")
                images.append(im.copy())
        # Pad at the front by repeating the first frame if too short.
        while len(images) < self.clip_length:
            images.insert(0, images[0].copy())
        return self.transform(images)  # (T, 3, H, W)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        rec = self.records[idx]
        clip = self._load_clip(rec.clip_id)
        target = torch.tensor(rec.target, dtype=torch.float32)
        return clip, target, rec.label
