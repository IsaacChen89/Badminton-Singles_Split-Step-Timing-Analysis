"""Action-clip dataset reading manifests produced by ``cvat_converter``.

Manifest schema (CSV with header):

    clip_id, video, player_id, center_frame, label, split

``label`` is ``0`` (normal) or ``1`` (split_step). ``split`` is ``train``,
``val``, or ``test``. The matching frame images live at::

    <root>/clips/<clip_id>/<frame_idx>.jpg

with exactly ``clip_length`` frames in lexicographic order.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterator, List, Literal, Optional, Protocol, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
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
    """Apply spatial/appearance augmentation consistently across a clip.

    Temporal frame dropping is simulated by repeating an earlier frame while
    preserving the final frame that owns the clip label.
    """

    def __init__(
        self,
        input_size: int = 224,
        resize_margin: int = 16,
        horizontal_flip_p: float = 0.5,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        bbox_translate: float = 0.05,
        bbox_scale_min: float = 0.9,
        bbox_scale_max: float = 1.1,
        blur_probability: float = 0.15,
        blur_sigma_min: float = 0.1,
        blur_sigma_max: float = 1.5,
        jpeg_probability: float = 0.15,
        jpeg_quality_min: int = 60,
        jpeg_quality_max: int = 95,
        frame_drop_probability: float = 0.05,
    ) -> None:
        _validate_probability("horizontal_flip_p", horizontal_flip_p)
        _validate_probability("blur_probability", blur_probability)
        _validate_probability("jpeg_probability", jpeg_probability)
        _validate_probability("frame_drop_probability", frame_drop_probability)
        if bbox_translate < 0:
            raise ValueError("bbox_translate must be >= 0.")
        if bbox_scale_min <= 0 or bbox_scale_min > bbox_scale_max:
            raise ValueError("bbox scales must satisfy 0 < min <= max.")
        if blur_sigma_min <= 0 or blur_sigma_min > blur_sigma_max:
            raise ValueError("blur sigmas must satisfy 0 < min <= max.")
        if not (1 <= jpeg_quality_min <= jpeg_quality_max <= 100):
            raise ValueError("JPEG quality must satisfy 1 <= min <= max <= 100.")

        self.input_size = input_size
        self.resize_size = input_size + resize_margin
        self.horizontal_flip_p = horizontal_flip_p
        self.bbox_translate = bbox_translate
        self.bbox_scale_min = bbox_scale_min
        self.bbox_scale_max = bbox_scale_max
        self.blur_probability = blur_probability
        self.blur_sigma_min = blur_sigma_min
        self.blur_sigma_max = blur_sigma_max
        self.jpeg_probability = jpeg_probability
        self.jpeg_quality_min = jpeg_quality_min
        self.jpeg_quality_max = jpeg_quality_max
        self.frame_drop_probability = frame_drop_probability
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

        images = _simulate_dropped_frames(images, self.frame_drop_probability)
        translate_x = _sample_uniform(-self.bbox_translate, self.bbox_translate)
        translate_y = _sample_uniform(-self.bbox_translate, self.bbox_translate)
        bbox_scale = _sample_uniform(self.bbox_scale_min, self.bbox_scale_max)
        apply_blur = bool(torch.rand(()) < self.blur_probability)
        blur_sigma = _sample_uniform(self.blur_sigma_min, self.blur_sigma_max)
        apply_jpeg = bool(torch.rand(()) < self.jpeg_probability)
        jpeg_quality = int(
            torch.randint(self.jpeg_quality_min, self.jpeg_quality_max + 1, ()).item()
        )
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
            width, height = im.size
            im = F.affine(
                im,
                angle=0.0,
                translate=[
                    int(round(translate_x * width)),
                    int(round(translate_y * height)),
                ],
                scale=bbox_scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=tuple(round(channel * 255) for channel in _IMAGENET_MEAN),
            )
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
            if apply_blur:
                im = F.gaussian_blur(
                    im,
                    kernel_size=[5, 5],
                    sigma=[blur_sigma, blur_sigma],
                )
            if apply_jpeg:
                im = _jpeg_reencode(im, jpeg_quality)
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


def build_train_transform(
    input_size: int = 224,
    *,
    bbox_translate: float = 0.05,
    bbox_scale_min: float = 0.9,
    bbox_scale_max: float = 1.1,
    blur_probability: float = 0.15,
    blur_sigma_min: float = 0.1,
    blur_sigma_max: float = 1.5,
    jpeg_probability: float = 0.15,
    jpeg_quality_min: int = 60,
    jpeg_quality_max: int = 95,
    frame_drop_probability: float = 0.05,
) -> ClipTransform:
    return ConsistentClipTrainTransform(
        input_size=input_size,
        bbox_translate=bbox_translate,
        bbox_scale_min=bbox_scale_min,
        bbox_scale_max=bbox_scale_max,
        blur_probability=blur_probability,
        blur_sigma_min=blur_sigma_min,
        blur_sigma_max=blur_sigma_max,
        jpeg_probability=jpeg_probability,
        jpeg_quality_min=jpeg_quality_min,
        jpeg_quality_max=jpeg_quality_max,
        frame_drop_probability=frame_drop_probability,
    )


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def _sample_uniform(low: float, high: float) -> float:
    if low == high:
        return float(low)
    return float(torch.empty(1).uniform_(float(low), float(high)).item())


def _simulate_dropped_frames(
    images: List[Image.Image], probability: float
) -> List[Image.Image]:
    """Repeat preceding interior frames to mimic detector/video frame loss."""
    if probability <= 0 or len(images) < 3:
        return images
    augmented = list(images)
    for index in range(1, len(images) - 1):
        if bool(torch.rand(()) < probability):
            augmented[index] = augmented[index - 1]
    return augmented


def _jpeg_reencode(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as encoded:
        return encoded.convert("RGB").copy()


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


class EventBalancedBatchSampler(Sampler[List[int]]):
    """Build batches from unique positive events and diverse hard negatives."""

    def __init__(
        self,
        records: Sequence[ClipRecord],
        batch_size: int,
        *,
        positive_fraction: float = 0.35,
        boundary_negative_fraction: float = 0.25,
        event_gap_frames: int = 4,
        boundary_radius_frames: int = 8,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        _validate_probability("positive_fraction", positive_fraction)
        _validate_probability(
            "boundary_negative_fraction",
            boundary_negative_fraction,
        )
        if positive_fraction + boundary_negative_fraction > 1.0:
            raise ValueError(
                "positive_fraction + boundary_negative_fraction must be <= 1."
            )
        if event_gap_frames <= 0:
            raise ValueError("event_gap_frames must be > 0.")
        if boundary_radius_frames < 0:
            raise ValueError("boundary_radius_frames must be >= 0.")

        self.records = list(records)
        self.batch_size = int(batch_size)
        self.positive_fraction = float(positive_fraction)
        self.boundary_negative_fraction = float(boundary_negative_fraction)
        self.event_gap_frames = int(event_gap_frames)
        self.boundary_radius_frames = int(boundary_radius_frames)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        grouped: dict[tuple[str, int], List[int]] = defaultdict(list)
        for index, record in enumerate(self.records):
            grouped[(record.video, record.player_id)].append(index)
        for indices in grouped.values():
            indices.sort(key=lambda index: self.records[index].center_frame)

        self.positive_events: List[List[int]] = []
        positive_frames: dict[tuple[str, int], List[int]] = {}
        for key, indices in grouped.items():
            positive_indices = [
                index for index in indices if self.records[index].label == 1
            ]
            positive_frames[key] = [
                self.records[index].center_frame for index in positive_indices
            ]
            current_event: List[int] = []
            previous_frame: Optional[int] = None
            for index in positive_indices:
                frame = self.records[index].center_frame
                if (
                    current_event
                    and previous_frame is not None
                    and frame - previous_frame > self.event_gap_frames
                ):
                    self.positive_events.append(current_event)
                    current_event = []
                current_event.append(index)
                previous_frame = frame
            if current_event:
                self.positive_events.append(current_event)

        boundary_by_video: dict[str, List[int]] = defaultdict(list)
        random_by_video: dict[str, List[int]] = defaultdict(list)
        for key, indices in grouped.items():
            frames = positive_frames[key]
            for index in indices:
                record = self.records[index]
                if record.label == 1:
                    continue
                if _is_near_positive(
                    record.center_frame,
                    frames,
                    self.boundary_radius_frames,
                ):
                    boundary_by_video[record.video].append(index)
                else:
                    random_by_video[record.video].append(index)

        self.boundary_by_video = dict(boundary_by_video)
        self.random_by_video = dict(random_by_video)
        self.all_indices = list(range(len(self.records)))

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.records) // self.batch_size
        return math.ceil(len(self.records) / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        total_batches = len(self)
        for batch_index in range(total_batches):
            remaining = len(self.records) - batch_index * self.batch_size
            size = self.batch_size if self.drop_last else min(self.batch_size, remaining)
            positive_count = min(size, round(size * self.positive_fraction))
            boundary_count = min(
                size - positive_count,
                round(size * self.boundary_negative_fraction),
            )
            random_count = size - positive_count - boundary_count

            used: set[int] = set()
            batch = self._sample_positive_events(positive_count, rng, used)
            batch.extend(
                _sample_video_balanced(
                    self.boundary_by_video,
                    boundary_count,
                    rng,
                    used,
                )
            )
            batch.extend(
                _sample_video_balanced(
                    self.random_by_video,
                    random_count,
                    rng,
                    used,
                )
            )
            if len(batch) < size:
                available = [index for index in self.all_indices if index not in used]
                rng.shuffle(available)
                batch.extend(available[: size - len(batch)])
            rng.shuffle(batch)
            yield batch

    def _sample_positive_events(
        self,
        count: int,
        rng: random.Random,
        used: set[int],
    ) -> List[int]:
        if count <= 0 or not self.positive_events:
            return []
        if count <= len(self.positive_events):
            events = rng.sample(self.positive_events, count)
        else:
            events = list(self.positive_events)
            events.extend(
                rng.choice(self.positive_events)
                for _ in range(count - len(self.positive_events))
            )
            rng.shuffle(events)
        selected: List[int] = []
        for event in events:
            candidates = [index for index in event if index not in used]
            if not candidates:
                continue
            index = rng.choice(candidates)
            used.add(index)
            selected.append(index)
        return selected


def _is_near_positive(frame: int, positive_frames: Sequence[int], radius: int) -> bool:
    if not positive_frames:
        return False
    position = bisect_left(positive_frames, frame)
    neighbors = positive_frames[max(0, position - 1) : position + 1]
    return any(abs(frame - positive) <= radius for positive in neighbors)


def _sample_video_balanced(
    pools: dict[str, List[int]],
    count: int,
    rng: random.Random,
    used: set[int],
) -> List[int]:
    """Sample clips while giving each available video equal probability."""
    if count <= 0 or not pools:
        return []
    videos = list(pools)
    selected: List[int] = []
    attempts = 0
    max_attempts = max(20, count * 20)
    while len(selected) < count and attempts < max_attempts:
        attempts += 1
        video = rng.choice(videos)
        candidates = [index for index in pools[video] if index not in used]
        if not candidates:
            continue
        index = rng.choice(candidates)
        used.add(index)
        selected.append(index)
    return selected


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
