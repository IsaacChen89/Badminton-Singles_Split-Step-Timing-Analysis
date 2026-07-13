"""CVAT for Video 1.1 -> training datasets.

This module is the bridge between human labelers (using CVAT) and our two
training pipelines:

1. **YOLO detection dataset** — fine-tunes the player detector on this exact
   broadcast angle. Output layout::

       <out>/
         data.yaml
         images/{train,val}/<stem>_f<frame>.jpg
         labels/{train,val}/<stem>_f<frame>.txt   # YOLO: cls cx cy w h

2. **Action dataset** — clips of cropped player ROIs labeled with the
   ``split_step`` attribute. Output layout::

       <out>/
         manifest.csv          # clip_id, video, player_id, center_frame, label, split
         clips/<clip_id>/<frame_idx>.jpg ...

Inputs
------
The converter accepts **either** the raw ``annotations.xml`` **or** the full
**CVAT for Video 1.1 .zip** export — the zip is unpacked transparently.

Convention
~~~~~~~~~~
Place files like::

    data/raw/rally_001.mp4
    data/cvat/rally_001_cvat.zip

and call ``convert_cvat`` in *auto* mode (or ``main.py convert-cvat --auto``)
— the converter pairs each video with its CVAT export by base name. The
trailing ``_cvat`` suffix on the zip is configurable.

Expected CVAT XML structure (Video 1.1)::

    <annotations>
      <meta>
        <task>
          <original_size><width>...</width><height>...</height></original_size>
        </task>
      </meta>
      <track id="0" label="player1" source="manual">
        <box frame="0" outside="0" occluded="0" keyframe="1"
             xtl="..." ytl="..." xbr="..." ybr="..." z_order="0">
          <attribute name="split_step">0</attribute>
        </box>
        ...
      </track>
      <track id="1" label="player2" source="manual">
        ...
      </track>
    </annotations>

We tolerate missing/extra attributes and ``outside="1"`` (skipped) frames.
"""

from __future__ import annotations

import csv
import shutil
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import yaml

from ..utils.geometry import clip_bbox, crop_with_padding, xyxy_to_yolo
from ..utils.logging import get_logger
from .splitting import (
    assign_group_splits,
    assign_stratified_group_splits,
    split_train_val_test,
)
from .video_io import VideoReader

logger = get_logger("cvat")


# Common video extensions we'll match against CVAT zips/XMLs by base name.
VIDEO_EXTENSIONS: Tuple[str, ...] = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")
CVAT_EXTENSIONS: Tuple[str, ...] = (".zip", ".xml")
DEFAULT_CVAT_SUFFIX = "_cvat"  # rally_001.mp4 <-> rally_001_cvat.zip


# -------------------------------------------------------------------- #
# Data classes
# -------------------------------------------------------------------- #
@dataclass
class CvatBox:
    frame: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    outside: bool
    occluded: bool
    attributes: Dict[str, str] = field(default_factory=dict)

    def action_label(self, attr_name: str = "split_step") -> Optional[int]:
        """Parse a per-frame action label from a named CVAT box attribute."""
        return parse_action_label(self.attributes, attr_name)

    @property
    def split_step(self) -> Optional[int]:
        """Backward-compatible alias for ``action_label('split_step')``."""
        return self.action_label("split_step")


@dataclass
class CvatTrack:
    track_id: int
    label: str
    boxes: List[CvatBox] = field(default_factory=list)

    def by_frame(self) -> Dict[int, CvatBox]:
        return {b.frame: b for b in self.boxes if not b.outside}


@dataclass
class CvatAnnotations:
    width: int
    height: int
    tracks: List[CvatTrack]
    source_xml: Optional[str] = None
    source_archive: Optional[str] = None  # populated when loaded from .zip

    def track_for_label(self, label: str) -> Optional[CvatTrack]:
        for t in self.tracks:
            if t.label == label:
                return t
        return None

    def all_frames(self) -> List[int]:
        seen: set[int] = set()
        for t in self.tracks:
            for b in t.boxes:
                if not b.outside:
                    seen.add(b.frame)
        return sorted(seen)


@dataclass
class CvatJob:
    """A single (video, annotations) pair to be exported."""

    video_path: Path
    annotations: CvatAnnotations

    @property
    def stem(self) -> str:
        return self.video_path.stem


def parse_action_label(
    attributes: Dict[str, str],
    attr_name: str = "split_step",
) -> Optional[int]:
    """Coerce a CVAT box attribute into ``0`` (normal) or ``1`` (split_step).

    Accepts common naming/value conventions, e.g.::

        split_step=0/1
        movement_state=normal/split_step
    """
    v = attributes.get(attr_name)
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "split", "split_step"}:
        return 1
    if s in {"0", "false", "no", "normal"}:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return None


def _label_matches(track_label: str, expected: str) -> bool:
    return track_label.strip().lower() == expected.strip().lower()


def build_track_player_map(
    tracks: Sequence[CvatTrack],
    player1_label: str,
    player2_label: str,
) -> Dict[int, int]:
    """Map CVAT ``track_id`` -> player slot (1 or 2).

    When ``player1_label`` and ``player2_label`` differ, tracks are matched by
    label name. When they are the same (e.g. both ``Player``), the first two
    matching tracks ordered by ``track_id`` become player 1 and player 2.
    """
    p1 = player1_label.strip()
    p2 = player2_label.strip()
    if p1.lower() != p2.lower():
        label_to_pid = {p1.lower(): 1, p2.lower(): 2}
        return {
            tr.track_id: label_to_pid[tr.label.strip().lower()]
            for tr in tracks
            if tr.label.strip().lower() in label_to_pid
        }

    matching = sorted(
        (tr for tr in tracks if _label_matches(tr.label, p1)),
        key=lambda tr: tr.track_id,
    )
    if len(matching) > 2:
        logger.warning(
            f"More than 2 tracks with label '{p1}'; assigning player slots "
            f"to the first two by track id."
        )
    return {tr.track_id: idx + 1 for idx, tr in enumerate(matching[:2])}


# -------------------------------------------------------------------- #
# Zip / XML loading
# -------------------------------------------------------------------- #
def _extract_xml_from_zip(zip_path: Path, dest_dir: Path) -> Path:
    """Extract ``annotations.xml`` from a CVAT for Video 1.1 zip.

    Returns the path to the extracted XML inside ``dest_dir``.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a zip archive: {zip_path}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        # CVAT Video 1.1 conventionally puts annotations.xml at the archive root.
        candidates = [
            n
            for n in names
            if Path(n).name.lower() == "annotations.xml"
        ]
        if not candidates:
            raise FileNotFoundError(
                f"'annotations.xml' not found inside {zip_path.name}. "
                f"Archive contents: {names[:10]}{'...' if len(names) > 10 else ''}"
            )
        # Prefer the shallowest match (root) if multiple are present.
        chosen = sorted(candidates, key=lambda n: n.count("/"))[0]
        out = dest_dir / "annotations.xml"
        with zf.open(chosen) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    logger.debug(f"Extracted {chosen} from {zip_path.name} -> {out}")
    return out


def parse_cvat_xml(path: str | Path) -> CvatAnnotations:
    """Parse a CVAT for Video 1.1 export.

    Accepts either the raw ``annotations.xml`` or the full CVAT ``.zip``
    archive — the zip is unpacked into a temporary directory and the embedded
    ``annotations.xml`` is parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    archive: Optional[str] = None
    if path.suffix.lower() == ".zip":
        archive = str(path)
        tmp = Path(tempfile.mkdtemp(prefix="cvat_"))
        try:
            xml_path = _extract_xml_from_zip(path, tmp)
            ann = _parse_xml_file(xml_path)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        ann.source_archive = archive
        ann.source_xml = str(path)  # report the user-facing input path
        logger.info(f"Parsed CVAT zip {path.name}: {len(ann.tracks)} tracks")
        return ann

    ann = _parse_xml_file(path)
    logger.info(f"Parsed CVAT XML {path.name}: {len(ann.tracks)} tracks")
    return ann


def _parse_xml_file(xml_path: Path) -> CvatAnnotations:
    """Inner XML parser shared by zip and direct-XML paths."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width: Optional[int] = None
    height: Optional[int] = None
    size_el = root.find(".//original_size")
    if size_el is not None:
        try:
            w_el = size_el.find("width")
            h_el = size_el.find("height")
            if w_el is not None and w_el.text:
                width = int(w_el.text)
            if h_el is not None and h_el.text:
                height = int(h_el.text)
        except ValueError:
            pass

    tracks: List[CvatTrack] = []
    for tr in root.findall("track"):
        try:
            track_id = int(tr.attrib.get("id", "-1"))
        except ValueError:
            track_id = -1
        label = tr.attrib.get("label", "")
        boxes: List[CvatBox] = []
        for box in tr.findall("box"):
            try:
                frame = int(box.attrib["frame"])
                xtl = float(box.attrib["xtl"])
                ytl = float(box.attrib["ytl"])
                xbr = float(box.attrib["xbr"])
                ybr = float(box.attrib["ybr"])
            except (KeyError, ValueError) as exc:
                logger.warning(f"Skipping malformed box: {exc}")
                continue
            outside = box.attrib.get("outside", "0") == "1"
            occluded = box.attrib.get("occluded", "0") == "1"
            attrs: Dict[str, str] = {}
            for attr in box.findall("attribute"):
                name = attr.attrib.get("name", "")
                value = (attr.text or "").strip()
                if name:
                    attrs[name] = value
            boxes.append(
                CvatBox(
                    frame=frame,
                    bbox=(xtl, ytl, xbr, ybr),
                    outside=outside,
                    occluded=occluded,
                    attributes=attrs,
                )
            )
        boxes.sort(key=lambda b: b.frame)
        tracks.append(CvatTrack(track_id=track_id, label=label, boxes=boxes))

    if width is None or height is None:
        width = width or 0
        height = height or 0
        logger.warning(
            "original_size missing/incomplete in XML; will rely on video dims."
        )
    return CvatAnnotations(
        width=width,
        height=height,
        tracks=tracks,
        source_xml=str(xml_path),
    )


# -------------------------------------------------------------------- #
# Video <-> CVAT pairing
# -------------------------------------------------------------------- #
def _strip_suffix(stem: str, suffix: str) -> str:
    return stem[: -len(suffix)] if suffix and stem.endswith(suffix) else stem


def find_cvat_for_video(
    video_path: str | Path,
    cvat_dir: str | Path,
    suffix: str = DEFAULT_CVAT_SUFFIX,
) -> Optional[Path]:
    """Locate a CVAT export (zip preferred, xml fallback) matching ``video_path``.

    Match order (first hit wins):
      1. ``<cvat_dir>/<video_stem><suffix>.zip``
      2. ``<cvat_dir>/<video_stem><suffix>.xml``
      3. ``<cvat_dir>/<video_stem>.zip``
      4. ``<cvat_dir>/<video_stem>.xml``
    """
    cvat_dir = Path(cvat_dir)
    stem = Path(video_path).stem
    candidates = [
        cvat_dir / f"{stem}{suffix}.zip",
        cvat_dir / f"{stem}{suffix}.xml",
        cvat_dir / f"{stem}.zip",
        cvat_dir / f"{stem}.xml",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_video_for_cvat(
    cvat_path: str | Path,
    raw_dir: str | Path,
    suffix: str = DEFAULT_CVAT_SUFFIX,
) -> Optional[Path]:
    """Locate a video matching a CVAT export path."""
    raw_dir = Path(raw_dir)
    stem = _strip_suffix(Path(cvat_path).stem, suffix)
    for ext in VIDEO_EXTENSIONS:
        cand = raw_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def pair_videos_and_cvat(
    raw_dir: str | Path,
    cvat_dir: str | Path,
    suffix: str = DEFAULT_CVAT_SUFFIX,
) -> List[Tuple[Path, Path]]:
    """Walk ``raw_dir`` and pair each video with its CVAT export under ``cvat_dir``.

    Videos without a matching CVAT file are skipped (with a warning).
    """
    raw_dir = Path(raw_dir)
    cvat_dir = Path(cvat_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir does not exist: {raw_dir}")
    if not cvat_dir.exists():
        raise FileNotFoundError(f"cvat_dir does not exist: {cvat_dir}")

    videos: List[Path] = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(sorted(raw_dir.glob(f"*{ext}")))
    videos = sorted({v for v in videos})

    pairs: List[Tuple[Path, Path]] = []
    missing: List[Path] = []
    for v in videos:
        cvat = find_cvat_for_video(v, cvat_dir, suffix=suffix)
        if cvat is None:
            missing.append(v)
            continue
        pairs.append((v, cvat))
    if missing:
        logger.warning(
            f"{len(missing)} video(s) had no matching CVAT export under {cvat_dir} "
            f"(suffix='{suffix}'): {[m.name for m in missing]}"
        )
    if not pairs:
        raise RuntimeError(
            f"No (video, CVAT) pairs found in {raw_dir} <-> {cvat_dir}. "
            f"Expected files like 'rally_001.mp4' and 'rally_001{suffix}.zip'."
        )
    logger.info(f"Discovered {len(pairs)} video<->CVAT pair(s).")
    return pairs


def discover_jobs(
    raw_dir: str | Path,
    cvat_dir: str | Path,
    suffix: str = DEFAULT_CVAT_SUFFIX,
) -> List[CvatJob]:
    """Auto-discover and parse all CVAT jobs under the conventional layout."""
    pairs = pair_videos_and_cvat(raw_dir, cvat_dir, suffix=suffix)
    jobs: List[CvatJob] = []
    for video, cvat_file in pairs:
        ann = parse_cvat_xml(cvat_file)
        jobs.append(CvatJob(video_path=video, annotations=ann))
    return jobs


# -------------------------------------------------------------------- #
# YOLO detection export
# -------------------------------------------------------------------- #
def _log_group_split(group_splits: Dict[str, str], seed: int) -> None:
    n_train = sum(1 for s in group_splits.values() if s == "train")
    n_val = sum(1 for s in group_splits.values() if s == "val")
    n_test = sum(1 for s in group_splits.values() if s == "test")
    logger.info(
        f"Group split over {len(group_splits)} video(s): "
        f"train={n_train}, val={n_val}, test={n_test} (seed={seed})"
    )
    for split in ("train", "val", "test"):
        videos = sorted(stem for stem, assigned in group_splits.items() if assigned == split)
        if videos:
            logger.info(f"Group split {split}: {', '.join(videos)}")


def _coerce_jobs(
    jobs: "CvatJob | Iterable[CvatJob]",
) -> List[CvatJob]:
    if isinstance(jobs, CvatJob):
        return [jobs]
    return list(jobs)


def export_yolo_detection(
    jobs: "CvatJob | Iterable[CvatJob]",
    output_dir: str | Path,
    val_split: float = 0.2,
    test_split: float = 0.2,
    every_n_frames: int = 1,
    yolo_class_name: str = "player",
    seed: int = 42,
    group_split: bool = True,
) -> Path:
    """Export a YOLO-detection-ready dataset from one or more CVAT jobs.

    Both ``player1`` and ``player2`` tracks collapse into a single ``player``
    class so a vanilla single-class YOLO can be fine-tuned out of the box.
    The output directory is overwritten if it exists.

    Labeled frames are split three ways (``train`` / ``val`` / ``test``) and
    routed to ``images/<split>`` + ``labels/<split>``. The train share is
    implicit (``1 - val_split - test_split``). When ``group_split`` is true
    (default), whole videos are assigned to one split via a seeded shuffle.
    Otherwise each video's frames are split independently at random.

    Returns the path to the written ``data.yaml``.
    """
    job_list = _coerce_jobs(jobs)
    if not job_list:
        raise ValueError("No CVAT jobs provided.")

    output_dir = Path(output_dir)
    has_test = test_split > 0
    split_dirs = {
        "train": (output_dir / "images" / "train", output_dir / "labels" / "train"),
        "val": (output_dir / "images" / "val", output_dir / "labels" / "val"),
    }
    if has_test:
        split_dirs["test"] = (
            output_dir / "images" / "test",
            output_dir / "labels" / "test",
        )
    for img_dir, lbl_dir in split_dirs.values():
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    group_splits: Dict[str, str] = {}
    if group_split:
        group_splits = assign_group_splits(
            [job.stem for job in job_list], val_split, test_split, seed=seed
        )
        _log_group_split(group_splits, seed)

    total_written = 0
    for ji, job in enumerate(job_list):
        annotations = job.annotations
        video_path = job.video_path
        stem = job.stem

        frames_with_any_box = annotations.all_frames()
        if not frames_with_any_box:
            logger.warning(f"[{stem}] no labeled frames; skipping YOLO export.")
            continue

        sampled = [f for f in frames_with_any_box if (f % max(1, every_n_frames)) == 0]
        if group_split:
            video_split = group_splits[stem]
            logger.info(
                f"[{stem}] YOLO: {len(sampled)} frames -> split={video_split}"
            )
        else:
            # Per-job seed so repeated runs are reproducible regardless of order.
            train_set, val_set, test_set = split_train_val_test(
                sampled, val_split, test_split, seed=seed + ji
            )
            logger.info(
                f"[{stem}] YOLO: {len(sampled)} frames "
                f"(train={len(train_set)}, val={len(val_set)}, test={len(test_set)})"
            )

        sampled_set = set(sampled)
        n_written = 0
        with VideoReader(video_path) as reader:
            img_w = annotations.width or reader.width
            img_h = annotations.height or reader.height
            for frame_idx, frame in reader.iter_frames():
                if frame_idx not in sampled_set:
                    continue
                if group_split:
                    split = video_split
                elif frame_idx in val_set:
                    split = "val"
                elif frame_idx in test_set:
                    split = "test"
                else:
                    split = "train"
                img_dir, lbl_dir = split_dirs[split]

                lines: List[str] = []
                for tr in annotations.tracks:
                    box = next(
                        (b for b in tr.boxes if b.frame == frame_idx and not b.outside),
                        None,
                    )
                    if box is None:
                        continue
                    bx = clip_bbox(box.bbox, img_w, img_h)
                    if (bx[2] - bx[0]) <= 1 or (bx[3] - bx[1]) <= 1:
                        continue
                    cx, cy, w, h = xyxy_to_yolo(bx, img_w, img_h)
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                if not lines:
                    continue

                fname = f"{stem}_f{frame_idx:06d}"
                cv2.imwrite(
                    str(img_dir / f"{fname}.jpg"),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92],
                )
                (lbl_dir / f"{fname}.txt").write_text("\n".join(lines) + "\n")
                n_written += 1
        total_written += n_written
        logger.info(f"[{stem}] YOLO frames written: {n_written}")

    if total_written == 0:
        raise RuntimeError("No YOLO frames were written; check XML/video alignment.")

    data_yaml: Dict[str, object] = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
    }
    if has_test:
        data_yaml["test"] = "images/test"
    data_yaml["names"] = {0: yolo_class_name}
    yaml_path = output_dir / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
    logger.info(f"YOLO export done: {total_written} total frames -> {output_dir}")
    return yaml_path


# -------------------------------------------------------------------- #
# Action dataset export
# -------------------------------------------------------------------- #
def export_action_dataset(
    jobs: "CvatJob | Iterable[CvatJob]",
    output_dir: str | Path,
    clip_length: int = 16,
    clip_stride: int = 1,
    val_split: float = 0.2,
    test_split: float = 0.2,
    player1_label: str = "player1",
    player2_label: str = "player2",
    split_attribute: str = "split_step",
    pad_ratio: float = 0.15,
    crop_size: int = 224,
    seed: int = 42,
    group_split: bool = True,
    positive_label_ratio: float = 0.15,
    soft_transition_frames: int = 3,
    soft_transition_min: float = 0.25,
    centered_clips: bool = False,
) -> Path:
    """Export per-player crop clips for the action model.

    For each labeled frame ``f`` of a player track, we build a clip of
    ``clip_length`` frames. By default this is a trailing causal window
    (``f - clip_length + 1`` … ``f``). If ``centered_clips`` is true, the
    window is centered around ``f``. The manifest stores both a hard ``label``
    for metrics and a soft ``target`` for BCE training. The soft target is the
    average action value among labeled frames in the clip window. Positive
    motion frames stay at 1.0; nearby normal frames can be softened by
    ``soft_transition_frames`` so boundary timing is not punished as harshly.

    Clips are split three ways into ``train`` / ``val`` / ``test`` (the train
    share is implicit: ``1 - val_split - test_split``) and the chosen split
    is recorded in the ``split`` column of the manifest. When ``group_split``
    is true (default), all clips from a video share one split via a seeded
    shuffle over video stems. Otherwise clips are split independently at
    random within each video.

    Writes one global ``manifest.csv`` covering all jobs.
    """
    job_list = _coerce_jobs(jobs)
    if not job_list:
        raise ValueError("No CVAT jobs provided.")

    output_dir = Path(output_dir)
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    total_clips = 0
    total_frames = 0
    total_skipped = 0
    positive_label_ratio = max(0.0, min(1.0, float(positive_label_ratio)))
    soft_transition_frames = max(0, int(soft_transition_frames))
    soft_transition_min = max(0.0, min(1.0, float(soft_transition_min)))
    targets_by_stem: Dict[str, List[Tuple[int, int, int, float]]] = {}
    label_counts_by_stem: Dict[str, Dict[int, int]] = {}

    for job in job_list:
        stem = job.stem
        track_to_pid = build_track_player_map(
            job.annotations.tracks, player1_label, player2_label
        )
        labels_by_pid: Dict[int, Dict[int, int]] = {}
        for tr in job.annotations.tracks:
            pid = track_to_pid.get(tr.track_id)
            if pid is None:
                continue
            for box in tr.boxes:
                if box.outside:
                    continue
                label = box.action_label(split_attribute)
                if label is None:
                    continue
                labels_by_pid.setdefault(pid, {})[box.frame] = int(label)
                if (box.frame % max(1, clip_stride)) != 0:
                    continue
        targets: List[Tuple[int, int, int, float]] = []
        for pid, labels_by_frame in labels_by_pid.items():
            soft_labels_by_frame = _soft_transition_labels(
                labels_by_frame,
                transition_frames=soft_transition_frames,
                min_value=soft_transition_min,
            )
            for frame, label in sorted(labels_by_frame.items()):
                if (frame % max(1, clip_stride)) != 0:
                    continue
                frames = _clip_frame_indices(frame, clip_length, centered=centered_clips)
                target = _window_action_target(soft_labels_by_frame, frames, fallback=label)
                hard_label = int(target >= positive_label_ratio)
                targets.append((pid, frame, hard_label, target))
        if not targets:
            logger.warning(
                f"[{stem}] no '{split_attribute}' targets found; skipping action export."
            )
            continue
        targets.sort(key=lambda x: (x[1], x[0]))
        counts = {0: 0, 1: 0}
        for _pid, _frame, label, _target in targets:
            counts[label] = counts.get(label, 0) + 1
        targets_by_stem[stem] = targets
        label_counts_by_stem[stem] = counts

    group_splits: Dict[str, str] = {}
    if group_split:
        group_splits = assign_stratified_group_splits(
            label_counts_by_stem, val_split, test_split, seed=seed
        )
        _log_group_split(group_splits, seed)

    for ji, job in enumerate(job_list):
        annotations = job.annotations
        video_path = job.video_path
        stem = job.stem
        track_to_pid = build_track_player_map(
            annotations.tracks, player1_label, player2_label
        )

        # Targets were precomputed so group split can ignore empty videos and
        # balance class counts across splits before clips are written.
        targets = targets_by_stem.get(stem, [])
        if not targets:
            continue
        total_clips += len(targets)

        track_boxes: Dict[int, Dict[int, CvatBox]] = {}
        for tr in annotations.tracks:
            pid = track_to_pid.get(tr.track_id)
            if pid is None:
                continue
            track_boxes[pid] = tr.by_frame()

        if group_split:
            video_split = group_splits[stem]
        else:
            train_indices, val_indices, test_indices = split_train_val_test(
                range(len(targets)), val_split, test_split, seed=seed + ji
            )

        # Map each needed source frame -> [(clip_idx, position_in_clip), ...]
        needed: Dict[int, List[Tuple[int, int]]] = {}
        for clip_idx, (pid, center, _label, _target) in enumerate(targets):
            for k, f in enumerate(
                _clip_frame_indices(center, clip_length, centered=centered_clips)
            ):
                if f < 0:
                    continue
                needed.setdefault(f, []).append((clip_idx, k))

        clip_dirs: List[Path] = []
        for clip_idx, (pid, center, label, target) in enumerate(targets):
            clip_id = f"{stem}_p{pid}_f{center:06d}_c{clip_idx:06d}"
            cdir = clips_dir / clip_id
            cdir.mkdir(parents=True, exist_ok=True)
            clip_dirs.append(cdir)
            if group_split:
                split = video_split
            elif clip_idx in val_indices:
                split = "val"
            elif clip_idx in test_indices:
                split = "test"
            else:
                split = "train"
            manifest_rows.append(
                {
                    "clip_id": clip_id,
                    "video": stem,
                    "player_id": str(pid),
                    "center_frame": str(center),
                    "label": str(label),
                    "target": f"{target:.6f}",
                    "split": split,
                }
            )

        n_frames_written = 0
        n_skipped = 0
        with VideoReader(video_path) as reader:
            max_needed_frame = max(needed.keys()) if needed else -1
            for frame_idx, frame in reader.iter_frames(end=max_needed_frame + 1):
                uses = needed.get(frame_idx)
                if not uses:
                    continue
                for clip_idx, position in uses:
                    pid, _center, _label, _target = targets[clip_idx]
                    box = track_boxes[pid].get(frame_idx)
                    if box is None:
                        nearest = _nearest_box(track_boxes[pid], frame_idx)
                        if nearest is None:
                            n_skipped += 1
                            continue
                        box = nearest
                    bx = clip_bbox(box.bbox, frame.shape[1], frame.shape[0])
                    crop = crop_with_padding(frame, bx, pad_ratio=pad_ratio)
                    if crop.size == 0:
                        n_skipped += 1
                        continue
                    resized = cv2.resize(crop, (crop_size, crop_size))
                    cv2.imwrite(
                        str(clip_dirs[clip_idx] / f"{position:02d}.jpg"),
                        resized,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                    )
                    n_frames_written += 1
        total_frames += n_frames_written
        total_skipped += n_skipped
        if group_split:
            logger.info(
                f"[{stem}] action: {len(targets)} clips -> split={video_split}, "
                f"{n_frames_written} frames written, {n_skipped} skipped."
            )
        else:
            logger.info(
                f"[{stem}] action: {len(targets)} clips "
                f"(train={len(train_indices)}, val={len(val_indices)}, "
                f"test={len(test_indices)}), "
                f"{n_frames_written} frames written, {n_skipped} skipped."
            )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_id",
                "video",
                "player_id",
                "center_frame",
                "label",
                "target",
                "split",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info(
        f"Action export done: {total_clips} clips across {len(job_list)} job(s), "
        f"{total_frames} frames written, {total_skipped} skipped -> {output_dir}"
    )
    return manifest_path


def _nearest_box(per_frame: Dict[int, CvatBox], frame_idx: int) -> Optional[CvatBox]:
    if not per_frame:
        return None
    keys = sorted(per_frame.keys())
    best: Optional[int] = None
    best_diff = 10**9
    for k in keys:
        d = abs(k - frame_idx)
        if d < best_diff:
            best = k
            best_diff = d
    return per_frame.get(best) if best is not None else None


def _clip_frame_indices(center: int, clip_length: int, *, centered: bool = False) -> List[int]:
    clip_length = max(1, int(clip_length))
    if centered:
        before = clip_length // 2
        start = center - before
    else:
        start = center - clip_length + 1
    return [start + i for i in range(clip_length)]


def _soft_transition_labels(
    labels_by_frame: Dict[int, int],
    *,
    transition_frames: int,
    min_value: float,
) -> Dict[int, float]:
    """Return per-frame soft targets with ramps around positive intervals.

    Hard positive frames stay at 1.0. Labeled normal frames up to
    ``transition_frames`` before/after a positive interval receive a linearly
    tapered target from ``min_value`` at the outer edge to nearly 1.0 at the
    interval boundary. Frames far from positives remain 0.0.
    """
    soft = {frame: float(int(label) == 1) for frame, label in labels_by_frame.items()}
    if transition_frames <= 0 or not labels_by_frame:
        return soft

    positive_frames = sorted(frame for frame, label in labels_by_frame.items() if int(label) == 1)
    if not positive_frames:
        return soft

    intervals: List[Tuple[int, int]] = []
    start = positive_frames[0]
    prev = positive_frames[0]
    for frame in positive_frames[1:]:
        if frame == prev + 1:
            prev = frame
            continue
        intervals.append((start, prev))
        start = frame
        prev = frame
    intervals.append((start, prev))

    labeled_frames = set(labels_by_frame)
    for start, end in intervals:
        for offset in range(1, transition_frames + 1):
            value = _transition_value(offset, transition_frames, min_value)
            before = start - offset
            after = end + offset
            if before in labeled_frames and int(labels_by_frame[before]) == 0:
                soft[before] = max(soft.get(before, 0.0), value)
            if after in labeled_frames and int(labels_by_frame[after]) == 0:
                soft[after] = max(soft.get(after, 0.0), value)
    return soft


def _transition_value(offset: int, transition_frames: int, min_value: float) -> float:
    # offset=1 is closest to the motion interval and should be strongest.
    strength = (transition_frames - offset + 1) / (transition_frames + 1)
    return max(0.0, min(1.0, min_value + (1.0 - min_value) * strength))


def _window_action_target(
    labels_by_frame: Dict[int, float],
    frame_indices: Sequence[int],
    *,
    fallback: int | float,
) -> float:
    labels = [labels_by_frame[f] for f in frame_indices if f in labels_by_frame]
    if not labels:
        return float(fallback)
    return float(sum(labels) / len(labels))


# -------------------------------------------------------------------- #
# High-level convenience: do both YOLO and action in one call
# -------------------------------------------------------------------- #
def convert_cvat(
    jobs: "CvatJob | Iterable[CvatJob]",
    output_root: str | Path = "data",
    *,
    do_yolo: bool = True,
    do_action: bool = True,
    val_split: float = 0.2,
    test_split: float = 0.2,
    every_n_frames: int = 1,
    yolo_class_name: str = "player",
    clip_length: int = 16,
    clip_stride: int = 1,
    crop_size: int = 224,
    pad_ratio: float = 0.15,
    player1_label: str = "player1",
    player2_label: str = "player2",
    split_attribute: str = "split_step",
    seed: int = 42,
    group_split: bool = True,
    positive_label_ratio: float = 0.15,
    soft_transition_frames: int = 3,
    soft_transition_min: float = 0.25,
    centered_action_clips: bool = False,
) -> Dict[str, Path]:
    """Run YOLO + action exports in one shot.

    Output goes to ``<output_root>/yolo`` and ``<output_root>/action`` so the
    standard project layout is::

        data/yolo/...
        data/action/...

    Returns a dict ``{"yolo": <data.yaml path>, "action": <manifest.csv path>}``
    with whichever exports were requested.
    """
    output_root = Path(output_root)
    out: Dict[str, Path] = {}
    if do_yolo:
        out["yolo"] = export_yolo_detection(
            jobs,
            output_dir=output_root / "yolo",
            val_split=val_split,
            test_split=test_split,
            every_n_frames=every_n_frames,
            yolo_class_name=yolo_class_name,
            seed=seed,
            group_split=group_split,
        )
    if do_action:
        out["action"] = export_action_dataset(
            jobs,
            output_dir=output_root / "action",
            clip_length=clip_length,
            clip_stride=clip_stride,
            val_split=val_split,
            test_split=test_split,
            player1_label=player1_label,
            player2_label=player2_label,
            split_attribute=split_attribute,
            pad_ratio=pad_ratio,
            crop_size=crop_size,
            seed=seed,
            group_split=group_split,
            positive_label_ratio=positive_label_ratio,
            soft_transition_frames=soft_transition_frames,
            soft_transition_min=soft_transition_min,
            centered_clips=centered_action_clips,
        )
    return out


def auto_convert(
    raw_dir: str | Path = "data/raw",
    cvat_dir: str | Path = "data/cvat",
    output_root: str | Path = "data",
    suffix: str = DEFAULT_CVAT_SUFFIX,
    **kwargs,
) -> Dict[str, Path]:
    """Discover videos + CVAT zips/xmls and run the full conversion.

    Equivalent to ``convert_cvat(discover_jobs(raw_dir, cvat_dir, suffix), ...)``.
    Forward any remaining kwargs to :func:`convert_cvat`.
    """
    jobs = discover_jobs(raw_dir, cvat_dir, suffix=suffix)
    return convert_cvat(jobs, output_root=output_root, **kwargs)


# -------------------------------------------------------------------- #
# CLI entry point: ``python -m src.data.cvat_converter ...``
# -------------------------------------------------------------------- #
def _cli() -> None:
    import typer

    app = typer.Typer(
        add_completion=False,
        help="Convert CVAT for Video 1.1 (XML or zip) into YOLO + action datasets.",
    )

    @app.command("auto")
    def cmd_auto(
        raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
        cvat_dir: Path = typer.Option(Path("data/cvat"), "--cvat-dir"),
        out: Path = typer.Option(Path("data"), "--out"),
        suffix: str = typer.Option(DEFAULT_CVAT_SUFFIX, "--suffix"),
        clip_len: int = typer.Option(16, "--clip-len"),
        val_split: float = typer.Option(0.2, "--val-split"),
        test_split: float = typer.Option(0.2, "--test-split"),
        every_n: int = typer.Option(1, "--every-n"),
        no_group_split: bool = typer.Option(
            False,
            "--no-group-split",
            help="Split randomly within each video instead of by whole video.",
        ),
    ) -> None:
        """Auto-pair videos in --raw-dir with CVAT files in --cvat-dir."""
        auto_convert(
            raw_dir=raw_dir,
            cvat_dir=cvat_dir,
            output_root=out,
            suffix=suffix,
            clip_length=clip_len,
            val_split=val_split,
            test_split=test_split,
            every_n_frames=every_n,
            group_split=not no_group_split,
        )

    @app.command("single")
    def cmd_single(
        video: Path = typer.Option(..., "--video", exists=True, dir_okay=False),
        cvat: Path = typer.Option(
            ...,
            "--cvat",
            exists=True,
            dir_okay=False,
            help="CVAT for Video 1.1 .xml or .zip.",
        ),
        out: Path = typer.Option(Path("data"), "--out"),
        clip_len: int = typer.Option(16, "--clip-len"),
        val_split: float = typer.Option(0.2, "--val-split"),
        test_split: float = typer.Option(0.2, "--test-split"),
        every_n: int = typer.Option(1, "--every-n"),
        no_group_split: bool = typer.Option(
            False,
            "--no-group-split",
            help="Split randomly within each video instead of by whole video.",
        ),
        only: str = typer.Option(
            "both", "--only", help="yolo | action | both"
        ),
    ) -> None:
        """Convert a single (video, CVAT) pair."""
        ann = parse_cvat_xml(cvat)
        job = CvatJob(video_path=video, annotations=ann)
        do_yolo = only in {"yolo", "both"}
        do_action = only in {"action", "both"}
        convert_cvat(
            job,
            output_root=out,
            do_yolo=do_yolo,
            do_action=do_action,
            clip_length=clip_len,
            val_split=val_split,
            test_split=test_split,
            every_n_frames=every_n,
            group_split=not no_group_split,
        )

    app()


if __name__ == "__main__":  # pragma: no cover
    _cli()
