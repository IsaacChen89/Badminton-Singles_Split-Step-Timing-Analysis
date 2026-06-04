# Badminton Singles — Split Step Timing Analysis

The split step is an essential badminton skill. Correct timing can help player mover faster.

---

A Python pipeline framework and training environment built with PyTorch, designed for analysing the timing of split step in professional badminton singles:

1. **Detects and tracks both players** with colored bounding boxes
  (Player 1 = red, Player 2 = blue) using YOLO26 + BoT-SORT.
2. **Classifies each player's split step** with a lightweight CNN-LSTM
  temporal model.
3. **Renders an annotated MP4** with persistent `SPLIT STEP` labels for
  visual review.
4. **Trains itself** from labels you produce in **CVAT for Video 1.1**.

```text
       Video               YOLO26           BoT-SORT          CNN-LSTM           Hysteresis
   ┌─────────┐         ┌───────────┐     ┌─────────────┐    ┌────────────┐    ┌─────────────┐
   │  match  │  ───►   │  detect   │ ──► │  track IDs  │ ─► │  per-clip  │ ─► │  smoothed   │
   │  .mp4   │         │  players  │     │ (BoT-SORT)  │    │ split prob │    │ SPLIT STEP  │
   └─────────┘         └───────────┘     └──────┬──────┘    └─────┬──────┘    └──────┬──────┘
                                                │                 │                  │
                                          PlayerAssigner   ROI crop deque    Annotated video
                                          (red / blue)        (T=16)         (red/blue + label)
```

---

## Table of contents

1. [Project layout](#project-layout)
2. [Install](#install)
3. [Run inference (annotate a match)](#run-inference)
4. [Tracking robustness](#tracking-robustness)
5. [Label your own data with CVAT](#label-your-own-data-with-cvat)
6. [Convert CVAT exports → training datasets](#convert-cvat-exports--training-datasets)
7. [Train the YOLO detector](#train-the-yolo-detector)
8. [Train the split-step model](#train-the-split-step-model)
9. [Configuration reference (`config.yaml`)](#configuration-reference)

---

## Project layout

```
badminton-splitstep-analyzer/
├── README.md
├── requirements.txt
├── config.yaml
├── main.py                         # typer CLI entry point
├── src/
│   ├── detection/                  # YOLO26 wrapper
│   │   └── yolo_detector.py
│   ├── tracking/                   # BoT-SORT + Player1/Player2 mapping
│   │   ├── tracker.py
│   │   └── player_assigner.py
│   ├── action/                     # CNN-LSTM + training + smoothing
│   │   ├── model.py
│   │   ├── dataset.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── smoothing.py
│   ├── data/                       # I/O + the CVAT converter
│   │   ├── cvat_converter.py       # ★ primary CVAT bridge
│   │   └── video_io.py
│   ├── utils/                      # config, logging, geometry
│   └── visualization/              # bounding boxes, SPLIT STEP labels, HUD
├── models/                         # checkpoints (gitignored)
├── data/                           # raw videos, CVAT exports, derived datasets
├── outputs/                        # annotated MP4s
└── scripts/                        # thin wrappers around main.py sub-commands
```

---

## Install

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

CUDA is optional but recommended on Linux/Windows. On Apple Silicon, `device:
auto` picks **MPS** (Metal GPU) when available. The CLI auto-detects the best backend (`device: auto` in `config.yaml`).

To pre-download the stock YOLO checkpoint into `./models/`:

```bash
python scripts/download_yolo_weights.py            # -> models/yolo_player/yolo26n.pt
python scripts/download_yolo_weights.py yolo26n-cls.pt  # BoT-SORT Re-ID (strong tracking)
```

Verify your environment:

```bash
python main.py info
```

---

## Run inference

Annotate a video, producing `outputs/match_annotated.mp4`:

```bash
python main.py analyze \
  --video  data/raw/match.mp4 \
  --output outputs/match_annotated.mp4
```

- Input videos at any frame rate (25, 30, 60 FPS, …) are automatically resampled to **30 FPS** before being fed to the pipeline so the action model sees the same temporal cadence it was trained on. 
- The annotated output MP4 is written at the same target rate. 
- Override with `--target-fps <N>` or `pipeline.target_fps` in `config.yaml`; pass `--target-fps 0` (or set the config to `null`) to disable resampling and keep the source FPS.

Useful options:


| Flag                                                | Effect                                                                                             |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `--debug-ids`                                       | Show raw tracker IDs next to each player box (useful when diagnosing ID swaps).                    |
| `--min-confidence 0.3`                              | Drop detections below this YOLO confidence.                                                        |
| `--frame-skip 2`                                    | Run detection/tracking only on every 2nd frame (≈2× faster).                                       |
| `--target-fps 30`                                   | Resample the input video to this FPS before processing (default 30, pass 0 to disable).            |
| `--device cpu` / `--device cuda` / `--device mps` | Force device (default `auto`: cuda → mps → cpu). |
| `--yolo-weights models/yolo_player/yolo_player_best.pt` | Use a fine-tuned YOLO checkpoint.                                                              |
| `--action-weights models/action_player/action_best.pt` | Use a specific action checkpoint.                                                               |
| `--no-hud`                                          | Hide the corner HUD.                                                                               |
| `--max-frames 600`                                  | Stop after 600 frames (debugging).                                                                 |


> If `models/action_player/action_best.pt` does not yet exist (you haven't trained the split-step model), `analyze` still runs and writes the bounding boxes — the `SPLIT STEP` label simply never fires. Train the action model and re-run.

---

## Tracking robustness

`analyze` keeps **Player 1 = red** and **Player 2 = blue** stable across long matches by chaining YOLO detections → BoT-SORT → a player assigner (sticky IDs, short bbox prediction, court-half fallback). 
Tracker YAML presets live in [`src/tracking/configs/`](src/tracking/configs/).

### Tracking modes

Set `--tracking-mode` or `tracking.mode` in `config.yaml`:


| Mode | When to use |
| ---- | ----------- |
| `strong` (default) | Full matches; occlusion and similar uniforms. BoT-SORT + Re-ID + prediction + court fallback. |
| `normal` | Faster; lighter tracking when detections are already clean. |
| `court-side-fallback` | Red/blue keep swapping — ignore tracker IDs; assign by court half every frame. |

### Player 1 court position

Set `--player1-position` or `assignment.player1_position` to match your camera:


| Value | Player 1 | Player 2 |
| ----- | -------- | -------- |
| `top` (default) | upper half | lower half |
| `left` / `right` | left / right half | opposite half |

### Quick tips

- Start with **`--tracking-mode strong`** (default).
- Match **`player1_position`** to your camera (end-on → `top`; side-on → `left`/`right`).
- Use **`--debug-ids`** once; churning IDs → raise **`--min-confidence`** or fine-tune YOLO.
- Fine-tuned **`models/yolo_player/yolo_player_best.pt`** reduces ID swaps more than tracker tweaks alone.
- **`strong`** mode uses `models/yolo_player/yolo26n-cls.pt` for Re-ID (not the repo root).

All tuning knobs are under `tracking:` and `assignment:` in `config.yaml`.

---

## Label your own data with CVAT

We use **CVAT for Video 1.1** because it natively supports per-frame attributes on tracked bounding boxes.

### One-time task setup

The converter reads label and attribute names from `config.yaml` under the `cvat:` section. Two conventions are supported out of the box:

#### Convention A — distinct player labels (default in docs)

1. **Create a CVAT task** with your match video.
2. **Define two labels** (each as a `Bounding box` with `Track`):
   - `player1`
   - `player2`
3. Add a per-frame **mutable attribute** named exactly `split_step`
   (Type: `Number` with values `0,1` works equally well — the converter accepts either).

Set in `config.yaml`:

```yaml
cvat:
  player1_label: player1
  player2_label: player2
  split_attribute: split_step
```

#### Convention B — single shared label

1. **Create a CVAT task** with your rally video.
2. **Define one label** (as a `Bounding box` with `Track`):
   - `Player`
3. Draw **two tracks** with that same label — one per player.
4. Add a per-frame **mutable attribute** on the label, e.g. `movement_state`
   with values `normal` and `split_step`.

Set in `config.yaml`:

```yaml
cvat:
  player1_label: Player
  player2_label: Player
  split_attribute: movement_state
```

When both `player1_label` and `player2_label` are the same, the converter assigns player slots by **CVAT track id**: track `0` → player 1 (red), track `1` → player 2 (blue). Draw the upper-court player first so track order matches inference.

The converter coerces attribute values into binary action labels (`0` = normal, `1` = split step). Accepted value spellings include:

| Meaning | Accepted values |
| ------- | ---------------- |
| Normal | `0`, `false`, `no`, `normal` |
| Split step | `1`, `true`, `yes`, `split`, `split_step` |

In CVAT's label editor, choose **Mutable: yes** so the attribute can change per frame.

### Labeling workflow

1. Draw a track around the upper-court player at frame 0, then move forward and CVAT will interpolate. Adjust keyframes whenever the box drifts. (Convention A: label `player1`. Convention B: label `Player`, first track.)
2. Repeat for the lower-court player. (Convention A: `player2`. Convention B: second `Player` track.)
3. **For every frame** (or every keyframe — see the converter's `clip_stride`), set the action attribute when the player is in a split step:
   - Convention A: `split_step = 1`, otherwise `0`
   - Convention B: `movement_state = split_step`, otherwise `normal`
4. Save often.

### Export

`Tasks → ⋯ → Export task dataset → CVAT for video 1.1`. The converter **accepts the resulting `.zip` directly — no manual unzipping required**.

The recommended layout pairs each video with its CVAT export by base name:

```
data/raw/rally_001.mp4
data/cvat/rally_001_cvat.zip
```

If you'd rather keep the raw `annotations.xml`, that works too — the converter accepts either `.zip` or `.xml`.

> The converter is tolerant of `outside="1"` keyframes and missing attributes, but it skips frames that have neither.

---

## Convert CVAT exports → training datasets

The single **CVAT converter** (`[src/data/cvat_converter.py](src/data/cvat_converter.py)`) drives both training pipelines and accepts CVAT `.zip` or raw `annotations.xml` interchangeably.

### Auto mode (recommended)

Drop files into the conventional layout:

```
data/raw/rally_001.mp4         data/cvat/rally_001_cvat.zip
data/raw/rally_002.mp4         data/cvat/rally_002_cvat.zip
...
```

then run:

```bash
python main.py convert-cvat --auto
```

The converter will:

1. Walk `data/raw/` and pair each video with `data/cvat/<stem>_cvat.zip`
  (or `<stem>_cvat.xml`, or `<stem>.zip`, or `<stem>.xml` — first match wins).
2. Auto-extract `annotations.xml` from each `.zip`.
3. Produce a single combined dataset:

```
data/yolo/
  data.yaml                                # includes train:, val:, test: keys
  images/{train,val,test}/<video>_f000123.jpg
  labels/{train,val,test}/<video>_f000123.txt   # YOLO: 0 cx cy w h

data/action/
  manifest.csv    # clip_id, video, player_id, center_frame, label, split
  clips/<clip_id>/00.jpg ... 15.jpg
```

Both datasets are split three ways. The defaults are **60% train / 20% val / 20% test** (controlled by `cvat.val_split` and `cvat.test_split` in `config.yaml`; the train share is implicit `1 - val - test`). The `split` column of `manifest.csv` holds one of `train`, `val`, or `test`, and the YOLO `data.yaml` includes a `test:` key for a held-out test split. Set `--test-split 0` (or `cvat.test_split: 0.0`) to reproduce the old two-way `train`/`val`-only behavior.

Override the file-naming suffix with `--suffix` (default `_cvat`):

```bash
python main.py convert-cvat --auto --suffix _annotations
```

### Single explicit pair

`.zip` is preferred (auto-extracted); `.xml` works too:

```bash
python main.py convert-cvat \
  --video data/raw/rally_001.mp4 \
  --cvat  data/cvat/rally_001_cvat.zip
```

If the CVAT file lives in `data/cvat/` and follows the naming convention, you can omit `--cvat` and the converter will look it up automatically:

```bash
python main.py convert-cvat --video data/raw/rally_001.mp4
```

### Restrict to one dataset

Pass `--mode yolo` or `--mode action` (default is `both`). The `--val-split` and `--test-split` flags override the corresponding `config.yaml` values (defaults: 0.2 each, giving a 60/20/20 split):

```bash
python main.py convert-cvat --auto --mode yolo  --val-split 0.2 --test-split 0.2 --every-n 1
python main.py convert-cvat --auto --mode action --clip-len 16 --val-split 0.2 --test-split 0.2
```

Both `player1` and `player2` tracks collapse into a single `player` class for YOLO so a vanilla single-class detector trains out of the box. For action, each labeled frame becomes a clip of `clip-len` *trailing* frames so the label corresponds to the most recent frame — the same convention used at inference time.

### CVAT label / attribute mapping

The converter does **not** hard-code CVAT names. It reads these keys from `config.yaml`:

| Key | Purpose | This project's value |
| --- | ------- | -------------------- |
| `cvat.player1_label` | Label (or first track) for player 1 | `Player` |
| `cvat.player2_label` | Label (or second track) for player 2 | `Player` |
| `cvat.split_attribute` | Per-frame box attribute to read | `movement_state` |

- **Distinct labels** (`player1` / `player2`): tracks are matched by label
  name.
- **Same label twice** (`Player` / `Player`): the first two matching tracks
  ordered by CVAT track id become player 1 and player 2.
- **`split_attribute`**: the named attribute on each box is parsed into
  class `0` (normal) or `1` (split step). YOLO export ignores this
  attribute and exports all player boxes regardless of label name.

If action export logs `no '<attribute>' targets found` for every video, check that your CVAT export uses the attribute name configured in `cvat.split_attribute` and that player labels match `config.yaml`.

---

## Train the YOLO detector

After running `convert-cvat --mode yolo` (or `--mode both`):

```bash
python main.py train-yolo \
  --data data/yolo/data.yaml \
  --base-model models/yolo_player/yolo26n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16
```

Training writes run artifacts under `models/yolo_player/` (`weights/best.pt`, `results.csv`, plots, …). When training finishes, **`train-yolo` automatically copies** `weights/best.pt` → `models/yolo_player/yolo_player_best.pt` for `analyze`.

---

## Train the split-step model

```bash
python main.py train-action \
  --manifest data/action/manifest.csv \
  --epochs 30 \
  --batch-size 16
```

Highlights:

- ResNet18 (frozen by default) → BiLSTM(128) → 2-class linear head.
- Class-weighted cross-entropy (badminton split steps are heavily
imbalanced — most frames are "normal").
- Cosine LR schedule, AdamW, AMP when CUDA is available.
- Best checkpoint by macro-F1 (on val) saved to
  `models/action_player/action_best.pt`.
- Per-epoch metrics dumped to `models/action_player/train_history.json`.
- **Training plots** saved to `models/action_player/training_curves.png` and
  (when a test split exists) `models/action_player/test_class_metrics.png`.
  Re-generate anytime with `python main.py plot-training`.
- **End-of-training test eval** — when the manifest contains a `test`
split, the best (by val F1) checkpoint is reloaded and scored on the
held-out test clips. Loss, accuracy, macro-F1, and a per-class
precision/recall report are persisted to `models/action_player/test_metrics.json`.

`config.yaml` exposes every relevant knob (clip length, backbone, LSTM size, augmentations, etc.).

---

## Configuration reference

`config.yaml` is the single source of truth. CLI flags override individual fields. Sections:

- `device` — `auto | cpu | cuda | mps` (`auto` prefers CUDA, then Apple MPS, then CPU)
- `pipeline` — `frame_skip`, `output_fps`, `draw_hud`
- `detection` — YOLO model paths, conf/iou thresholds, `imgsz`, class
filter, `max_det`
- `tracking` — `tracker_yaml` (`botsort.yaml` | `bytetrack.yaml`)
- `assignment` — `top_is_player1`, `reassign_after_lost_frames`
- `action` — backbone, clip length, LSTM size, freeze-backbone toggle
- `smoothing` — EMA `α`, hysteresis `prob_on/off`, `min_on_frames`,
`cooldown_frames`
- `train_action` / `train_yolo` — hyperparameters
- `cvat` — CVAT → dataset mapping:
  - `player1_label` / `player2_label` — track label(s) in your CVAT task.
    Use the same value for both when every player track shares one label
    (e.g. `Player`); player slots are then assigned by track id (`0` →
    player 1, `1` → player 2).
  - `split_attribute` — per-frame box attribute that encodes split step vs.
    normal (e.g. `movement_state` with values `normal` / `split_step`, or
    `split_step` with values `0` / `1`).
  - `val_split` + `test_split` (defaults 0.2 + 0.2 ⇒ 60/20/20), frame
    subsampling via `every_n_frames`

---

## License

MIT