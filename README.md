# Badminton Singles — Split Step Timing Analysis

The split step is an essential badminton skill. Correct timing can help a player move faster.

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

## Result

![Annotated match demo](output_video.gif)

The top-left: frame index, timeline time, HUD, and tracking mode.

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
│   │   ├── smoothing.py
│   │   └── plots.py
│   ├── data/                       # I/O + the CVAT converter
│   │   ├── cvat_converter.py       # ★ primary CVAT bridge
│   │   └── video_io.py
│   ├── utils/                      # config, logging, geometry
│   └── visualization/              # bounding boxes, SPLIT STEP labels, HUD
├── models/                         # runtime checkpoints (gitignored)
├── trained_models/                 # release checkpoints (move into models/ after clone)
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

To download the stock YOLO checkpoint into `./models/`:

```bash
python scripts/download_yolo_weights.py            # -> models/yolo26n.pt
python scripts/download_yolo_weights.py yolo26n-cls.pt  # -> models/yolo26n-cls.pt (BoT-SORT Re-ID)
```

Verify your environment:

```bash
python main.py info
```

If you see `ModuleNotFoundError: No module named 'torch'`, activate `.venv` first.

### Trained models

Release checkpoints under `trained_models/`. The pipeline loads versioned runs from `models/`, so copy (or move) them after install:

```bash
mkdir -p models
cp -R trained_models/yolo_player_1 models/
cp -R trained_models/action_player_1 models/
```

After that you can run inference with `--yolo-run 1` and `--action-run 1` (see [Run inference](#run-inference)).

| Folder | Checkpoint used by analyze |
| ------ | -------------------------- |
| `models/yolo_player_1/` | `weights/best.pt` |
| `models/action_player_1/` | `action_best.pt` |

---

## Run inference

Annotate a video, producing `outputs/match_annotated.mp4`:

```bash
python main.py analyze \
  --video  data/raw/match.mp4 \
  --output outputs/match_annotated.mp4 \
  --yolo-run 1 \
  --action-run 1
```

- Input videos at any frame rate (25, 30, 60 FPS, …) are automatically resampled to **30 FPS** before being fed to the pipeline so the action model sees the same temporal cadence it was trained on. 
- The annotated output MP4 is written at the same target rate. 
- Override with `--target-fps <N>` or `pipeline.target_fps` in `config.yaml`; pass `--target-fps 0` (or set the config to `null`) to disable resampling and keep the source FPS.
- The top-left shows frame index, timeline time, **HUD** (processing FPS), and tracking mode. Processing FPS is inference throughput; it is not the video’s playback FPS.
- Use `--yolo-run N` and `--action-run N` to pick versioned checkpoints (`models/yolo_player_<N>/`, `models/action_player_<N>/`), or pass explicit `--yolo-weights` / `--action-weights` paths. If no checkpoint is found, `analyze` falls back to legacy flat paths in `config.yaml`, then the stock YOLO weights. If no action checkpoint is found, `analyze` still writes bounding boxes — the `SPLIT STEP` label simply never fires.

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
- Fine-tuned YOLO (e.g. **`--yolo-run 1`**) reduces ID swaps more than tracker tweaks alone.
- **`strong`** mode uses `models/yolo26n-cls.pt` for Re-ID (not the repo root).

All tuning knobs are under `tracking:` and `assignment:` in `config.yaml`.

---

## Label your own data with CVAT

Using **CVAT for Video 1.1** because it natively supports per-frame attributes on tracked bounding boxes.

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

The converter treats the attribute as binary: `1` = split step, `0` = normal (it also accepts common boolean/string variants like `true/false` and `yes/no`).
In CVAT, make the attribute **Mutable: yes**.

### Labeling workflow

1. Draw the tracks for player 1 and player 2 (CVAT interpolation will fill between keyframes).
2. For each labeled frame (or every Nth frame; see `action.clip_stride` at `convert-cvat` time), set the split-step attribute to `1` during split step and `0` otherwise.
3. Save often.

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

The converter will match each video in `data/raw/` with a CVAT export in `data/cvat/` (zip or xml), extract `annotations.xml`, and write:
- `data/yolo/` for detector training
- `data/action/` (including `manifest.csv` + clip images) for split-step training

`manifest.csv` contains:
- `label` — hard 0/1 label
- `target` — BCE target aligned to the causal clip’s final frame

Action clip density is controlled by `action.clip_stride` in `config.yaml` at **convert** time (not during `train-action`). Re-run `convert-cvat` after changing `clip_stride`.

Both datasets are split three ways. The defaults are **60% train / 20% val / 20% test** (`cvat.val_split` / `cvat.test_split`; train share is `1 - val - test`). With `group_split: true`, whole videos stay in one split. The `split` column of `manifest.csv` is `train`, `val`, or `test`. Set `--test-split 0` for train/val only.

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

YOLO exports both players as one `player` class. Action export builds trailing clips (same as inference) and uses the final frame's annotation for both the metric `label` and BCE `target`.

### CVAT label / attribute mapping

The converter doesn’t hard-code CVAT names; it reads these from `config.yaml`:
- `cvat.player1_label`
- `cvat.player2_label`
- `cvat.split_attribute`

If export logs `no '<attribute>' targets found`, your CVAT attribute name doesn’t match `cvat.split_attribute`.

---

## Train the YOLO detector

After running `convert-cvat --mode yolo` (or `--mode both`):

```bash
python main.py train-yolo \
  --data data/yolo/data.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 32
```

Each run is saved under an auto-incremented folder: `models/yolo_player_1/`, `models/yolo_player_2/`, … (`weights/best.pt`, Ultralytics plots, `args.yaml`). Gaps are preserved (if `_1` and `_3` exist, the next run is `_4`).

After training, pick a checkpoint with `--yolo-run N` in `analyze`, or pass `--yolo-weights models/yolo_player_<N>/weights/best.pt`.

---

## Train the split-step model

```bash
python main.py train-action \
  --manifest data/action/manifest.csv \
  --epochs 20 \
  --batch-size 32
```

Resume from a prior checkpoint with `--resume-run N` or `--resume path/to/action_best.pt` (weights only; other settings come from `config.yaml`):

```bash
python main.py train-action --manifest data/action/manifest.csv --resume-run 1 --epochs 10
```

### Highlights

- ResNet18 → BiLSTM → linear head; default loss is BCE (`action.num_classes: 1`).
- New runs load **ImageNet** backbone weights by default (`--no-pretrained` skips this).
- Optional early-stage freeze via `freeze_backbone_stages` (e.g. `[conv1, layer1, layer2]`); keep `freeze_batchnorm_stats: true` when using ImageNet.
- Differential LRs (`backbone_lr` / `head_lr`), AdamW, cosine schedule, EMA, and temperature calibration.
- Event-balanced sampling + event F1 (`split_step_event_f1`) for checkpoint / early stopping; `min_delta` ignores tiny gains.
- Soft boundary labels and clip-consistent `augmentation_*` apply to training only; val/test use hard labels.
- Each run writes `models/action_player_<N>/` (`action_best.pt`, metrics, plots, `run_info.json`). Re-plot with `python main.py plot-training --action-run 1`.

If video labels look too strict, lower `smoothing.prob_on` and compare against `best_threshold` in `run_info.json`.

---

## Configuration reference

`config.yaml` is the single source of truth. CLI flags override individual fields. Sections:

- `device` — `auto | cpu | cuda | mps` (`auto` prefers CUDA, then Apple MPS, then CPU)
- `pipeline` — `frame_skip`, `target_fps`, `output_fps`, `draw_hud`
- `detection` — YOLO model paths, conf/iou thresholds, `imgsz`, class
  filter, `max_det`
- `tracking` — `tracker_yaml` (`botsort.yaml` | `bytetrack.yaml`)
- `assignment` — `top_is_player1`, `reassign_after_lost_frames`
- `action` — backbone, `clip_length` / `clip_stride` (stride used at **convert-cvat** time), `num_classes`, LSTM size, `freeze_backbone`, `freeze_backbone_stages`, `freeze_batchnorm_stats`, `dropout`, `feature_dropout`
- `smoothing` — EMA `α`, hysteresis `prob_on` / `prob_off`, `min_on_frames`, `cooldown_frames` (inference only)
- `train_action` — hyperparameters (`loss`, `lr`, `backbone_lr`, `head_lr`, `best_metric`, `early_stopping_metric`, threshold sweep, EMA, temperature calibration, event-balanced sampling, class weights, `max_pos_weight`) and training-only `augmentation_*` controls
- `train_yolo` — `base_model`, `epochs`, `imgsz`, `batch` (runs write to `models/yolo_player_<N>/`)
- `cvat` — CVAT → dataset mapping:
  - `player1_label` / `player2_label` — track label(s) in your CVAT task.
    Use the same value for both when every player track shares one label
    (e.g. `Player`); player slots are then assigned by track id (`0` →
    player 1, `1` → player 2).
  - `split_attribute` — per-frame box attribute that encodes split step vs.
    normal (e.g. `movement_state` with values `normal` / `split_step`, or
    `split_step` with values `0` / `1`).
  - `centered_action_clips` — trailing vs centered clip windows
  - `val_split` + `test_split` (defaults 0.2 + 0.2 ⇒ 60/20/20), frame
    subsampling via `every_n_frames`

---

## License

MIT