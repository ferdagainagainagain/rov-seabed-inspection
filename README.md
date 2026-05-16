# ROV Inspect

ROV seabed video inspection pipeline that selects representative frames, annotates them with a local VLM, and produces a Markdown inspection report.

The current clean pipeline is intentionally small and script based. The older experimental code under `old_repo/` is archived reference material and is not part of the current workflow.

## Pipeline Overview

### Stage 1: Keyframe Selection

Script: `scripts/select_keyframes.py`

Input: an ROV video, optionally with a depth CSV.

The selector:
- samples frames from the video
- applies optional depth filtering and basic quality filtering
- selects visually novel frames
- supports classical descriptors, DINOv3 embeddings, and hybrid novelty scoring
- writes selected JPEGs, `keyframes.csv`, and `contact_sheet.jpg`

### Stage 2: Per-Frame VLM Annotation

Script: `scripts/analyze_keyframes_vlm.py`

Input: a folder of selected keyframes.

The annotation step:
- analyzes each keyframe independently
- uses local `mlx-vlm`
- defaults to `mlx-community/Qwen3-VL-4B-Instruct-4bit`
- uses the status-based schema described below
- writes `frame_reports.jsonl`, `frame_reports.json`, `frame_reports.csv`, and `frame_reports.md`

### Stage 3: Final Report Synthesis

Script: `scripts/synthesize_report.py`

Input: `frame_reports.json`.

The synthesis step:
- sorts and groups adjacent semantically similar frame reports
- keeps conservative representatives so possible findings are not discarded
- separates environmental findings from visible ROV equipment or tether
- writes `final_report.md`, `final_keyframes.csv`, `final_frame_reports.json`, copied `final_frames/` when requested, and `final_contact_sheet.jpg` when possible

## Repository Structure

```text
scripts/select_keyframes.py        # video to selected keyframes
scripts/analyze_keyframes_vlm.py   # selected keyframes to per-frame VLM reports
scripts/synthesize_report.py       # frame reports to final Markdown report
src/rov_inspect/                   # shared video, feature, telemetry, and contact sheet helpers
demo/video7/                       # local demo run for Video 7
outputs/                           # generated local outputs, ignored by Git
old_repo/                          # archived previous implementation, ignored by Git
```

## Environment

The current local environment is:

```bash
old_repo/rovenv/bin/python
```

This keeps the current working setup reproducible. A cleaner standalone environment file can be created later.

The VLM stage uses `mlx-vlm`, so run it from a normal macOS terminal with Apple Metal access. Headless or sandboxed sessions may not be able to start the local model.

## Usage

### 1. Select Keyframes

Example for Video 7:

```bash
old_repo/rovenv/bin/python scripts/select_keyframes.py \
  --video 'data/messina/VIDEO 7/videos/2025-05-13_11-25-23_DEEP_TREKKER_SD.mp4' \
  --output outputs/keyframes_video7 \
  --descriptor-backend dino \
  --sample-every-sec 1.0 \
  --novelty-threshold 0.45 \
  --min-gap-sec 3 \
  --max-gap-sec 120 \
  --depth-csv 'data/messina/VIDEO 7/data/depth_log.csv' \
  --min-depth-m 1.0 \
  --depth-filter-mode boundary \
  --depth-boundary-sec 60 \
  --device auto
```

Expected output:

```text
outputs/keyframes_video7/<video_stem>/
  frame_0001_t00013.0.jpg
  keyframes.csv
  contact_sheet.jpg
```

### 2. Annotate Keyframes With Qwen3-VL

```bash
old_repo/rovenv/bin/python scripts/analyze_keyframes_vlm.py \
  --images-dir outputs/keyframes_video7/2025-05-13_11-25-23_DEEP_TREKKER_SD \
  --output-dir outputs/frame_reports/video7 \
  --overwrite
```

Expected output:

```text
outputs/frame_reports/video7/
  frame_reports.jsonl
  frame_reports.json
  frame_reports.csv
  frame_reports.md
```

### 3. Synthesize Final Report

```bash
old_repo/rovenv/bin/python scripts/synthesize_report.py \
  --frame-reports outputs/frame_reports/video7/frame_reports.json \
  --output-dir outputs/final_reports/video7 \
  --title "ROV Seabed Inspection Summary - Video 7" \
  --copy-final-frames \
  --overwrite
```

Expected output:

```text
outputs/final_reports/video7/
  final_report.md
  final_keyframes.csv
  final_frame_reports.json
  final_contact_sheet.jpg
  final_frames/
```

## Output Schema

The preferred interpretation fields are status fields:

- `algae_status`
- `waste_status`
- `fauna_status`
- `structure_status`

Allowed values:

- `clear`: clearly visible
- `possible`: ambiguous but worth flagging
- `none`: no visual evidence

`possible` is important for underwater ROV footage because visibility, lighting, turbidity, and partial occlusion can make biological and anthropogenic content uncertain.

The pipeline also distinguishes visible ROV equipment from environmental findings:

- `rov_equipment_status`: `none`, `possible`, or `clear`
- `rov_equipment_type`: `none`, `tether`, `cable`, `robot_part`, or `other`

ROV equipment or tether should not be interpreted as environmental debris or as a fixed man-made structure.

For backwards compatibility, the annotation step still writes boolean fields:

- `algae_present`
- `waste_present`
- `fauna_present`
- `structure_present`
- `rov_equipment_present`

These are derived from status values: `true` when status is `possible` or `clear`, otherwise `false`.

## Demo: Video 7

A copied demo run is available at:

```text
demo/video7/
```

Open the final report here:

```text
demo/video7/final_report/final_report.md
```

The demo includes copied keyframes, per-frame reports, and the final synthesized report. The original source video is large and is ignored by Git.

## Limitations

- Local VLM outputs should be interpreted cautiously.
- A `possible` finding is not a certain fact.
- Underwater visibility, turbidity, lighting, and partial occlusion can affect model predictions.
- The current pipeline is intentionally conservative and high-recall, so it may keep extra frames or flag ambiguous findings for human review.
