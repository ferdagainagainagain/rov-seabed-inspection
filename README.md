# ROV Inspect

ROV seabed video inspection pipeline that selects representative frames, annotates them with a local VLM, and produces a Markdown inspection report.

The current clean pipeline is intentionally small and script based. The older experimental code under `old_repo/` is archived reference material and is not part of the current workflow.

## Pipeline Overview

The pipeline has three stages. They can be run individually or chained
together by `scripts/run_pipeline.py`. Each stage takes its parameters from a
per-video YAML config under `configs/` (CLI flags still work and override the
YAML).

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
scripts/run_pipeline.py                    # run Stage 1 -> 2 -> 3 in one command
scripts/select_keyframes.py                # Stage 1: video to selected keyframes
scripts/analyze_keyframes_vlm.py           # Stage 2: keyframes to per-frame VLM reports
scripts/synthesize_report.py               # Stage 3: frame reports to final Markdown report
scripts/evaluate_against_ground_truth.py   # compare VLM annotations to hand-labelled GT
configs/                                   # YAML config files, one per video
ground_truth/                              # hand-labelled GT YAML files, one per video
src/rov_inspect/                           # shared modules (schema, telemetry, features, ...)
demo/video7/                               # local demo run for Video 7
outputs/                                   # generated local outputs, ignored by Git
old_repo/                                  # archived previous implementation, ignored by Git
```

## Environment

Activate the project virtualenv before running anything:

```bash
source .venv/bin/activate
```

The VLM stage uses `mlx-vlm`, so run it from a normal macOS terminal with Apple Metal access. Headless or sandboxed sessions may not be able to start the local model.

## Usage

Each stage reads its parameters from a per-video YAML config under `configs/`.
CLI flags still work and override the values supplied in YAML.

Run the full pipeline (Stage 1 → 2 → 3) in one command:

```bash
python scripts/run_pipeline.py --config configs/video1.yaml
```

Or run the stages individually:

```bash
python scripts/select_keyframes.py     --config configs/video1.yaml
python scripts/analyze_keyframes_vlm.py --config configs/video1.yaml
python scripts/synthesize_report.py    --config configs/video1.yaml
```

The YAML has one section per stage (see `configs/video1.yaml` for a working
example):

```yaml
keyframes:
  video: data/VIDEO 1/videos/2025-05-13_09-49-42_DEEP_TREKKER_SD.mp4
  output: outputs/keyframes_video1
  descriptor_backend: dino
  novelty_threshold: 0.25
  depth_csv: data/VIDEO 1/data/depth_log.csv
  depth_filter_mode: boundary

vlm:
  images_dir: outputs/keyframes_video1/2025-05-13_09-49-42_DEEP_TREKKER_SD
  output_dir: outputs/frame_reports/video1
  overwrite: true

synthesize:
  frame_reports: outputs/frame_reports/video1/frame_reports.json
  output_dir: outputs/final_reports/video1
  title: ROV Seabed Inspection Summary - Video 1
  copy_final_frames: true
  overwrite: true
```

Each stage produces:

```text
outputs/keyframes_video1/<video_stem>/   # Stage 1: frame_NNNN_*.jpg, keyframes.csv, contact_sheet.jpg
outputs/frame_reports/video1/            # Stage 2: frame_reports.{jsonl,json,csv,md}
outputs/final_reports/video1/            # Stage 3: final_report.md, final_keyframes.csv,
                                         #          final_frame_reports.json, final_contact_sheet.jpg,
                                         #          final_frames/  (when copy_final_frames: true)
```

To run on a new video, copy `configs/video1.yaml` to `configs/videoN.yaml`
and edit the paths inside. Pass any flag on the command line to override a
YAML value, e.g. `--overwrite` or `--title "Other run"`.

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
