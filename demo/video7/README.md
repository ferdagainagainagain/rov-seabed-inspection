# Video 7 Demo

This folder contains a copied demo run of the current ROV inspection pipeline for Video 7.

## Contents

```text
demo/video7/
  input/          # copied source video, ignored by Git because it is large
  keyframes/      # Stage 1 selected frames, metadata CSV, and contact sheet
  frame_reports/  # Stage 2 per-frame Qwen3-VL reports
  final_report/   # Stage 3 final Markdown report and representative frames
```

Source video:

```text
2025-05-13_11-25-23_DEEP_TREKKER_SD.mp4
```

The copied source video is about 215 MB. It should be stored externally or included only in a submitted archive, not pushed to GitHub.

## Stage 1: Keyframe Selection

The keyframes were produced with DINO novelty scoring, boundary-only depth filtering, and conservative max-gap coverage:

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

Demo copy:

```text
demo/video7/keyframes/
```

Generated artifacts in this folder include selected JPEG frames, `keyframes.csv`, and `contact_sheet.jpg`.

## Stage 2: Per-Frame VLM Annotation

The frame reports were produced with the default local VLM:

```text
mlx-community/Qwen3-VL-4B-Instruct-4bit
```

Command:

```bash
old_repo/rovenv/bin/python scripts/analyze_keyframes_vlm.py \
  --images-dir outputs/keyframes_video7/2025-05-13_11-25-23_DEEP_TREKKER_SD \
  --output-dir outputs/frame_reports/video7 \
  --overwrite
```

Demo copy:

```text
demo/video7/frame_reports/
```

Generated artifacts in this folder include `frame_reports.jsonl`, `frame_reports.json`, `frame_reports.csv`, and `frame_reports.md`.

## Stage 3: Final Report Synthesis

The final report was produced from `frame_reports.json` without re-running the VLM:

```bash
old_repo/rovenv/bin/python scripts/synthesize_report.py \
  --frame-reports outputs/frame_reports/video7/frame_reports.json \
  --output-dir outputs/final_reports/video7 \
  --title "ROV Seabed Inspection Summary - Video 7" \
  --copy-final-frames \
  --overwrite
```

Open the final Markdown report here:

```text
demo/video7/final_report/final_report.md
```

Generated artifacts in this folder include `final_report.md`, `final_keyframes.csv`, `final_frame_reports.json`, `final_contact_sheet.jpg`, and `final_frames/`.

## Schema Notes

The preferred interpretation fields use status values:

- `none`: no visual evidence
- `possible`: ambiguous but worth flagging
- `clear`: clearly visible

ROV equipment or tether is separated from environmental debris and fixed structures where visible.
