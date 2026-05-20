# Run — Analisi VIDEO 1

## 0. Attiva il venv

```bash
source .venv/bin/activate
```

## Pipeline completa in un comando

```bash
python scripts/run_pipeline.py --config configs/video1.yaml
```

Esegue in sequenza Stage 1 → 2 → 3. Si ferma e ritorna exit code != 0 se uno stage fallisce.

Per eseguire gli stage singolarmente:

## 1. Stage 1 — Selezione keyframes

```bash
python scripts/select_keyframes.py --config configs/video1.yaml
```

Output atteso:

```
outputs/keyframes_video1/2025-05-13_09-49-42_DEEP_TREKKER_SD/
  frame_0001_t00013.0.jpg
  keyframes.csv
  contact_sheet.jpg
```

## 2. Stage 2 — Annotazione VLM (Qwen3-VL via mlx-vlm)

```bash
python scripts/analyze_keyframes_vlm.py --config configs/video1.yaml
```

Output atteso:

```
outputs/frame_reports/video1/
  frame_reports.jsonl
  frame_reports.json
  frame_reports.csv
  frame_reports.md
```

## 3. Stage 3 — Sintesi report finale

```bash
python scripts/synthesize_report.py --config configs/video1.yaml
```

Output atteso:

```
outputs/final_reports/video1/
  final_report.md
  final_keyframes.csv
  final_frame_reports.json
  final_contact_sheet.jpg
  final_frames/
```

## 4. Valutazione contro il ground truth (opzionale)

Richiede `ground_truth/video1.yaml` compilato a mano.

```bash
python scripts/evaluate_against_ground_truth.py --config configs/video1.yaml
```

Output atteso:

```
outputs/evaluation/video1/
  evaluation_metrics.json
  evaluation.md
```

## Note

- Tutti i parametri vivono in `configs/video1.yaml`. Per cambiare una soglia o un percorso, modifica il YAML.
- Qualunque flag CLI sovrascrive il valore preso dal YAML, p.es. `--overwrite` o `--title "Altro titolo"`.
- Lo Stage 2 richiede `mlx-vlm` con Apple Metal: eseguilo da un terminale macOS normale, non headless.
- Per analizzare un altro video, copia `configs/video1.yaml` in `configs/videoN.yaml` e aggiorna i percorsi al suo interno.
