#!/usr/bin/env python3
"""Analyze selected ROV keyframes with a local Vision-Language Model (mlx-vlm)."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rov_inspect.config import load_section
from rov_inspect.paths import resolve_path
from rov_inspect.schema import (
    BOOLEAN_PRESENCE_FIELDS,
    STATUS_FIELDS,
    normalize_annotation,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
QWEN3_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
GEMMA_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
ALLOWED_MODELS = [QWEN3_MODEL, GEMMA_MODEL]
DEFAULT_MODEL = QWEN3_MODEL

REPORT_FIELDS = [
    "image_path",
    "image_name",
    "timestamp_sec",
    "substrate",
    "rocks_present",
    "cobbles_present",
    *BOOLEAN_PRESENCE_FIELDS,
    *STATUS_FIELDS,
    "rov_equipment_type",
    "water_visibility",
    "inspection_importance",
    "short_description",
    "uncertainty",
    "model_name",
    "raw_model_output",
]

VLM_PROMPT = """
You are analyzing one frame from an underwater ROV seabed inspection video.

Describe only what is visible in the image. Do not invent species. Do not invent objects.
Focus on seabed type, rocks/cobbles, algae, fauna, waste, structures, and visibility.

Use "clear" only when the element is clearly visible.
Use "possible" when the element may be present but is ambiguous.
Use "none" when there is no visual evidence.
Do not mark fauna as "clear" unless an animal is visibly identifiable.
Do not mark structure as "clear" unless there is an obvious man-made fixed structure.
Do not mark waste as "clear" unless there is an obvious anthropogenic item.
If a visible cable, tether, robot body part, light rig, or inspection equipment appears to belong to the ROV system, mark it as ROV equipment.
Do not count ROV equipment as environmental waste.
Do not count ROV equipment as a man-made structure.
Use waste_status only for possible or clear debris/anthropogenic material in the environment.
Use structure_status only for possible or clear fixed man-made structures in the inspected environment.
If unsure whether a cable belongs to the ROV or the environment, set rov_equipment_status = "possible" and mention the uncertainty in short_description.
If unsure, prefer "possible" instead of "clear".
Use substrate = "mixed" when loose sediment and rocks/boulders are both important in the frame.
Use low uncertainty only when visibility is clear and labels are obvious.
Do not guess species.

Return only valid JSON with this schema:

{
  "substrate": "sand | gravel | rocks | mixed | unclear",
  "rocks_present": true,
  "cobbles_present": true,
  "algae_status": "none | possible | clear",
  "waste_status": "none | possible | clear",
  "fauna_status": "none | possible | clear",
  "structure_status": "none | possible | clear",
  "rov_equipment_status": "none | possible | clear",
  "rov_equipment_type": "none | tether | cable | robot_part | other",
  "water_visibility": "good | medium | poor | unclear",
  "inspection_importance": "low | medium | high",
  "short_description": "one short sentence",
  "uncertainty": "low | medium | high"
}
""".strip()


@dataclass
class LocalVLM:
    """Loaded mlx-vlm model bundle."""

    model: object
    processor: object
    config: dict[str, Any]
    model_name: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze keyframe images with a local mlx-vlm model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, type=Path, help="YAML config with default parameter values.")
    parser.add_argument("--images-dir", default=None, type=Path, help="Folder containing keyframe images.")
    parser.add_argument("--output-dir", default=None, type=Path, help="Folder for frame report outputs.")
    parser.add_argument(
        "--model-name", default=DEFAULT_MODEL, choices=ALLOWED_MODELS, help="Local mlx-vlm model name."
    )
    parser.add_argument("--limit", default=None, type=int, help="Analyze at most this many images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing report files.")
    parser.add_argument("--dry-run", action="store_true", help="Print images that would be analyzed only.")
    parser.add_argument("--max-tokens", default=512, type=int, help="Maximum generation tokens per image.")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    config_path = _peek_config_path()
    if config_path is not None:
        parser.set_defaults(**_yaml_defaults(config_path))

    args = parser.parse_args()
    for required in ("images_dir", "output_dir"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required (set it on the command line or in the YAML config)")
    return args


def _peek_config_path() -> Path | None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, type=Path)
    pre_args, _ = pre_parser.parse_known_args()
    return pre_args.config


def _yaml_defaults(config_path: Path) -> dict:
    defaults = load_section(config_path, section="vlm")
    for key in ("images_dir", "output_dir", "config"):
        if key in defaults and defaults[key] is not None:
            defaults[key] = Path(defaults[key])
    return defaults


def main() -> int:
    args = parse_args()
    try:
        return _run(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    images_dir = resolve_path(args.images_dir)
    output_dir = resolve_path(args.output_dir)
    image_paths = find_images(images_dir)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError(f"No frame images found under: {images_dir}")

    if args.dry_run:
        print(f"Dry run: {len(image_paths)} image(s) would be analyzed")
        for image_path in image_paths:
            print(image_path)
        return 0

    ensure_output_ready(output_dir, overwrite=args.overwrite)
    vlm = load_local_vlm(args.model_name)

    iterable = tqdm(image_paths, desc="Analyzing", unit="image") if tqdm is not None else image_paths
    records = [
        analyze_one_image(image_path=path, vlm=vlm, max_tokens=args.max_tokens)
        for path in iterable
    ]

    write_outputs(records, output_dir)
    print(f"Images analyzed: {len(records)}")
    print(f"Output directory: {output_dir}")
    for name in ("frame_reports.jsonl", "frame_reports.json", "frame_reports.csv", "frame_reports.md"):
        print(f"{name.split('.')[-1].upper()}: {output_dir / name}")
    return 0


def find_images(images_dir: Path) -> list[Path]:
    """Return frame image paths in natural filename order."""

    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")

    paths = [
        path
        for path in images_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "contact_sheet" not in path.name.lower()
    ]
    return sorted(paths, key=natural_sort_key)


def analyze_one_image(image_path: Path, vlm: LocalVLM, max_tokens: int) -> dict[str, Any]:
    """Run the VLM on one image and return a fully-normalized record."""

    try:
        raw_output = generate_with_vlm(image_path, vlm=vlm, max_tokens=max_tokens)
        annotation = extract_json_object(raw_output)
    except Exception as exc:
        raw_output = str(exc)
        annotation = normalize_annotation({})
        annotation["short_description"] = "VLM analysis failed."
        annotation["uncertainty"] = "high"

    return {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "timestamp_sec": parse_timestamp_sec(image_path.name),
        **annotation,
        "model_name": vlm.model_name,
        "raw_model_output": raw_output,
    }


def generate_with_vlm(image_path: Path, vlm: LocalVLM, max_tokens: int) -> str:
    """Send one image to the local mlx-vlm model and return raw text."""

    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    formatted_prompt = apply_chat_template(vlm.processor, vlm.config, VLM_PROMPT, num_images=1)
    output = generate(
        vlm.model,
        vlm.processor,
        formatted_prompt,
        image=[str(image_path)],
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    return str(output.text) if hasattr(output, "text") else str(output)


def load_local_vlm(model_name: str = DEFAULT_MODEL) -> LocalVLM:
    """Load a local mlx-vlm model."""

    from mlx_vlm import load
    from mlx_vlm.utils import load_config

    print(f"Loading local VLM model: {model_name}")
    model, processor = load(model_name)
    return LocalVLM(model=model, processor=processor, config=load_config(model_name), model_name=model_name)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse direct JSON or extract the broadest {...} object, then normalize."""

    stripped = strip_code_fences(text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = _extract_braced_object(stripped)

    if not isinstance(parsed, dict):
        parsed = {}
    return normalize_annotation(parsed)


def _extract_braced_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


def default_report_record(image_path: Path, raw_model_output: str, model_name: str) -> dict[str, Any]:
    """Create a complete report record with safe annotation defaults."""

    return {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "timestamp_sec": parse_timestamp_sec(image_path.name),
        **normalize_annotation({}),
        "model_name": model_name,
        "raw_model_output": raw_model_output,
    }


def write_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    """Write JSONL, JSON, CSV, and Markdown reports."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frame_reports.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    (output_dir / "frame_reports.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (output_dir / "frame_reports.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    write_markdown_report(records, output_dir / "frame_reports.md")


def write_markdown_report(records: list[dict[str, Any]], path: Path) -> None:
    """Write a human-readable Markdown summary of every analyzed frame."""

    summary_fields = [
        "substrate",
        "rocks_present",
        "cobbles_present",
        "algae_status",
        "waste_status",
        "fauna_status",
        "structure_status",
        "rov_equipment_status",
        "rov_equipment_type",
        "water_visibility",
        "inspection_importance",
        "uncertainty",
        "short_description",
    ]
    lines: list[str] = ["# Frame Reports\n"]
    for record in records:
        lines.append(f"## {record['image_name']}")
        timestamp = record["timestamp_sec"]
        lines.append("Timestamp: n/a\n" if timestamp is None else f"Timestamp: {timestamp}s\n")
        relative_image = os.path.relpath(record["image_path"], start=path.parent)
        lines.append(f"![frame]({relative_image})\n")
        for field in summary_fields:
            lines.append(f"- {field}: {record[field]}")
        lines.append("")
        lines.append("JSON:")
        lines.append("```json")
        lines.append(json.dumps(record, indent=2, ensure_ascii=False))
        lines.append("```\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def ensure_output_ready(output_dir: Path, overwrite: bool) -> None:
    """Create the output directory and refuse to clobber existing reports."""

    existing = [
        output_dir / name
        for name in ("frame_reports.jsonl", "frame_reports.json", "frame_reports.csv", "frame_reports.md")
        if (output_dir / name).exists()
    ]
    if existing and not overwrite:
        raise RuntimeError(
            "Report outputs already exist. Use --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_timestamp_sec(filename: str) -> float | None:
    """Parse timestamps like t00012.0 or t12.5 from a filename."""

    match = re.search(r"_t(\d+(?:\.\d+)?)", filename) or re.search(r"\bt(\d+(?:\.\d+)?)", filename)
    return float(match.group(1)) if match else None


def natural_sort_key(path: Path) -> list[int | str]:
    """Sort paths naturally by numeric chunks in their relative string."""

    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def strip_code_fences(text: str) -> str:
    """Remove leading ```json / ``` fences and trailing ``` from a raw response."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:].strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    return stripped


if __name__ == "__main__":
    raise SystemExit(main())
