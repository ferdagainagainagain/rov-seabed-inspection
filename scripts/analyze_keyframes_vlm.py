#!/usr/bin/env python3
"""Analyze selected ROV keyframes with a local VLM."""

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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
GEMMA_MODEL = "mlx-community/gemma-4-e4b-it-4bit"
QWEN3_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
DEFAULT_MODEL = QWEN3_MODEL
ALLOWED_MODELS = [QWEN3_MODEL, GEMMA_MODEL]

SUBSTRATES = {"sand", "gravel", "rocks", "mixed", "unclear"}
VISIBILITIES = {"good", "medium", "poor", "unclear"}
IMPORTANCE_LEVELS = {"low", "medium", "high"}
UNCERTAINTY_LEVELS = {"low", "medium", "high"}
STATUS_VALUES = {"none", "possible", "clear"}
ROV_EQUIPMENT_TYPES = {"none", "tether", "cable", "robot_part", "other"}
STATUS_FIELDS = [
    "algae_status",
    "waste_status",
    "fauna_status",
    "structure_status",
    "rov_equipment_status",
]
BOOLEAN_FIELDS = [
    "rocks_present",
    "cobbles_present",
    "algae_present",
    "waste_present",
    "fauna_present",
    "structure_present",
    "rov_equipment_present",
]
REPORT_FIELDS = [
    "image_path",
    "image_name",
    "timestamp_sec",
    "substrate",
    *BOOLEAN_FIELDS,
    *STATUS_FIELDS,
    "rov_equipment_type",
    "water_visibility",
    "inspection_importance",
    "short_description",
    "uncertainty",
    "model_name",
    "raw_model_output",
]

VLM_PROMPT_V1 = """
You are analyzing one frame from an underwater ROV seabed inspection video.

Describe only what is visible in the image. Do not guess species. Do not invent objects.
Focus on seabed type, rocks/cobbles, algae, fauna, waste, structures, and visibility.

Return only valid JSON with this schema:

{
  "substrate": "sand | gravel | rocks | mixed | unclear",
  "rocks_present": true,
  "cobbles_present": true,
  "algae_present": true,
  "waste_present": true,
  "fauna_present": true,
  "structure_present": true,
  "water_visibility": "good | medium | poor | unclear",
  "inspection_importance": "low | medium | high",
  "short_description": "one short sentence",
  "uncertainty": "low | medium | high"
}
""".strip()

VLM_PROMPT_V2 = """
You are analyzing one frame from an underwater ROV seabed inspection video.

Describe only what is visible in the image. Do not guess species. Do not invent species or objects.
Focus on seabed type, rocks/cobbles, algae, fauna, waste, structures, and visibility.

Use substrate = "mixed" when both loose sediment and large rocks/boulders are important.
Use low uncertainty only when the scene is clearly visible and labels are obvious.
If an object may be anthropogenic debris but is not certain, mention "possible debris" in short_description and set inspection_importance to high or medium.

Return only valid JSON with this schema:

{
  "substrate": "sand | gravel | rocks | mixed | unclear",
  "rocks_present": true,
  "cobbles_present": true,
  "algae_present": true,
  "waste_present": true,
  "fauna_present": true,
  "structure_present": true,
  "water_visibility": "good | medium | poor | unclear",
  "inspection_importance": "low | medium | high",
  "short_description": "one short sentence",
  "uncertainty": "low | medium | high"
}
""".strip()

VLM_PROMPT_V3 = """
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

PROMPTS = {
    "v1": VLM_PROMPT_V1,
    "v2": VLM_PROMPT_V2,
    "v3": VLM_PROMPT_V3,
}


@dataclass
class LocalVLM:
    """Loaded mlx-vlm model bundle."""

    model: object
    processor: object
    config: dict[str, Any]
    model_name: str


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Analyze keyframe images with a local VLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images-dir", required=True, type=Path, help="Folder containing keyframe images.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder for frame report outputs.")
    parser.add_argument("--model-backend", default="local", choices=["local"], help="VLM backend.")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        choices=ALLOWED_MODELS,
        help="Local mlx-vlm model name.",
    )
    parser.add_argument("--model", dest="model_name", choices=ALLOWED_MODELS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--prompt-version",
        default="v3",
        choices=["v3"],
        help="Prompt/schema version. Default: v3 status schema.",
    )
    parser.add_argument("--limit", default=None, type=int, help="Analyze at most this many images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing report files.")
    parser.add_argument("--dry-run", action="store_true", help="Print images that would be analyzed only.")
    parser.add_argument("--max-tokens", default=512, type=int, help="Maximum generation tokens per image.")
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        images_dir = _resolve_path(args.images_dir)
        output_dir = _resolve_path(args.output_dir)
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
        prompt = PROMPTS[args.prompt_version]
        vlm = load_local_vlm(args.model_name) if args.model_backend == "local" else None

        records: list[dict[str, Any]] = []
        iterable = tqdm(image_paths, desc="Analyzing", unit="image") if tqdm is not None else image_paths
        for image_path in iterable:
            records.append(
                analyze_one_image(
                    image_path=image_path,
                    prompt=prompt,
                    backend=args.model_backend,
                    vlm=vlm,
                    max_tokens=args.max_tokens,
                    model_name=args.model_name,
                )
            )

        write_outputs(records, output_dir)
        print(f"Images analyzed: {len(records)}")
        print(f"Output directory: {output_dir}")
        print(f"JSONL: {output_dir / 'frame_reports.jsonl'}")
        print(f"JSON: {output_dir / 'frame_reports.json'}")
        print(f"CSV: {output_dir / 'frame_reports.csv'}")
        print(f"Markdown: {output_dir / 'frame_reports.md'}")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, NotImplementedError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def find_images(images_dir: Path) -> list[Path]:
    """Return frame image paths in natural filename order."""

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {images_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images path is not a directory: {images_dir}")

    image_paths = [
        path
        for path in images_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "contact_sheet" not in path.name.lower()
    ]
    return sorted(image_paths, key=natural_sort_key)


def analyze_one_image(
    image_path: Path,
    prompt: str,
    backend: str,
    vlm: LocalVLM | None,
    max_tokens: int,
    model_name: str,
) -> dict[str, Any]:
    """Run VLM analysis for one image and return a safe record."""

    try:
        raw_output = analyze_image_with_vlm(
            image_path=image_path,
            prompt=prompt,
            backend=backend,
            vlm=vlm,
            max_tokens=max_tokens,
        )
        parsed = extract_json_object(raw_output)
        record = default_report_record(image_path, raw_output, model_name)
        record.update(parsed)
        return normalize_record(record)
    except Exception as exc:
        record = default_report_record(image_path, str(exc), model_name)
        record["short_description"] = "VLM analysis failed."
        record["uncertainty"] = "high"
        return normalize_record(record)


def analyze_image_with_vlm(
    image_path: Path,
    prompt: str,
    backend: str,
    vlm: LocalVLM | None,
    max_tokens: int = 512,
) -> str:
    """Send one image to the requested VLM backend and return raw text."""

    if backend != "local":
        raise NotImplementedError(f"Unsupported model backend: {backend}")
    if vlm is None:
        raise RuntimeError("Local VLM was not loaded")

    try:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template
    except RuntimeError as exc:
        raise RuntimeError(
            "mlx-vlm could not start. This local VLM backend requires Apple Metal access; "
            "run from a normal macOS terminal if this session is headless or sandboxed."
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "mlx-vlm is not installed. Install mlx-vlm in the active environment to use the local VLM."
        ) from exc

    formatted_prompt = apply_chat_template(vlm.processor, vlm.config, prompt, num_images=1)
    output = generate(
        vlm.model,
        vlm.processor,
        formatted_prompt,
        image=[str(image_path)],
        max_tokens=max_tokens,
        temperature=0.0,
        verbose=False,
    )
    if hasattr(output, "text"):
        return str(output.text)
    return str(output)


def load_local_vlm(model_name: str = DEFAULT_MODEL) -> LocalVLM:
    """Load a local mlx-vlm model."""

    try:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config
    except RuntimeError as exc:
        raise RuntimeError(
            "mlx-vlm could not start. This local VLM backend requires Apple Metal access; "
            "run from a normal macOS terminal if this session is headless or sandboxed."
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "mlx-vlm is not installed. Install mlx-vlm in the active environment to use the local VLM."
        ) from exc

    print(f"Loading local VLM model: {model_name}")
    model, processor = load(model_name)
    config = load_config(model_name)
    return LocalVLM(model=model, processor=processor, config=config, model_name=model_name)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse direct JSON or extract the broadest {...} JSON object."""

    stripped = strip_code_fences(text)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            parsed = {}
        else:
            try:
                parsed = json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                parsed = {}

    if not isinstance(parsed, dict):
        parsed = {}
    return normalize_annotation(parsed)


def normalize_annotation(parsed: dict[str, Any]) -> dict[str, Any]:
    """Validate annotation fields and fill safe defaults."""

    normalized: dict[str, Any] = {}
    normalized["substrate"] = _enum_value(parsed.get("substrate"), SUBSTRATES, "unclear")
    normalized["rocks_present"] = _bool_value(parsed.get("rocks_present"), default=False)
    normalized["cobbles_present"] = _bool_value(parsed.get("cobbles_present"), default=False)
    for status_field in STATUS_FIELDS:
        legacy_boolean_field = status_field.replace("_status", "_present")
        normalized[status_field] = _status_value(
            parsed.get(status_field),
            parsed.get(legacy_boolean_field),
        )
    normalized["algae_present"] = _status_present(normalized["algae_status"])
    normalized["waste_present"] = _status_present(normalized["waste_status"])
    normalized["fauna_present"] = _status_present(normalized["fauna_status"])
    normalized["structure_present"] = _status_present(normalized["structure_status"])
    normalized["rov_equipment_present"] = _status_present(normalized["rov_equipment_status"])
    normalized["rov_equipment_type"] = _enum_value(
        parsed.get("rov_equipment_type"),
        ROV_EQUIPMENT_TYPES,
        "none",
    )
    if normalized["rov_equipment_status"] == "none":
        normalized["rov_equipment_type"] = "none"
    normalized["water_visibility"] = _enum_value(parsed.get("water_visibility"), VISIBILITIES, "unclear")
    normalized["inspection_importance"] = _enum_value(
        parsed.get("inspection_importance"),
        IMPORTANCE_LEVELS,
        "medium",
    )
    normalized["short_description"] = str(parsed.get("short_description") or "")
    normalized["uncertainty"] = _enum_value(parsed.get("uncertainty"), UNCERTAINTY_LEVELS, "high")
    return normalized


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


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a complete record with only expected fields."""

    annotation = normalize_annotation(record)
    normalized = {
        "image_path": str(record.get("image_path", "")),
        "image_name": str(record.get("image_name", "")),
        "timestamp_sec": record.get("timestamp_sec"),
        **annotation,
        "model_name": str(record.get("model_name", "")),
        "raw_model_output": str(record.get("raw_model_output", "")),
    }
    return normalized


def write_outputs(records: list[dict[str, Any]], output_dir: Path) -> None:
    """Write JSONL, JSON, CSV, and Markdown outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "frame_reports.jsonl"
    json_path = output_dir / "frame_reports.json"
    csv_path = output_dir / "frame_reports.csv"
    md_path = output_dir / "frame_reports.md"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write("# Frame Reports\n\n")
        for record in records:
            handle.write(f"## {record['image_name']}\n")
            if record["timestamp_sec"] is None:
                handle.write("Timestamp: n/a\n\n")
            else:
                handle.write(f"Timestamp: {record['timestamp_sec']}s\n\n")
            relative_image = os.path.relpath(record["image_path"], start=output_dir)
            handle.write(f"![frame]({relative_image})\n\n")
            handle.write(f"- substrate: {record['substrate']}\n")
            handle.write(f"- rocks_present: {record['rocks_present']}\n")
            handle.write(f"- cobbles_present: {record['cobbles_present']}\n")
            handle.write(f"- algae_status: {record['algae_status']}\n")
            handle.write(f"- waste_status: {record['waste_status']}\n")
            handle.write(f"- fauna_status: {record['fauna_status']}\n")
            handle.write(f"- structure_status: {record['structure_status']}\n")
            handle.write(f"- rov_equipment_status: {record['rov_equipment_status']}\n")
            handle.write(f"- rov_equipment_type: {record['rov_equipment_type']}\n")
            handle.write(f"- water_visibility: {record['water_visibility']}\n")
            handle.write(f"- inspection_importance: {record['inspection_importance']}\n")
            handle.write(f"- uncertainty: {record['uncertainty']}\n")
            handle.write(f"- short_description: {record['short_description']}\n\n")
            handle.write("JSON:\n")
            handle.write("```json\n")
            handle.write(json.dumps(record, indent=2, ensure_ascii=False))
            handle.write("\n```\n\n")


def ensure_output_ready(output_dir: Path, overwrite: bool) -> None:
    """Create output directory and protect existing reports unless overwriting."""

    existing_files = [
        output_dir / "frame_reports.jsonl",
        output_dir / "frame_reports.json",
        output_dir / "frame_reports.csv",
        output_dir / "frame_reports.md",
    ]
    if not overwrite:
        existing = [path for path in existing_files if path.exists()]
        if existing:
            raise RuntimeError(
                "Report outputs already exist. Use --overwrite to replace them: "
                + ", ".join(str(path) for path in existing)
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def parse_timestamp_sec(filename: str) -> float | None:
    """Parse timestamps like t00012.0 or t12.5 from a filename."""

    match = re.search(r"_t(\d+(?:\.\d+)?)", filename)
    if not match:
        match = re.search(r"\bt(\d+(?:\.\d+)?)", filename)
    if not match:
        return None
    return float(match.group(1))


def natural_sort_key(path: Path) -> list[int | str]:
    """Sort paths naturally by numeric chunks in their relative string."""

    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped[3:].strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
        if stripped.endswith("```"):
            stripped = stripped[:-3].strip()
    return stripped


def _enum_value(value: Any, allowed: set[str], default: str) -> str:
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    if normalized in allowed:
        return normalized
    return default


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _status_value(value: Any, legacy_boolean: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in STATUS_VALUES:
            return normalized
        if normalized in {"false", "no", "0", "absent"}:
            return "none"
        if normalized:
            return "possible"
    if isinstance(value, bool):
        return "clear" if value else "none"

    legacy_present = _bool_value(legacy_boolean, default=False)
    if legacy_present:
        return "possible"
    return "none"


def _status_present(value: str) -> bool:
    return value in {"possible", "clear"}


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute() or path.exists():
        return path
    project_relative = PROJECT_ROOT / path
    if project_relative.exists():
        return project_relative
    return path


if __name__ == "__main__":
    raise SystemExit(main())
