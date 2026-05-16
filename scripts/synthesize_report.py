#!/usr/bin/env python3
"""Synthesize a final Markdown report from per-frame VLM reports."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

STATUS_VALUES = {"none", "possible", "clear"}
ROV_EQUIPMENT_TYPES = {"none", "tether", "cable", "robot_part", "other"}
STATUS_FIELDS = ["algae_status", "waste_status", "fauna_status", "structure_status", "rov_equipment_status"]
ENVIRONMENTAL_STATUS_FIELDS = ["algae_status", "waste_status", "fauna_status", "structure_status"]
BOOLEAN_STATUS_FIELDS = [
    "algae_present",
    "waste_present",
    "fauna_present",
    "structure_present",
    "rov_equipment_present",
]
CSV_FIELDS = [
    "final_id",
    "original_image_path",
    "final_image_path",
    "image_name",
    "timestamp_sec",
    "substrate",
    "rocks_present",
    "cobbles_present",
    "algae_status",
    "waste_status",
    "fauna_status",
    "structure_status",
    "rov_equipment_status",
    "rov_equipment_type",
    "rov_equipment_present",
    "water_visibility",
    "inspection_importance",
    "uncertainty",
    "short_description",
    "keep_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a final deterministic ROV Markdown report from frame_reports.json."
    )
    parser.add_argument("--frame-reports", required=True, type=Path, help="Path to frame_reports.json.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder for final report outputs.")
    parser.add_argument("--title", default="ROV Seabed Inspection Summary", help="Markdown report title.")
    parser.add_argument("--merge-window-sec", default=20.0, type=float, help="Merge adjacent identical frames within this window.")
    parser.add_argument("--max-gap-sec", default=90.0, type=float, help="Keep at least one representative every this many seconds.")
    parser.add_argument("--copy-final-frames", action="store_true", help="Copy final images into output_dir/final_frames.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing final report outputs.")
    parser.add_argument("--llm-backend", default="none", choices=["none", "gemma"], help="Optional text-only synthesis backend.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        frame_reports_path = _resolve_path(args.frame_reports)
        output_dir = _resolve_path(args.output_dir)
        ensure_output_ready(output_dir, overwrite=args.overwrite)

        all_reports = sort_reports(load_frame_reports(frame_reports_path))
        final_reports = select_final_frames(
            all_reports,
            merge_window_sec=args.merge_window_sec,
            max_gap_sec=args.max_gap_sec,
        )
        copied_paths = copy_final_frames(final_reports, output_dir) if args.copy_final_frames else {}
        attach_final_paths(final_reports, copied_paths)

        synthesis = summarize_reports(all_reports, final_reports)
        if args.llm_backend == "gemma":
            try:
                synthesis = rewrite_summary_with_gemma(synthesis)
            except NotImplementedError:
                print("Gemma text synthesis is not implemented; using deterministic synthesis.", file=sys.stderr)

        write_final_json(final_reports, output_dir / "final_frame_reports.json")
        write_final_csv(final_reports, output_dir / "final_keyframes.csv")
        write_final_report(
            frame_reports_path=frame_reports_path,
            output_dir=output_dir,
            title=args.title,
            all_reports=all_reports,
            final_reports=final_reports,
            synthesis=synthesis,
            copied_frames=args.copy_final_frames,
        )
        maybe_make_contact_sheet(final_reports, output_dir / "final_contact_sheet.jpg")

        print(f"Analyzed frames: {len(all_reports)}")
        print(f"Final representative frames: {len(final_reports)}")
        print(f"Output directory: {output_dir}")
        print(f"Final report: {output_dir / 'final_report.md'}")
        print(f"Final CSV: {output_dir / 'final_keyframes.csv'}")
        print(f"Final JSON: {output_dir / 'final_frame_reports.json'}")
        if (output_dir / "final_contact_sheet.jpg").exists():
            print(f"Final contact sheet: {output_dir / 'final_contact_sheet.jpg'}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def load_frame_reports(path: Path) -> list[dict[str, Any]]:
    """Load and normalize frame reports from JSON."""

    if not path.exists():
        raise FileNotFoundError(f"frame_reports.json does not exist: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"frame_reports path is not a file: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("frame_reports.json must contain a list of records")
    return [normalize_report(record) for record in data if isinstance(record, dict)]


def sort_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort reports by timestamp when available, otherwise by image name."""

    return sorted(
        reports,
        key=lambda report: (
            report["timestamp_sec"] is None,
            float(report["timestamp_sec"] or 0.0),
            str(report["image_name"]),
        ),
    )


def semantic_signature(report: dict[str, Any]) -> tuple[Any, ...]:
    """Return the semantic fields used for adjacent-frame deduplication."""

    return (
        report["substrate"],
        report["rocks_present"],
        report["cobbles_present"],
        report["algae_status"],
        report["waste_status"],
        report["fauna_status"],
        report["structure_status"],
        report["rov_equipment_status"],
        report["rov_equipment_type"],
        report["water_visibility"],
    )


def is_important(report: dict[str, Any]) -> bool:
    """Return True for frames with possible/clear findings or high importance."""

    return (
        report["inspection_importance"] == "high"
        or report["waste_status"] != "none"
        or report["fauna_status"] != "none"
        or report["structure_status"] != "none"
    )


def rov_equipment_visible(report: dict[str, Any]) -> bool:
    """Return True when ROV equipment is clearly visible."""

    return report["rov_equipment_status"] == "clear"


def choose_representative(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the highest-priority report from a duplicate group."""

    if not group:
        raise ValueError("Cannot choose representative from an empty group")
    return sorted(group, key=representative_priority)[0].copy()


def representative_priority(report: dict[str, Any]) -> tuple[Any, ...]:
    importance_rank = {"high": 0, "medium": 1, "low": 2}
    uncertainty_rank = {"low": 0, "medium": 1, "high": 2}
    status_rank = {"clear": 0, "possible": 1, "none": 2}
    status_score = sum(status_rank[report[field]] for field in ENVIRONMENTAL_STATUS_FIELDS)
    timestamp = report["timestamp_sec"] if report["timestamp_sec"] is not None else float("inf")
    return (
        importance_rank.get(report["inspection_importance"], 1),
        uncertainty_rank.get(report["uncertainty"], 2),
        status_score,
        timestamp,
        report["image_name"],
    )


def select_final_frames(
    reports: list[dict[str, Any]],
    merge_window_sec: float,
    max_gap_sec: float,
) -> list[dict[str, Any]]:
    """Deduplicate adjacent semantic repeats while preserving coverage."""

    if merge_window_sec < 0:
        raise ValueError("--merge-window-sec must be non-negative")
    if max_gap_sec <= 0:
        raise ValueError("--max-gap-sec must be greater than zero")
    if not reports:
        return []

    selected: list[dict[str, Any]] = []
    group = [reports[0]]
    group_reason = "first_frame"

    for report in reports[1:]:
        previous = group[-1]
        same_signature = semantic_signature(report) == semantic_signature(previous)
        adjacent_delta = timestamp_delta(previous, report)
        last_selected = selected[-1] if selected else group[0]
        coverage_delta = timestamp_delta(last_selected, report)

        can_merge = (
            same_signature
            and adjacent_delta is not None
            and adjacent_delta <= merge_window_sec
            and not rov_equipment_visible(report)
            and not rov_equipment_visible(previous)
        )
        if can_merge and (coverage_delta is None or coverage_delta < max_gap_sec):
            group.append(report)
            continue

        selected.append(finalize_group(group, group_reason))
        if same_signature and coverage_delta is not None and coverage_delta >= max_gap_sec:
            group_reason = "max_gap_representative"
        elif rov_equipment_visible(report):
            group_reason = "rov_equipment_visible"
        elif is_important(report):
            group_reason = "important_status"
        else:
            group_reason = "semantic_change"
        group = [report]

    selected.append(finalize_group(group, group_reason))
    for final_id, report in enumerate(selected, start=1):
        report["final_id"] = final_id
    return selected


def finalize_group(group: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    representative = choose_representative(group)
    if len(group) > 1 and reason not in {"first_frame", "max_gap_representative", "important_status", "rov_equipment_visible"}:
        reason = "representative_of_duplicate_group"
    representative["keep_reason"] = reason
    return representative


def summarize_reports(all_reports: list[dict[str, Any]], final_reports: list[dict[str, Any]]) -> str:
    """Create deterministic cautious synthesis text."""

    substrate_counts = Counter(report["substrate"] for report in all_reports)
    visibility_counts = Counter(report["water_visibility"] for report in all_reports)
    substrate_summary = _format_counter(substrate_counts)
    visibility_summary = _format_counter(visibility_counts)

    lines = [
        f"{len(all_reports)} analyzed frames were reduced to {len(final_reports)} representative frames.",
        f"Dominant substrate labels across analyzed frames: {substrate_summary}.",
        f"Water visibility distribution: {visibility_summary}.",
    ]
    if any(report["rocks_present"] for report in all_reports):
        lines.append("Rocks were observed in the analyzed frames.")
    else:
        lines.append("Rocks were not marked as present in the analyzed frames.")
    if any(report["cobbles_present"] for report in all_reports):
        lines.append("Cobbles were observed in the analyzed frames.")
    else:
        lines.append("Cobbles were not marked as present in the analyzed frames.")

    lines.extend(status_summary_lines(all_reports))
    lines.append(rov_equipment_summary_line(all_reports))
    return "\n\n".join(lines)


def status_summary_lines(reports: list[dict[str, Any]]) -> list[str]:
    return [
        _status_sentence(reports, "algae_status", "algal cover", "possible algal cover", "clear algal cover"),
        _environmental_waste_sentence(reports),
        _status_sentence(reports, "fauna_status", "fauna", "possible fauna", "clearly visible fauna"),
        _environmental_structure_sentence(reports),
    ]


def rov_equipment_summary_line(reports: list[dict[str, Any]]) -> str:
    counts = Counter(report["rov_equipment_status"] for report in reports)
    types = Counter(
        report["rov_equipment_type"]
        for report in reports
        if report["rov_equipment_status"] != "none" and report["rov_equipment_type"] != "none"
    )
    type_text = ""
    if types:
        type_text = " Types: " + ", ".join(f"{name}: {count}" for name, count in types.most_common()) + "."
    if counts["clear"] > 0:
        return f"ROV equipment/tether was clearly visible in {counts['clear']} frame(s).{type_text}"
    if counts["possible"] > 0:
        return f"Possible ROV equipment/tether was visible in {counts['possible']} frame(s).{type_text}"
    return "No visible ROV equipment/tether was marked in the analyzed frames."


def write_final_report(
    frame_reports_path: Path,
    output_dir: Path,
    title: str,
    all_reports: list[dict[str, Any]],
    final_reports: list[dict[str, Any]],
    synthesis: str,
    copied_frames: bool,
) -> None:
    """Write final Markdown report."""

    report_path = output_dir / "final_report.md"
    original_md = frame_reports_path.with_suffix(".md")
    original_keyframe_folder = infer_keyframe_folder(all_reports)
    final_frames_dir = output_dir / "final_frames"

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("## Source / Navigation\n\n")
        handle.write(f"- Frame reports JSON: `{_relative_or_original(frame_reports_path, output_dir)}`\n")
        if original_md.exists():
            handle.write(f"- Full per-frame Markdown: `{_relative_or_original(original_md, output_dir)}`\n")
        if original_keyframe_folder is not None:
            handle.write(f"- Original keyframe folder: `{_relative_or_original(original_keyframe_folder, output_dir)}`\n")
        if copied_frames:
            handle.write(f"- Final frames folder: `{_relative_or_original(final_frames_dir, output_dir)}`\n")
        handle.write("\n## General Synthesis\n\n")
        handle.write(synthesis)
        handle.write("\n\n## Representative Keyframes\n\n")

        for report in final_reports:
            handle.write(f"### {report['image_name']}\n\n")
            handle.write(f"- Timestamp: {_format_timestamp(report['timestamp_sec'])}\n")
            image_path = Path(report.get("final_image_path") or report["image_path"])
            handle.write(f"- Image path: `{_relative_or_original(image_path, output_dir)}`\n\n")
            handle.write(f"![frame]({_relative_or_original(image_path, output_dir)})\n\n")
            handle.write(f"- Short description: {report['short_description']}\n")
            handle.write(f"- Substrate: {report['substrate']}\n")
            handle.write(f"- Rocks present: {report['rocks_present']}\n")
            handle.write(f"- Cobbles present: {report['cobbles_present']}\n")
            handle.write(f"- Algae status: {report['algae_status']}\n")
            handle.write(f"- Waste status: {report['waste_status']}\n")
            handle.write(f"- Fauna status: {report['fauna_status']}\n")
            handle.write(f"- Structure status: {report['structure_status']}\n")
            handle.write(f"- ROV equipment status: {report['rov_equipment_status']}\n")
            handle.write(f"- ROV equipment type: {report['rov_equipment_type']}\n")
            handle.write(f"- Water visibility: {report['water_visibility']}\n")
            handle.write(f"- Inspection importance: {report['inspection_importance']}\n")
            handle.write(f"- Uncertainty: {report['uncertainty']}\n")
            handle.write(f"- Keep reason: {report['keep_reason']}\n\n")

        handle.write("## Full Analysis Reference\n\n")
        handle.write(f"The complete per-frame analysis is available in: `{_relative_or_original(frame_reports_path, output_dir)}`\n")


def write_final_csv(final_reports: list[dict[str, Any]], path: Path) -> None:
    """Write final selected frame metadata CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for report in final_reports:
            writer.writerow({field: report.get(field) for field in CSV_FIELDS})


def write_final_json(final_reports: list[dict[str, Any]], path: Path) -> None:
    """Write final selected frame reports JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(final_reports, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_final_frames(final_reports: list[dict[str, Any]], output_dir: Path) -> dict[str, Path]:
    """Copy final frame images into output_dir/final_frames."""

    frames_dir = output_dir / "final_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Path] = {}
    for final_id, report in enumerate(final_reports, start=1):
        source = Path(report["image_path"])
        if not source.exists():
            continue
        timestamp = report["timestamp_sec"]
        suffix = source.suffix.lower() or ".jpg"
        if timestamp is None:
            name = f"final_{final_id:04d}{suffix}"
        else:
            name = f"final_{final_id:04d}_t{float(timestamp):07.1f}{suffix}"
        destination = frames_dir / name
        shutil.copy2(source, destination)
        copied[report["image_path"]] = destination
    return copied


def attach_final_paths(final_reports: list[dict[str, Any]], copied_paths: dict[str, Path]) -> None:
    for report in final_reports:
        copied = copied_paths.get(report["image_path"])
        report["original_image_path"] = report["image_path"]
        report["final_image_path"] = str(copied) if copied is not None else ""


def maybe_make_contact_sheet(final_reports: list[dict[str, Any]], output_path: Path) -> None:
    """Create a final contact sheet when OpenCV and image files are available."""

    try:
        import cv2

        from rov_inspect.contact_sheet import make_contact_sheet
    except ImportError:
        return

    frames = []
    labels = []
    for report in final_reports:
        image_path = Path(report.get("final_image_path") or report["image_path"])
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        frames.append(image)
        labels.append(f"{report['final_id']:04d} t={_format_timestamp(report['timestamp_sec'])}")
    if frames:
        make_contact_sheet(frames, labels, output_path)


def normalize_report(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize new status records and older boolean-only records."""

    normalized = dict(record)
    normalized["image_path"] = str(record.get("image_path", ""))
    normalized["image_name"] = str(record.get("image_name") or Path(normalized["image_path"]).name)
    normalized["timestamp_sec"] = _float_or_none(record.get("timestamp_sec"))
    normalized["substrate"] = _enum(record.get("substrate"), {"sand", "gravel", "rocks", "mixed", "unclear"}, "unclear")
    normalized["rocks_present"] = _bool(record.get("rocks_present"))
    normalized["cobbles_present"] = _bool(record.get("cobbles_present"))
    normalized["algae_status"] = _status(record.get("algae_status"), record.get("algae_present"))
    normalized["waste_status"] = _status(record.get("waste_status"), record.get("waste_present"))
    normalized["fauna_status"] = _status(record.get("fauna_status"), record.get("fauna_present"))
    normalized["structure_status"] = _status(record.get("structure_status"), record.get("structure_present"))
    normalized["rov_equipment_status"] = _status(record.get("rov_equipment_status"), record.get("rov_equipment_present"))
    normalized["rov_equipment_type"] = _enum(record.get("rov_equipment_type"), ROV_EQUIPMENT_TYPES, "none")
    if normalized["rov_equipment_status"] == "none":
        normalized["rov_equipment_type"] = "none"
    normalized["algae_present"] = normalized["algae_status"] != "none"
    normalized["waste_present"] = normalized["waste_status"] != "none"
    normalized["fauna_present"] = normalized["fauna_status"] != "none"
    normalized["structure_present"] = normalized["structure_status"] != "none"
    normalized["rov_equipment_present"] = normalized["rov_equipment_status"] != "none"
    normalized["water_visibility"] = _enum(record.get("water_visibility"), {"good", "medium", "poor", "unclear"}, "unclear")
    normalized["inspection_importance"] = _enum(record.get("inspection_importance"), {"low", "medium", "high"}, "medium")
    normalized["uncertainty"] = _enum(record.get("uncertainty"), {"low", "medium", "high"}, "high")
    normalized["short_description"] = str(record.get("short_description") or "")
    return normalized


def timestamp_delta(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if left["timestamp_sec"] is None or right["timestamp_sec"] is None:
        return None
    return abs(float(right["timestamp_sec"]) - float(left["timestamp_sec"]))


def infer_keyframe_folder(reports: list[dict[str, Any]]) -> Path | None:
    paths = [Path(report["image_path"]).parent for report in reports if report.get("image_path")]
    if not paths:
        return None
    first = paths[0]
    if all(path == first for path in paths):
        return first
    return None


def ensure_output_ready(output_dir: Path, overwrite: bool) -> None:
    existing = [
        output_dir / "final_report.md",
        output_dir / "final_keyframes.csv",
        output_dir / "final_frame_reports.json",
        output_dir / "final_contact_sheet.jpg",
    ]
    if not overwrite and any(path.exists() for path in existing):
        raise RuntimeError(f"Output files already exist in {output_dir}. Use --overwrite to replace them.")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            if path.exists():
                path.unlink()
        frames_dir = output_dir / "final_frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)


def rewrite_summary_with_gemma(summary: str) -> str:
    """Placeholder for optional text-only Gemma rewriting."""

    raise NotImplementedError("Optional Gemma text synthesis is not implemented yet")


def _status_sentence(reports: list[dict[str, Any]], field: str, label: str, possible_text: str, clear_text: str) -> str:
    counts = Counter(report[field] for report in reports)
    if counts["clear"] > 0:
        if counts["possible"] > 0:
            return f"{clear_text.capitalize()} was marked in {counts['clear']} frame(s), with {possible_text} in {counts['possible']} frame(s)."
        return f"{clear_text.capitalize()} was marked in {counts['clear']} frame(s)."
    if counts["possible"] > 0:
        return f"{possible_text.capitalize()} was flagged in {counts['possible']} frame(s)."
    verb = "were" if label.endswith("s") else "was"
    return f"No {label} {verb} detected."


def _environmental_waste_sentence(reports: list[dict[str, Any]]) -> str:
    counts = Counter(report["waste_status"] for report in reports)
    if counts["clear"] > 0:
        if counts["possible"] > 0:
            return f"Clear environmental debris / anthropogenic items were marked in {counts['clear']} frame(s), with possible debris in {counts['possible']} frame(s)."
        return f"Clear environmental debris / anthropogenic items were marked in {counts['clear']} frame(s)."
    if counts["possible"] > 0:
        return f"Possible environmental debris was flagged in {counts['possible']} frame(s)."
    return "No clear environmental debris was detected."


def _environmental_structure_sentence(reports: list[dict[str, Any]]) -> str:
    counts = Counter(report["structure_status"] for report in reports)
    if counts["clear"] > 0:
        if counts["possible"] > 0:
            return f"Clear fixed man-made structures were marked in {counts['clear']} frame(s), with possible structure-like material in {counts['possible']} frame(s)."
        return f"Clear fixed man-made structures were marked in {counts['clear']} frame(s)."
    if counts["possible"] > 0:
        return f"Possible structure-like material was flagged in {counts['possible']} frame(s)."
    return "No fixed man-made structures were detected."


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in counter.most_common())


def _format_timestamp(timestamp: Any) -> str:
    if timestamp is None:
        return "n/a"
    return f"{float(timestamp):.1f}s"


def _relative_or_original(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path, start=start)
    except ValueError:
        return str(path)


def _enum(value: Any, allowed: set[str], default: str) -> str:
    if isinstance(value, str) and value.strip().lower() in allowed:
        return value.strip().lower()
    return default


def _status(value: Any, legacy_boolean: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in STATUS_VALUES:
            return normalized
        if normalized:
            return "possible"
    if _bool(legacy_boolean):
        return "possible"
    return "none"


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return False


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
