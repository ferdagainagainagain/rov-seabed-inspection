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

from rov_inspect.config import load_section
from rov_inspect.paths import resolve_path
from rov_inspect.schema import (
    BOOLEAN_PRESENCE_FIELDS,
    ENVIRONMENTAL_STATUS_FIELDS,
    STATUS_FIELDS,
    normalize_report,
)

CSV_FIELDS = [
    "final_id",
    "original_image_path",
    "final_image_path",
    "image_name",
    "timestamp_sec",
    "substrate",
    "rocks_present",
    "cobbles_present",
    *STATUS_FIELDS,
    "rov_equipment_type",
    *BOOLEAN_PRESENCE_FIELDS,
    "water_visibility",
    "inspection_importance",
    "uncertainty",
    "short_description",
    "keep_reason",
]

MARKDOWN_DETAIL_FIELDS = [
    ("Short description", "short_description"),
    ("Substrate", "substrate"),
    ("Rocks present", "rocks_present"),
    ("Cobbles present", "cobbles_present"),
    ("Algae status", "algae_status"),
    ("Waste status", "waste_status"),
    ("Fauna status", "fauna_status"),
    ("Structure status", "structure_status"),
    ("ROV equipment status", "rov_equipment_status"),
    ("ROV equipment type", "rov_equipment_type"),
    ("Water visibility", "water_visibility"),
    ("Inspection importance", "inspection_importance"),
    ("Uncertainty", "uncertainty"),
    ("Keep reason", "keep_reason"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a final deterministic ROV Markdown report from frame_reports.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, type=Path, help="YAML config with default parameter values.")
    parser.add_argument("--frame-reports", default=None, type=Path, help="Path to frame_reports.json.")
    parser.add_argument("--output-dir", default=None, type=Path, help="Folder for final report outputs.")
    parser.add_argument("--title", default="ROV Seabed Inspection Summary", help="Markdown report title.")
    parser.add_argument(
        "--merge-window-sec", default=20.0, type=float, help="Merge adjacent identical frames within this window."
    )
    parser.add_argument(
        "--max-gap-sec", default=90.0, type=float, help="Keep at least one representative every this many seconds."
    )
    parser.add_argument(
        "--copy-final-frames", action="store_true", help="Copy final images into output_dir/final_frames."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing final report outputs.")

    config_path = _peek_config_path()
    if config_path is not None:
        parser.set_defaults(**_yaml_defaults(config_path))

    args = parser.parse_args()
    for required in ("frame_reports", "output_dir"):
        if getattr(args, required) is None:
            parser.error(f"--{required.replace('_', '-')} is required (set it on the command line or in the YAML config)")
    return args


def _peek_config_path() -> Path | None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, type=Path)
    pre_args, _ = pre_parser.parse_known_args()
    return pre_args.config


def _yaml_defaults(config_path: Path) -> dict:
    defaults = load_section(config_path, section="synthesize")
    for key in ("frame_reports", "output_dir", "config"):
        if key in defaults and defaults[key] is not None:
            defaults[key] = Path(defaults[key])
    return defaults


def main() -> int:
    args = parse_args()
    try:
        return _run(args)
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    frame_reports_path = resolve_path(args.frame_reports)
    output_dir = resolve_path(args.output_dir)
    ensure_output_ready(output_dir, overwrite=args.overwrite)

    all_reports = sort_reports(load_frame_reports(frame_reports_path))
    final_reports = select_final_frames(
        all_reports, merge_window_sec=args.merge_window_sec, max_gap_sec=args.max_gap_sec
    )
    copied_paths = copy_final_frames(final_reports, output_dir) if args.copy_final_frames else {}
    attach_final_paths(final_reports, copied_paths)

    synthesis = summarize_reports(all_reports, final_reports)
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
    contact_sheet = output_dir / "final_contact_sheet.jpg"
    if contact_sheet.exists():
        print(f"Final contact sheet: {contact_sheet}")
    return 0


def load_frame_reports(path: Path) -> list[dict[str, Any]]:
    """Load and normalize frame reports from JSON."""

    if not path.is_file():
        raise FileNotFoundError(f"frame_reports.json does not exist: {path}")
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
    """True for frames with possible/clear environmental findings or high importance."""

    return (
        report["inspection_importance"] == "high"
        or report["waste_status"] != "none"
        or report["fauna_status"] != "none"
        or report["structure_status"] != "none"
    )


def rov_equipment_visible(report: dict[str, Any]) -> bool:
    return report["rov_equipment_status"] == "clear"


def choose_representative(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the highest-priority report from a duplicate group."""

    if not group:
        raise ValueError("Cannot choose representative from an empty group")
    return sorted(group, key=_representative_priority)[0].copy()


def _representative_priority(report: dict[str, Any]) -> tuple[Any, ...]:
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
    reports: list[dict[str, Any]], merge_window_sec: float, max_gap_sec: float
) -> list[dict[str, Any]]:
    """Deduplicate adjacent semantic repeats while preserving coverage."""

    if merge_window_sec < 0:
        raise ValueError("--merge-window-sec must be non-negative")
    if max_gap_sec <= 0:
        raise ValueError("--max-gap-sec must be greater than zero")
    if not reports:
        return []

    selected: list[dict[str, Any]] = []
    group: list[dict[str, Any]] = [reports[0]]
    group_reason = "first_frame"

    for report in reports[1:]:
        previous = group[-1]
        same_signature = semantic_signature(report) == semantic_signature(previous)
        adjacent_delta = timestamp_delta(previous, report)
        coverage_delta = timestamp_delta(selected[-1] if selected else group[0], report)
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

        selected.append(_finalize_group(group, group_reason))
        group_reason = _next_group_reason(report, same_signature, coverage_delta, max_gap_sec)
        group = [report]

    selected.append(_finalize_group(group, group_reason))
    for final_id, report in enumerate(selected, start=1):
        report["final_id"] = final_id
    return selected


def _next_group_reason(
    report: dict[str, Any], same_signature: bool, coverage_delta: float | None, max_gap_sec: float
) -> str:
    if same_signature and coverage_delta is not None and coverage_delta >= max_gap_sec:
        return "max_gap_representative"
    if rov_equipment_visible(report):
        return "rov_equipment_visible"
    if is_important(report):
        return "important_status"
    return "semantic_change"


def _finalize_group(group: list[dict[str, Any]], reason: str) -> dict[str, Any]:
    representative = choose_representative(group)
    if len(group) > 1 and reason not in {
        "first_frame",
        "max_gap_representative",
        "important_status",
        "rov_equipment_visible",
    }:
        reason = "representative_of_duplicate_group"
    representative["keep_reason"] = reason
    return representative


def summarize_reports(all_reports: list[dict[str, Any]], final_reports: list[dict[str, Any]]) -> str:
    """Create a deterministic cautious synthesis paragraph."""

    substrate_counts = Counter(report["substrate"] for report in all_reports)
    visibility_counts = Counter(report["water_visibility"] for report in all_reports)

    lines = [
        f"{len(all_reports)} analyzed frames were reduced to {len(final_reports)} representative frames.",
        f"Dominant substrate labels across analyzed frames: {_format_counter(substrate_counts)}.",
        f"Water visibility distribution: {_format_counter(visibility_counts)}.",
        _presence_sentence(all_reports, "rocks_present", "Rocks"),
        _presence_sentence(all_reports, "cobbles_present", "Cobbles"),
        _status_sentence(all_reports, "algae_status", "algal cover", "possible algal cover", "clear algal cover"),
        _status_sentence(
            all_reports,
            "waste_status",
            "clear environmental debris",
            "possible environmental debris",
            "Clear environmental debris / anthropogenic items",
            no_finding_text="No clear environmental debris was detected.",
        ),
        _status_sentence(all_reports, "fauna_status", "fauna", "possible fauna", "clearly visible fauna"),
        _status_sentence(
            all_reports,
            "structure_status",
            "fixed man-made structures",
            "possible structure-like material",
            "Clear fixed man-made structures",
            no_finding_text="No fixed man-made structures were detected.",
        ),
        _rov_equipment_summary_line(all_reports),
    ]
    return "\n\n".join(lines)


def _presence_sentence(reports: list[dict[str, Any]], field: str, label: str) -> str:
    if any(report[field] for report in reports):
        return f"{label} were observed in the analyzed frames."
    return f"{label} were not marked as present in the analyzed frames."


def _status_sentence(
    reports: list[dict[str, Any]],
    field: str,
    label: str,
    possible_text: str,
    clear_text: str,
    no_finding_text: str | None = None,
) -> str:
    """Render a single status-field summary sentence."""

    counts = Counter(report[field] for report in reports)
    if counts["clear"] > 0:
        if counts["possible"] > 0:
            return (
                f"{clear_text.capitalize()} was marked in {counts['clear']} frame(s), "
                f"with {possible_text} in {counts['possible']} frame(s)."
            )
        return f"{clear_text.capitalize()} was marked in {counts['clear']} frame(s)."
    if counts["possible"] > 0:
        return f"{possible_text.capitalize()} was flagged in {counts['possible']} frame(s)."
    if no_finding_text is not None:
        return no_finding_text
    verb = "were" if label.endswith("s") else "was"
    return f"No {label} {verb} detected."


def _rov_equipment_summary_line(reports: list[dict[str, Any]]) -> str:
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
    """Write the final Markdown report."""

    report_path = output_dir / "final_report.md"
    original_md = frame_reports_path.with_suffix(".md")
    original_keyframe_folder = _infer_keyframe_folder(all_reports)
    final_frames_dir = output_dir / "final_frames"

    lines: list[str] = [f"# {title}\n", "## Source / Navigation\n"]
    lines.append(f"- Frame reports JSON: `{_relative_or_original(frame_reports_path, output_dir)}`")
    if original_md.exists():
        lines.append(f"- Full per-frame Markdown: `{_relative_or_original(original_md, output_dir)}`")
    if original_keyframe_folder is not None:
        lines.append(f"- Original keyframe folder: `{_relative_or_original(original_keyframe_folder, output_dir)}`")
    if copied_frames:
        lines.append(f"- Final frames folder: `{_relative_or_original(final_frames_dir, output_dir)}`")
    lines.append("\n## General Synthesis\n")
    lines.append(synthesis)
    lines.append("\n## Representative Keyframes\n")

    for report in final_reports:
        image_path = Path(report.get("final_image_path") or report["image_path"])
        relative = _relative_or_original(image_path, output_dir)
        lines.append(f"### {report['image_name']}\n")
        lines.append(f"- Timestamp: {_format_timestamp(report['timestamp_sec'])}")
        lines.append(f"- Image path: `{relative}`\n")
        lines.append(f"![frame]({relative})\n")
        for label, key in MARKDOWN_DETAIL_FIELDS:
            lines.append(f"- {label}: {report[key]}")
        lines.append("")

    lines.append("## Full Analysis Reference\n")
    lines.append(
        f"The complete per-frame analysis is available in: `{_relative_or_original(frame_reports_path, output_dir)}`"
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_final_csv(final_reports: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for report in final_reports:
            writer.writerow({field: report.get(field) for field in CSV_FIELDS})


def write_final_json(final_reports: list[dict[str, Any]], path: Path) -> None:
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
        suffix = source.suffix.lower() or ".jpg"
        timestamp = report["timestamp_sec"]
        if timestamp is None:
            destination = frames_dir / f"final_{final_id:04d}{suffix}"
        else:
            destination = frames_dir / f"final_{final_id:04d}_t{float(timestamp):07.1f}{suffix}"
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


def timestamp_delta(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if left["timestamp_sec"] is None or right["timestamp_sec"] is None:
        return None
    return abs(float(right["timestamp_sec"]) - float(left["timestamp_sec"]))


def ensure_output_ready(output_dir: Path, overwrite: bool) -> None:
    """Create output directory and remove or refuse to clobber existing outputs."""

    expected = [
        output_dir / name
        for name in (
            "final_report.md",
            "final_keyframes.csv",
            "final_frame_reports.json",
            "final_contact_sheet.jpg",
        )
    ]
    if not overwrite and any(path.exists() for path in expected):
        raise RuntimeError(f"Output files already exist in {output_dir}. Use --overwrite to replace them.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in expected:
            path.unlink(missing_ok=True)
        frames_dir = output_dir / "final_frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)


def _infer_keyframe_folder(reports: list[dict[str, Any]]) -> Path | None:
    paths = [Path(report["image_path"]).parent for report in reports if report.get("image_path")]
    if not paths:
        return None
    first = paths[0]
    return first if all(path == first for path in paths) else None


def _format_counter(counter: Counter) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{key}: {value}" for key, value in counter.most_common())


def _format_timestamp(timestamp: Any) -> str:
    return "n/a" if timestamp is None else f"{float(timestamp):.1f}s"


def _relative_or_original(path: Path, start: Path) -> str:
    try:
        return os.path.relpath(path, start=start)
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
