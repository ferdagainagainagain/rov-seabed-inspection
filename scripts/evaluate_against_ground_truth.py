#!/usr/bin/env python3
"""Compare Stage 2 VLM annotations against a hand-annotated ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rov_inspect.config import load_section
from rov_inspect.paths import resolve_path

# VLM status categories that we can evaluate against the GT events.
CATEGORIES = ["fauna", "waste", "structure", "rov_equipment"]

# Map a GT `event.label` to the VLM `<category>_status` field.
GT_EVENT_TO_VLM_CATEGORY: dict[str, str] = {
    "fauna": "fauna",
    "waste": "waste",
    "structure": "structure",
    "rov_equipment": "rov_equipment",
}

VALID_SUBSTRATES = {"sand", "gravel", "cobbles", "rocks", "mud", "mixed", "unclear"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 2 VLM annotations against a hand-annotated ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, type=Path, help="YAML config (uses its `evaluation` section).")
    parser.add_argument("--ground-truth", default=None, type=Path, help="Ground-truth YAML.")
    parser.add_argument("--frame-reports", default=None, type=Path, help="Path to frame_reports.json.")
    parser.add_argument("--output-dir", default=None, type=Path, help="Folder for evaluation outputs.")

    args = parser.parse_args()
    if args.config is not None:
        defaults = load_section(args.config, section="evaluation")
        for attr, key in (("ground_truth", "ground_truth"), ("frame_reports", "frame_reports"), ("output_dir", "output_dir")):
            if getattr(args, attr) is None and key in defaults and defaults[key] is not None:
                setattr(args, attr, Path(defaults[key]))

    if args.ground_truth is None or args.frame_reports is None or args.output_dir is None:
        parser.error("--ground-truth, --frame-reports and --output-dir are required (CLI or YAML).")
    return args


def main() -> int:
    args = parse_args()
    try:
        return _run(args)
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace) -> int:
    gt_path = resolve_path(args.ground_truth)
    reports_path = resolve_path(args.frame_reports)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(gt_path)
    reports = load_frame_reports(reports_path)
    metrics = evaluate(gt, reports)

    json_path = output_dir / "evaluation_metrics.json"
    md_path = output_dir / "evaluation.md"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_paths = render_plots(metrics, gt, reports, output_dir)
    md_path.write_text(render_markdown(metrics, gt_path, reports_path, plot_paths), encoding="utf-8")

    print(f"Keyframes evaluated: {metrics['n_keyframes']}")
    print(f"GT events: {metrics['n_gt_events']}")
    print(f"Metrics JSON: {json_path}")
    print(f"Markdown:     {md_path}")
    for plot_path in plot_paths:
        print(f"Plot:         {plot_path}")
    return 0


def load_ground_truth(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Ground-truth root must be a mapping: {path}")
    for section in ("substrate", "events"):
        if data.get(section) is None:
            data[section] = []
    _validate_intervals(data["substrate"], VALID_SUBSTRATES, "substrate", path)
    _validate_events(data["events"], path)
    return data


def _validate_intervals(intervals: list, allowed_labels: set, name: str, path: Path) -> None:
    if not isinstance(intervals, list):
        raise ValueError(f"{name} section must be a list: {path}")
    for entry in intervals:
        if not isinstance(entry, dict):
            raise ValueError(f"{name} entry must be a mapping: {entry}")
        for required in ("start", "end", "label"):
            if required not in entry:
                raise ValueError(f"{name} entry missing '{required}': {entry}")
        if entry["label"] not in allowed_labels:
            raise ValueError(
                f"{name} label '{entry['label']}' is not in {sorted(allowed_labels)}"
            )
        if float(entry["start"]) > float(entry["end"]):
            raise ValueError(f"{name} entry has start > end: {entry}")


def _validate_events(events: list, path: Path) -> None:
    if not isinstance(events, list):
        raise ValueError(f"events section must be a list: {path}")
    for entry in events:
        if not isinstance(entry, dict):
            raise ValueError(f"event entry must be a mapping: {entry}")
        for required in ("start", "end", "label"):
            if required not in entry:
                raise ValueError(f"event missing '{required}': {entry}")
        if entry["label"] not in GT_EVENT_TO_VLM_CATEGORY:
            raise ValueError(
                f"event label '{entry['label']}' not recognized "
                f"(allowed: {sorted(GT_EVENT_TO_VLM_CATEGORY)})"
            )
        if float(entry["start"]) > float(entry["end"]):
            raise ValueError(f"event has start > end: {entry}")


def load_frame_reports(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"frame_reports.json must contain a list: {path}")
    return [record for record in data if isinstance(record, dict)]


def gt_labels_at(timestamp: float, gt: dict) -> dict[str, Any]:
    """Return active GT labels at a timestamp."""

    substrate = _label_at(timestamp, gt.get("substrate", []))
    active_categories: set[str] = set()
    for event in gt.get("events", []):
        if float(event["start"]) <= timestamp <= float(event["end"]):
            active_categories.add(GT_EVENT_TO_VLM_CATEGORY[event["label"]])
    return {"substrate": substrate, "categories": active_categories}


def _label_at(timestamp: float, intervals: list) -> str | None:
    for entry in intervals:
        if float(entry["start"]) <= timestamp <= float(entry["end"]):
            return entry["label"]
    return None


def evaluate(gt: dict, reports: list[dict]) -> dict[str, Any]:
    strict = {cat: _zero_counts() for cat in CATEGORIES}
    lenient = {cat: _zero_counts() for cat in CATEGORIES}
    substrate_confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    calibration: dict[str, dict[str, dict[str, int]]] = {
        cat: {status: {"hit": 0, "total": 0} for status in ("none", "possible", "clear")}
        for cat in CATEGORIES
    }

    n_keyframes = 0
    for report in reports:
        timestamp = report.get("timestamp_sec")
        if timestamp is None:
            continue
        n_keyframes += 1
        gt_at_t = gt_labels_at(float(timestamp), gt)

        for cat in CATEGORIES:
            gt_positive = cat in gt_at_t["categories"]
            vlm_status = report.get(f"{cat}_status", "none")
            _update(strict[cat], gt_positive, vlm_status == "clear")
            _update(lenient[cat], gt_positive, vlm_status in {"possible", "clear"})
            bucket = calibration[cat][vlm_status if vlm_status in {"none", "possible", "clear"} else "none"]
            bucket["total"] += 1
            if gt_positive:
                bucket["hit"] += 1

        gt_sub = gt_at_t["substrate"]
        vlm_sub = report.get("substrate")
        if gt_sub is not None and vlm_sub is not None:
            substrate_confusion[gt_sub][vlm_sub] += 1

    return {
        "n_keyframes": n_keyframes,
        "n_gt_events": len(gt.get("events", [])),
        "strict": {cat: _metrics(strict[cat]) for cat in CATEGORIES},
        "lenient": {cat: _metrics(lenient[cat]) for cat in CATEGORIES},
        "substrate_confusion": {gt_label: dict(preds) for gt_label, preds in substrate_confusion.items()},
        "substrate_accuracy": _diag_accuracy(substrate_confusion),
        "uncertainty_calibration": {
            cat: {
                status: {
                    "total": calibration[cat][status]["total"],
                    "gt_positive_rate": _safe_div(
                        calibration[cat][status]["hit"], calibration[cat][status]["total"]
                    ),
                }
                for status in ("none", "possible", "clear")
            }
            for cat in CATEGORIES
        },
    }


def _zero_counts() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _update(counts: dict[str, int], gt_positive: bool, pred_positive: bool) -> None:
    if gt_positive and pred_positive:
        counts["tp"] += 1
    elif gt_positive and not pred_positive:
        counts["fn"] += 1
    elif not gt_positive and pred_positive:
        counts["fp"] += 1
    else:
        counts["tn"] += 1


def _metrics(counts: dict[str, int]) -> dict[str, Any]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, tp + fp + fn + tn)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _diag_accuracy(confusion: dict[str, dict[str, int]]) -> float:
    total = sum(count for preds in confusion.values() for count in preds.values())
    correct = sum(preds.get(gt_label, 0) for gt_label, preds in confusion.items())
    return round(_safe_div(correct, total), 4)


def render_markdown(
    metrics: dict[str, Any],
    gt_path: Path,
    reports_path: Path,
    plot_paths: list[Path] | None = None,
) -> str:
    lines: list[str] = [
        "# Evaluation Against Ground Truth\n",
        f"- Ground truth: `{gt_path}`",
        f"- Frame reports: `{reports_path}`",
        f"- Keyframes evaluated: {metrics['n_keyframes']}",
        f"- GT events: {metrics['n_gt_events']}\n",
    ]
    if plot_paths:
        lines.append("## Plots\n")
        for plot_path in plot_paths:
            lines.append(f"![{plot_path.stem}]({plot_path.name})\n")
    for regime in ("strict", "lenient"):
        lines.append(f"## {regime.capitalize()} regime")
        lines.append(
            "Strict counts a keyframe as positive only when `status == 'clear'`. "
            "Lenient counts both `possible` and `clear`.\n"
            if regime == "strict"
            else ""
        )
        lines.append("| Category | Precision | Recall | F1 | Accuracy | TP | FP | FN | TN |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for cat in CATEGORIES:
            m = metrics[regime][cat]
            lines.append(
                f"| {cat} | {m['precision']} | {m['recall']} | {m['f1']} | "
                f"{m['accuracy']} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
            )
        lines.append("")

    lines.append(f"## Substrate (accuracy {metrics['substrate_accuracy']})")
    lines.append(_confusion_table(metrics["substrate_confusion"]))
    lines.append("")

    lines.append("## Uncertainty calibration")
    lines.append(
        "For each category, what fraction of keyframes labelled "
        "`none` / `possible` / `clear` by the VLM are actually positive in the GT? "
        "Ideal pattern: low for `none`, higher for `possible`, highest for `clear`.\n"
    )
    lines.append("| Category | none (rate / n) | possible (rate / n) | clear (rate / n) |")
    lines.append("|---|---|---|---|")
    for cat in CATEGORIES:
        cells = []
        for status in ("none", "possible", "clear"):
            bucket = metrics["uncertainty_calibration"][cat][status]
            rate = round(bucket["gt_positive_rate"], 4)
            cells.append(f"{rate} / {bucket['total']}")
        lines.append(f"| {cat} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")
    return "\n".join(lines)


def _confusion_table(confusion: dict[str, dict[str, int]]) -> str:
    if not confusion:
        return "_no data — GT and VLM did not overlap on any keyframe._"
    pred_labels = sorted({label for preds in confusion.values() for label in preds})
    header = "| GT \\ VLM | " + " | ".join(pred_labels) + " |"
    sep = "|" + "---|" * (len(pred_labels) + 1)
    rows = [header, sep]
    for gt_label in sorted(confusion):
        cells = [str(confusion[gt_label].get(pred, 0)) for pred in pred_labels]
        rows.append(f"| {gt_label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STATUS_COLORS = {"clear": "#2ca02c", "possible": "#ff7f0e", "none": "#bdbdbd"}
GT_FILL_COLOR = "#1f77b4"


def render_plots(
    metrics: dict[str, Any],
    gt: dict[str, Any],
    reports: list[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Render evaluation plots. Returns the list of created PNG paths (empty if matplotlib missing)."""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.", file=sys.stderr)
        return []

    created: list[Path] = []
    created.append(_plot_metrics_bar(metrics, output_dir / "metrics_bar.png", plt))
    confusion_path = _plot_substrate_confusion(metrics, output_dir / "substrate_confusion.png", plt)
    if confusion_path is not None:
        created.append(confusion_path)
    created.append(_plot_uncertainty_calibration(metrics, output_dir / "uncertainty_calibration.png", plt))
    created.append(_plot_timeline(gt, reports, output_dir / "timeline.png", plt))
    return created


def _plot_metrics_bar(metrics: dict[str, Any], path: Path, plt) -> Path:
    """Per-category prediction counts.

    For each category, two bars side by side (strict vs lenient regime) show how
    many keyframes the VLM flagged as positive, split into TP (correct) and FP
    (false positive). A dashed line marks the GT positive total (TP + FN), so the
    gap between the bar top and the line is the number of misses (FN).
    """

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    categories = CATEGORIES
    strict_tp = [metrics["strict"][cat]["tp"] for cat in categories]
    strict_fp = [metrics["strict"][cat]["fp"] for cat in categories]
    strict_fn = [metrics["strict"][cat]["fn"] for cat in categories]
    lenient_tp = [metrics["lenient"][cat]["tp"] for cat in categories]
    lenient_fp = [metrics["lenient"][cat]["fp"] for cat in categories]
    gt_positives = [tp + fn for tp, fn in zip(strict_tp, strict_fn)]

    x = list(range(len(categories)))
    width = 0.38
    strict_left = [i - width / 2 for i in x]
    lenient_left = [i + width / 2 for i in x]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Strict regime (solid)
    ax.bar(strict_left, strict_tp, width, color="#2ca02c", edgecolor="black", linewidth=0.4)
    ax.bar(strict_left, strict_fp, width, bottom=strict_tp, color="#d62728", edgecolor="black", linewidth=0.4)

    # Lenient regime (hatched, lighter)
    ax.bar(lenient_left, lenient_tp, width, color="#2ca02c", alpha=0.55, hatch="//", edgecolor="white")
    ax.bar(lenient_left, lenient_fp, width, bottom=lenient_tp, color="#d62728", alpha=0.55, hatch="//", edgecolor="white")

    for i, gt_count in enumerate(gt_positives):
        ax.hlines(gt_count, i - width, i + width, colors="black", linestyles="--", linewidth=1.5)

    # Bar count annotations
    for i in x:
        strict_total = strict_tp[i] + strict_fp[i]
        lenient_total = lenient_tp[i] + lenient_fp[i]
        if strict_total > 0:
            ax.text(strict_left[i], strict_total + 0.15, f"S={strict_total}", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(strict_left[i], 0.15, "S=0", ha="center", va="bottom", fontsize=7, color="grey")
        if lenient_total > 0:
            ax.text(lenient_left[i], lenient_total + 0.15, f"L={lenient_total}", ha="center", va="bottom", fontsize=7)
        else:
            ax.text(lenient_left[i], 0.15, "L=0", ha="center", va="bottom", fontsize=7, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel("Keyframes flagged as positive by the VLM")
    ax.set_title(
        "VLM predictions per category — strict (solid) vs lenient (hatched). "
        "Dashed line = GT positive total."
    )

    legend_handles = [
        Patch(facecolor="#2ca02c", edgecolor="black", label="TP — VLM agrees with GT (positive)"),
        Patch(facecolor="#d62728", edgecolor="black", label="FP — VLM positive, GT negative"),
        Patch(facecolor="lightgrey", hatch="//", edgecolor="white", label="Lenient regime (possible + clear)"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="GT positive total (TP + FN)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_substrate_confusion(metrics: dict[str, Any], path: Path, plt) -> Path | None:
    """Heatmap of GT substrate vs VLM substrate."""

    confusion = metrics["substrate_confusion"]
    if not confusion:
        return None

    gt_labels = sorted(confusion)
    pred_labels = sorted({label for preds in confusion.values() for label in preds})
    matrix = [[confusion[gt].get(pred, 0) for pred in pred_labels] for gt in gt_labels]

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pred_labels)))
    ax.set_xticklabels(pred_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(gt_labels)))
    ax.set_yticklabels(gt_labels)
    ax.set_xlabel("VLM prediction")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Substrate confusion (accuracy {metrics['substrate_accuracy']:.2f})")

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value > 0:
                ax.text(j, i, str(value), ha="center", va="center", color="white" if value > max(map(max, matrix)) / 2 else "black", fontsize=10)

    fig.colorbar(image, ax=ax, label="keyframes")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_uncertainty_calibration(metrics: dict[str, Any], path: Path, plt) -> Path:
    """For each category, the GT-positive rate within each VLM status bucket."""

    categories = CATEGORIES
    statuses = ("none", "possible", "clear")
    x = range(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for offset, status in zip((-width, 0.0, width), statuses):
        rates = [metrics["uncertainty_calibration"][cat][status]["gt_positive_rate"] for cat in categories]
        counts = [metrics["uncertainty_calibration"][cat][status]["total"] for cat in categories]
        bars = ax.bar(
            [i + offset for i in x],
            rates,
            width,
            label=status,
            color=STATUS_COLORS[status],
        )
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"n={count}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel("GT-positive rate")
    ax.set_ylim(0, 1.1)
    ax.set_title("Uncertainty calibration: how often each VLM status is actually positive in GT")
    ax.legend(title="VLM status", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_timeline(gt: dict[str, Any], reports: list[dict[str, Any]], path: Path, plt) -> Path:
    """Timeline showing GT event ranges and VLM keyframe predictions across the video."""

    categories = CATEGORIES
    fig, ax = plt.subplots(figsize=(11, 4.2))

    gt_events_by_category: dict[str, list[tuple[float, float]]] = {cat: [] for cat in categories}
    for event in gt.get("events", []):
        category = GT_EVENT_TO_VLM_CATEGORY.get(event["label"])
        if category in gt_events_by_category:
            gt_events_by_category[category].append((float(event["start"]), float(event["end"])))

    for index, category in enumerate(categories):
        y = index
        for start, end in gt_events_by_category[category]:
            ax.add_patch(
                plt.Rectangle((start, y - 0.35), end - start, 0.7, color=GT_FILL_COLOR, alpha=0.25)
            )

        for report in reports:
            timestamp = report.get("timestamp_sec")
            if timestamp is None:
                continue
            status = report.get(f"{category}_status", "none")
            color = STATUS_COLORS.get(status, STATUS_COLORS["none"])
            ax.plot(float(timestamp), y, marker="o", color=color, markersize=6, markeredgecolor="black", markeredgewidth=0.4)

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.set_xlabel("Time (sec)")
    ax.set_title("Timeline: GT event ranges (blue) vs VLM keyframe predictions (dots)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=STATUS_COLORS["clear"], markeredgecolor="black", markersize=8, label="VLM clear"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=STATUS_COLORS["possible"], markeredgecolor="black", markersize=8, label="VLM possible"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=STATUS_COLORS["none"], markeredgecolor="black", markersize=8, label="VLM none"),
        plt.Rectangle((0, 0), 1, 1, color=GT_FILL_COLOR, alpha=0.25, label="GT positive"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        fontsize=9,
        frameon=False,
    )
    ax.set_ylim(-0.7, len(categories) - 0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
