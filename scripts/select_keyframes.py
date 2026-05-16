#!/usr/bin/env python3
"""CLI for selecting representative ROV seabed video frames."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if TYPE_CHECKING:
    from rov_inspect.dino_features import DinoModel
    from rov_inspect.keyframes import FrameCandidate
    import pandas as pd


@dataclass
class FrameFilterStats:
    """Counters collected while streaming sampled frames."""

    sampled_frames: int = 0
    skipped_by_depth: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select visually meaningful keyframes from an ROV seabed video."
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")
    parser.add_argument(
        "--output",
        default=Path("outputs/keyframes"),
        type=Path,
        help="Output root directory.",
    )
    parser.add_argument(
        "--sample-every-sec",
        default=1.0,
        type=float,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "--novelty-threshold",
        default=0.30,
        type=float,
        help="Cosine distance threshold for visual-change selection.",
    )
    parser.add_argument(
        "--min-gap-sec",
        default=5.0,
        type=float,
        help="Minimum seconds between selected keyframes.",
    )
    parser.add_argument(
        "--max-gap-sec",
        default=45.0,
        type=float,
        help="Fallback seconds between selected keyframes.",
    )
    parser.add_argument(
        "--depth-csv",
        default=None,
        type=Path,
        help="Optional depth telemetry CSV used to reject non-underwater frames.",
    )
    parser.add_argument(
        "--min-depth-m",
        default=1.0,
        type=float,
        help="Minimum interpolated depth in meters for a frame to be eligible.",
    )
    parser.add_argument(
        "--depth-stable-window-sec",
        default=2.0,
        type=float,
        help="Require depth to stay above the threshold for this many seconds.",
    )
    parser.add_argument(
        "--allow-missing-depth",
        action="store_true",
        help="Keep frames with missing depth telemetry instead of skipping them.",
    )
    parser.add_argument(
        "--depth-filter-mode",
        default="all",
        choices=["all", "boundary"],
        help="Apply depth filtering to all frames or only near video start/end.",
    )
    parser.add_argument(
        "--depth-boundary-sec",
        default=60.0,
        type=float,
        help="Start/end window size used when --depth-filter-mode boundary.",
    )
    parser.add_argument(
        "--descriptor-backend",
        default="classical",
        choices=["classical", "dino", "hybrid"],
        help="Descriptor backend used for novelty distance.",
    )
    parser.add_argument(
        "--hybrid-dino-weight",
        default=0.7,
        type=float,
        help="DINO distance weight for --descriptor-backend hybrid.",
    )
    parser.add_argument(
        "--adaptive-threshold",
        action="store_true",
        help="Set novelty threshold from median + k * MAD of consecutive distances.",
    )
    parser.add_argument(
        "--adaptive-k",
        default=2.0,
        type=float,
        help="MAD multiplier used with --adaptive-threshold.",
    )
    parser.add_argument(
        "--dino-model",
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="Hugging Face model name for DINO-style embeddings.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Device for DINO embeddings.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    from rov_inspect.contact_sheet import make_contact_sheet
    from rov_inspect.descriptors import adaptive_threshold
    from rov_inspect.keyframes import consecutive_distances, save_keyframes, select_keyframes
    from rov_inspect.telemetry import load_depth_log
    from rov_inspect.video_io import read_video_info

    try:
        video_path = _resolve_input_path(args.video)
        output_root = args.output
        depth_df = None
        depth_csv = None
        if args.depth_csv is not None:
            depth_csv = _resolve_input_path(args.depth_csv)
            depth_df = load_depth_log(depth_csv)

        info = read_video_info(video_path)
        output_dir = output_root / video_path.stem
        if depth_df is not None and args.depth_filter_mode == "boundary" and info.duration_sec is None:
            raise RuntimeError("Boundary depth filtering requires a known video duration")
        if args.depth_boundary_sec < 0:
            raise ValueError("--depth-boundary-sec must be non-negative")
        if not 0.0 <= args.hybrid_dino_weight <= 1.0:
            raise ValueError("--hybrid-dino-weight must be between 0 and 1")

        dino_model = None
        if args.descriptor_backend in {"dino", "hybrid"}:
            from rov_inspect.dino_features import load_dino_model

            print(f"Loading DINO model: {args.dino_model}")
            dino_model = load_dino_model(args.dino_model, args.device)
            print(f"DINO device: {dino_model.device}")

        filter_stats = FrameFilterStats()
        candidates = _candidate_iterator(
            video_path=video_path,
            sample_every_sec=args.sample_every_sec,
            depth_df=depth_df,
            min_depth_m=args.min_depth_m,
            stable_window_sec=args.depth_stable_window_sec,
            allow_missing_depth=args.allow_missing_depth,
            depth_filter_mode=args.depth_filter_mode,
            depth_boundary_sec=args.depth_boundary_sec,
            video_duration_sec=info.duration_sec,
            descriptor_backend=args.descriptor_backend,
            dino_model=dino_model,
            dino_cache_dir=output_dir / "embeddings_cache",
            stats=filter_stats,
        )
        novelty_threshold = args.novelty_threshold
        if args.adaptive_threshold:
            candidate_list = list(candidates)
            distances = consecutive_distances(
                candidate_list,
                descriptor_backend=args.descriptor_backend,
                hybrid_dino_weight=args.hybrid_dino_weight,
            )
            novelty_threshold = adaptive_threshold(
                distances,
                k=args.adaptive_k,
                fallback=args.novelty_threshold,
            )
            candidates = iter(candidate_list)

        result = select_keyframes(
            candidates,
            novelty_threshold=novelty_threshold,
            min_gap_sec=args.min_gap_sec,
            max_gap_sec=args.max_gap_sec,
            descriptor_backend=args.descriptor_backend,
            hybrid_dino_weight=args.hybrid_dino_weight,
            adaptive_threshold_enabled=args.adaptive_threshold,
        )
        if filter_stats.sampled_frames == 0:
            raise RuntimeError(f"No frames could be sampled from video: {video_path}")
        if result.sampled_count == 0:
            raise RuntimeError(
                "No sampled frames passed the depth filter. "
                "Lower --min-depth-m, use --allow-missing-depth, or check --depth-csv alignment."
            )

        metadata, _ = save_keyframes(result.selected, output_dir)
        csv_path = output_dir / "keyframes.csv"
        metadata.to_csv(csv_path, index=False)

        contact_sheet_path = output_dir / "contact_sheet.jpg"
        labels = [
            f"{index:04d}  t={frame.timestamp_sec:.1f}s  {frame.reason}"
            for index, frame in enumerate(result.selected, start=1)
        ]
        make_contact_sheet([frame.image for frame in result.selected], labels, contact_sheet_path)

        print(f"Video: {video_path}")
        if info.duration_sec is None:
            print("Duration: unavailable")
        else:
            print(f"Duration: {info.duration_sec:.1f} sec")
        if depth_csv is not None:
            print(f"Depth CSV: {depth_csv}")
            print(
                "Depth columns: "
                f"time={depth_df.attrs.get('time_column')}, "
                f"depth={depth_df.attrs.get('depth_column')}"
            )
            print(f"Depth filter mode: {args.depth_filter_mode}")
            if args.depth_filter_mode == "boundary":
                print(f"Depth boundary window: {args.depth_boundary_sec:.1f} sec")
            print("Depth matching: linear interpolation between telemetry samples")
        print(f"Sampled frames: {filter_stats.sampled_frames}")
        print(f"Skipped by depth: {filter_stats.skipped_by_depth}")
        print(f"Skipped by quality: {result.skipped_by_quality}")
        print(f"Descriptor backend: {args.descriptor_backend}")
        print(f"Novelty threshold used: {novelty_threshold:.6f}")
        print(f"Adaptive threshold: {args.adaptive_threshold}")
        print(f"Selected keyframes: {len(result.selected)}")
        print(f"Output directory: {output_dir}")
        print(f"CSV path: {csv_path}")
        print(f"Contact sheet path: {contact_sheet_path}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def _candidate_iterator(
    video_path: Path,
    sample_every_sec: float,
    depth_df: "pd.DataFrame | None",
    min_depth_m: float,
    stable_window_sec: float,
    allow_missing_depth: bool,
    depth_filter_mode: str,
    depth_boundary_sec: float,
    video_duration_sec: float | None,
    descriptor_backend: str,
    dino_model: "DinoModel | None",
    dino_cache_dir: Path,
    stats: FrameFilterStats,
) -> Iterator["FrameCandidate"]:
    from rov_inspect.keyframes import make_candidate
    from rov_inspect.telemetry import depth_at_timestamp, depth_eligibility
    from rov_inspect.video_io import iter_sampled_frames

    sampled_frames = iter_sampled_frames(video_path, sample_every_sec)
    if tqdm is not None:
        sampled_frames = tqdm(sampled_frames, desc="Sampling", unit="frame")

    for sampled_frame in sampled_frames:
        stats.sampled_frames += 1
        depth_m = None
        depth_eligible = None
        telemetry_reason = "not_used"

        if depth_df is not None:
            should_apply_depth_filter = _should_apply_depth_filter(
                timestamp_sec=sampled_frame.timestamp_sec,
                mode=depth_filter_mode,
                boundary_sec=depth_boundary_sec,
                video_duration_sec=video_duration_sec,
            )
            if should_apply_depth_filter:
                depth_eligible, depth_m, telemetry_reason = depth_eligibility(
                    timestamp_sec=sampled_frame.timestamp_sec,
                    df=depth_df,
                    min_depth_m=min_depth_m,
                    stable_window_sec=stable_window_sec,
                    allow_missing_depth=allow_missing_depth,
                )
            else:
                depth_m = depth_at_timestamp(depth_df, sampled_frame.timestamp_sec)
                depth_eligible = True
                telemetry_reason = "outside_depth_boundary"

            if not depth_eligible:
                stats.skipped_by_depth += 1
                continue

        dino_descriptor = None
        if descriptor_backend in {"dino", "hybrid"}:
            if dino_model is None:
                raise RuntimeError("DINO model was not loaded")
            dino_descriptor = _dino_embedding_with_cache(
                sampled_frame.image,
                frame_index=sampled_frame.frame_index,
                timestamp_sec=sampled_frame.timestamp_sec,
                dino_model=dino_model,
                cache_dir=dino_cache_dir,
            )

        yield make_candidate(
            sampled_frame,
            depth_m=depth_m,
            depth_eligible=depth_eligible,
            telemetry_reason=telemetry_reason,
            dino_descriptor=dino_descriptor,
        )


def _should_apply_depth_filter(
    timestamp_sec: float,
    mode: str,
    boundary_sec: float,
    video_duration_sec: float | None,
) -> bool:
    """Return True when depth should be allowed to reject this frame."""

    if mode == "all":
        return True
    if mode != "boundary":
        raise ValueError(f"Unknown depth filter mode: {mode}")
    if video_duration_sec is None:
        raise RuntimeError("Boundary depth filtering requires a known video duration")

    return timestamp_sec <= boundary_sec or timestamp_sec >= video_duration_sec - boundary_sec


def _resolve_input_path(path: Path) -> Path:
    """Resolve relative paths from the current directory or project root."""

    path = path.expanduser()
    if path.is_absolute() or path.exists():
        return path

    project_relative_path = PROJECT_ROOT / path
    if project_relative_path.exists():
        return project_relative_path
    return path


def _dino_embedding_with_cache(
    frame,
    frame_index: int,
    timestamp_sec: float,
    dino_model: "DinoModel",
    cache_dir: Path,
):
    """Load or compute a DINO embedding for one frame."""

    import numpy as np

    from rov_inspect.dino_features import compute_dino_embedding

    cache_dir.mkdir(parents=True, exist_ok=True)
    model_key = _safe_cache_key(dino_model.model_name)
    cache_path = cache_dir / f"{model_key}_frame_{frame_index:08d}_t{timestamp_sec:.3f}.npy"
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)

    embedding = compute_dino_embedding(frame, dino_model)
    np.save(cache_path, embedding)
    return embedding


def _safe_cache_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


if __name__ == "__main__":
    raise SystemExit(main())
