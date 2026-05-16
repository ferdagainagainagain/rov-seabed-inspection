"""Small OpenCV helpers for reading sampled video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    """Basic metadata reported by OpenCV."""

    path: Path
    frame_count: int
    fps: float
    duration_sec: float | None
    width: int
    height: int


@dataclass(frozen=True)
class SampledFrame:
    """A frame sampled from a video at a known timestamp."""

    frame_index: int
    timestamp_sec: float
    image: np.ndarray


def open_video(video_path: Path) -> cv2.VideoCapture:
    """Open a video file and raise helpful errors on failure."""

    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video path is not a file: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")
    return capture


def read_video_info(video_path: Path) -> VideoInfo:
    """Read video metadata available from OpenCV."""

    capture = open_video(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    duration_sec = None
    if fps > 0 and frame_count > 0:
        duration_sec = frame_count / fps

    return VideoInfo(
        path=video_path,
        frame_count=frame_count,
        fps=fps,
        duration_sec=duration_sec,
        width=width,
        height=height,
    )


def iter_sampled_frames(video_path: Path, sample_every_sec: float) -> Iterator[SampledFrame]:
    """Yield one frame every ``sample_every_sec`` seconds."""

    if sample_every_sec <= 0:
        raise ValueError("--sample-every-sec must be greater than 0")

    capture = open_video(video_path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            raise ValueError(f"OpenCV did not report a valid FPS for: {video_path}")

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_step = max(1, int(round(sample_every_sec * fps)))

        if frame_count > 0:
            for frame_index in range(0, frame_count, frame_step):
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                yield SampledFrame(
                    frame_index=frame_index,
                    timestamp_sec=frame_index / fps,
                    image=frame,
                )
        else:
            next_timestamp = 0.0
            frame_index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_sec = frame_index / fps
                if timestamp_sec + 1e-9 >= next_timestamp:
                    yield SampledFrame(
                        frame_index=frame_index,
                        timestamp_sec=timestamp_sec,
                        image=frame,
                    )
                    next_timestamp += sample_every_sec
                frame_index += 1
    finally:
        capture.release()
