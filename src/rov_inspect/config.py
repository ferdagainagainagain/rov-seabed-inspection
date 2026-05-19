"""YAML config loading shared by the CLI scripts.

A config file is a YAML mapping with one section per pipeline stage::

    keyframes:
      video: data/...
      novelty_threshold: 0.25
    vlm:
      images_dir: outputs/keyframes_videoN/...
    synthesize:
      frame_reports: outputs/frame_reports/videoN/frame_reports.json

Each script loads only the section it cares about. CLI flags override the
values supplied by the YAML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_section(path: Path, section: str) -> dict[str, Any]:
    """Return the requested section from a YAML config file."""

    expanded = path.expanduser()
    if not expanded.is_file():
        raise FileNotFoundError(f"Config file does not exist: {expanded}")

    data = yaml.safe_load(expanded.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {expanded}")

    section_data = data.get(section, {}) or {}
    if not isinstance(section_data, dict):
        raise ValueError(f"Config section '{section}' must be a mapping: {expanded}")
    return section_data
