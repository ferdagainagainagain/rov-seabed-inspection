"""Small path-resolution helper shared by the CLI scripts."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: Path) -> Path:
    """Expand ``~`` and try the path both as-is and relative to the project root."""

    path = path.expanduser()
    if path.is_absolute() or path.exists():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    return path
