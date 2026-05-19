"""Per-frame report schema and normalization shared by Stage 2 and Stage 3."""

from __future__ import annotations

from typing import Any

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
ENVIRONMENTAL_STATUS_FIELDS = STATUS_FIELDS[:4]
BOOLEAN_PRESENCE_FIELDS = [field.replace("_status", "_present") for field in STATUS_FIELDS]


def as_enum(value: Any, allowed: set[str], default: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in allowed:
            return normalized
    return default


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def as_status(value: Any, legacy_boolean: Any = None) -> str:
    """Map either a status string or a legacy boolean field to a status value."""

    if isinstance(value, bool):
        return "clear" if value else "none"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in STATUS_VALUES:
            return normalized
        if normalized in {"false", "no", "0", "absent"}:
            return "none"
        if normalized:
            return "possible"
    if as_bool(legacy_boolean):
        return "possible"
    return "none"


def as_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_annotation(parsed: dict[str, Any]) -> dict[str, Any]:
    """Validate annotation fields and fill safe defaults."""

    normalized: dict[str, Any] = {
        "substrate": as_enum(parsed.get("substrate"), SUBSTRATES, "unclear"),
        "rocks_present": as_bool(parsed.get("rocks_present")),
        "cobbles_present": as_bool(parsed.get("cobbles_present")),
    }
    for status_field in STATUS_FIELDS:
        legacy_boolean_field = status_field.replace("_status", "_present")
        normalized[status_field] = as_status(
            parsed.get(status_field),
            parsed.get(legacy_boolean_field),
        )
    for status_field, present_field in zip(STATUS_FIELDS, BOOLEAN_PRESENCE_FIELDS):
        normalized[present_field] = normalized[status_field] != "none"

    normalized["rov_equipment_type"] = as_enum(
        parsed.get("rov_equipment_type"), ROV_EQUIPMENT_TYPES, "none"
    )
    if normalized["rov_equipment_status"] == "none":
        normalized["rov_equipment_type"] = "none"

    normalized["water_visibility"] = as_enum(parsed.get("water_visibility"), VISIBILITIES, "unclear")
    normalized["inspection_importance"] = as_enum(
        parsed.get("inspection_importance"), IMPORTANCE_LEVELS, "medium"
    )
    normalized["uncertainty"] = as_enum(parsed.get("uncertainty"), UNCERTAINTY_LEVELS, "high")
    normalized["short_description"] = str(parsed.get("short_description") or "")
    return normalized


def normalize_report(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a full frame report (annotation + metadata)."""

    from pathlib import Path

    image_path = str(record.get("image_path", ""))
    return {
        "image_path": image_path,
        "image_name": str(record.get("image_name") or Path(image_path).name),
        "timestamp_sec": as_float_or_none(record.get("timestamp_sec")),
        **normalize_annotation(record),
        "model_name": str(record.get("model_name", "")),
        "raw_model_output": str(record.get("raw_model_output", "")),
    }
