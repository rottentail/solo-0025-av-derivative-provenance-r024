"""Boundary validation for HTTP payloads and operation parameters."""

from __future__ import annotations

import base64
import binascii
import math
import re
from pathlib import PurePath
from typing import Any

from .errors import PlanInvalid, ValidationFailure


MAX_MEDIA_BYTES = 24 * 1024 * 1024
MAX_HTTP_BODY_BYTES = 36 * 1024 * 1024
MAX_FILENAME_LENGTH = 140
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")


def require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationFailure(f"{name} must be a JSON object.")
    return value


def require_string(
    value: Any,
    name: str,
    *,
    minimum: int = 1,
    maximum: int = 256,
) -> str:
    if not isinstance(value, str):
        raise ValidationFailure(f"{name} must be a string.")
    normalized = value.strip()
    if len(normalized) < minimum:
        raise ValidationFailure(f"{name} must not be empty.")
    if len(normalized) > maximum:
        raise ValidationFailure(f"{name} exceeds its length limit.")
    return normalized


def validate_filename(value: Any) -> str:
    filename = require_string(
        value,
        "filename",
        maximum=MAX_FILENAME_LENGTH,
    )
    if CONTROL_CHARACTERS.search(filename):
        raise ValidationFailure("filename contains control characters.")
    if PurePath(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValidationFailure("filename must be a base name without a path.")
    if filename in {".", ".."}:
        raise ValidationFailure("filename is not a usable media name.")
    return filename


def decode_media_payload(value: Any) -> bytes:
    encoded = require_string(
        value,
        "data_base64",
        minimum=4,
        maximum=MAX_HTTP_BODY_BYTES,
    )
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValidationFailure("data_base64 is not strict base64.") from exc
    if not data:
        raise ValidationFailure("data_base64 decoded to an empty payload.")
    if len(data) > MAX_MEDIA_BYTES:
        raise ValidationFailure(
            "decoded media exceeds the payload budget.",
            details={
                "size_bytes": len(data),
                "max_bytes": MAX_MEDIA_BYTES,
            },
        )
    return data


def require_number(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanInvalid(f"{name} must be numeric.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise PlanInvalid(f"{name} must be finite.")
    if numeric < minimum or numeric > maximum:
        raise PlanInvalid(
            f"{name} is outside the allowed range.",
            details={"minimum": minimum, "maximum": maximum},
        )
    return numeric


def require_integer(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanInvalid(f"{name} must be an integer.")
    if value < minimum or value > maximum:
        raise PlanInvalid(
            f"{name} is outside the allowed range.",
            details={"minimum": minimum, "maximum": maximum},
        )
    return value


def require_boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PlanInvalid(f"{name} must be boolean.")
    return value


MAX_SEGMENT_FRAMES = 11_520_000
MAX_SEGMENTS_PER_RUN = 4096
EXECUTION_MODES = {"inline", "segmented"}


def _execution_integer(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationFailure(f"execution.{name} must be an integer.")
    if value < 1 or value > maximum:
        raise ValidationFailure(
            f"execution.{name} is outside the allowed range.",
            details={"minimum": 1, "maximum": maximum},
        )
    return value


def validate_execution(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    request = require_object(value, "execution")
    unknown = sorted(set(request) - {"mode", "segment_frames", "segments_per_run"})
    if unknown:
        raise ValidationFailure(
            "execution contains unsupported fields.",
            details={"fields": unknown},
        )
    mode = request.get("mode", "inline")
    if not isinstance(mode, str) or mode not in EXECUTION_MODES:
        raise ValidationFailure(
            "execution.mode must be one of: inline, segmented."
        )
    segment_frames = request.get("segment_frames")
    segments_per_run = request.get("segments_per_run")
    if mode == "inline":
        if segment_frames is not None or segments_per_run is not None:
            raise ValidationFailure(
                "execution tuning requires segmented mode."
            )
        return {"mode": "inline"}
    parsed: dict[str, Any] = {"mode": "segmented"}
    if segment_frames is not None:
        parsed["segment_frames"] = _execution_integer(
            segment_frames,
            "segment_frames",
            maximum=MAX_SEGMENT_FRAMES,
        )
    else:
        parsed["segment_frames"] = None
    if segments_per_run is not None:
        parsed["segments_per_run"] = _execution_integer(
            segments_per_run,
            "segments_per_run",
            maximum=MAX_SEGMENTS_PER_RUN,
        )
    else:
        parsed["segments_per_run"] = 1
    return parsed
