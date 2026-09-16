"""Validation utilities for non-verbal time intervals."""

from __future__ import annotations

import math
from collections.abc import Sequence


Interval = tuple[float, float]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def normalize_intervals(intervals: object | None) -> list[Interval]:
    """Validate, sort, and merge touching NV intervals expressed in seconds."""

    if intervals is None:
        return []
    if isinstance(intervals, (str, bytes)):
        raise TypeError("NV intervals must be numeric pairs, not a string")
    values = list(intervals)  # type: ignore[arg-type]
    if len(values) == 2 and all(_is_number(value) for value in values):
        values = [values]

    parsed: list[Interval] = []
    for value in values:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
            raise TypeError("Each NV interval must be a two-item numeric sequence")
        start, end = value
        if not _is_number(start) or not _is_number(end):
            raise TypeError("NV interval boundaries must be numbers")
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError(f"Invalid NV interval: {(start, end)}")
        parsed.append((start, end))

    merged: list[Interval] = []
    for start, end in sorted(parsed):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def target_duration_seconds(samples: int, sample_rate: int, frame_rate: float) -> float:
    """Return a duration that preserves the final target-audio token frame."""

    if samples <= 0 or sample_rate <= 0 or frame_rate <= 0:
        raise ValueError("samples, sample_rate, and frame_rate must all be positive")
    target_frames = math.ceil((samples / sample_rate) * frame_rate)
    return (target_frames + 0.01) / frame_rate
