# SPDX-License-Identifier: GPL-3.0-or-later
"""Pure-Python timing contract. All internal intervals are [start, end)."""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
from pathlib import Path

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_PROMPTS = 256
MAX_SECONDS = Decimal("1200")


def number(value, name, *, positive=True):
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a number, not {value!r}") from exc
    if not result.is_finite() or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'non-negative'} and finite")
    return result


def round_frame(value):
    return int(Decimal(str(value)).to_integral_value(rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class Prompt:
    text: str
    duration: Decimal
    start: Decimal


@dataclass(frozen=True)
class FramePrompt:
    text: str
    start: int
    end: int

    @property
    def count(self):
        return self.end - self.start


def _one(row, first, second, required=True):
    if first in row and second in row:
        raise ValueError(f"Use {first} OR {second}, not both")
    result = row.get(first, row.get(second))
    if result is None and required:
        raise ValueError(f"Missing {first}")
    return result


def parse_prompts(data):
    """Accept an object array, {prompts}, native meta, or single native prompt.

    Starts, when supplied, are seconds relative to this generated section,
    not scene-global seconds. Gaps and overlaps are deliberately rejected.
    """
    if isinstance(data, dict):
        if data.get("schema_version", 1) != 1:
            raise ValueError("Unsupported prompt schema_version; expected 1")
        formats = sum(("prompts" in data, "texts" in data, "text" in data))
        if formats != 1:
            raise ValueError("Choose exactly one prompt format: prompts, texts/durations, or text/duration")
        if "prompts" in data:
            rows = data["prompts"]
        elif "texts" in data:
            texts, durations = data["texts"], data.get("durations")
            if not isinstance(texts, list) or not isinstance(durations, list) or len(texts) != len(durations):
                raise ValueError("texts and durations must be equal-length arrays")
            rows = [{"text": t, "duration": d} for t, d in zip(texts, durations)]
        else:
            rows = [data]
    else:
        rows = data
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PROMPTS:
        raise ValueError(f"Provide an array with 1–{MAX_PROMPTS} prompts")
    result, cursor = [], Decimal(0)
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Prompt {i + 1} must be an object")
        text = row.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError(f"Prompt {i + 1}: text must contain 1–4000 characters")
        duration = number(_one(row, "duration", "duration_seconds"), f"Prompt {i + 1} duration")
        start = _one(row, "start", "start_seconds", required=False)
        if start is not None and abs(number(start, "start", positive=False) - cursor) > Decimal("0.000001"):
            raise ValueError(f"Prompt {i + 1} must start at {cursor}s: gaps/overlaps are not supported")
        result.append(Prompt(text.strip(), duration, cursor))
        cursor += duration
        if cursor > MAX_SECONDS:
            raise ValueError(f"Total duration exceeds the {MAX_SECONDS}s safety limit")
    return result


def load_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON file exceeds the 2 MiB limit")
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON value: {value}")
    return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)


def frame_prompts(prompts, fps, start_frame=0):
    """Round cumulative boundaries, avoiding per-prompt rounding drift."""
    rate = number(fps, "FPS")
    if not isinstance(start_frame, int) or isinstance(start_frame, bool):
        raise ValueError("start_frame must be an integer")
    result = []
    for p in prompts:
        a = start_frame + round_frame(p.start * rate)
        b = start_frame + round_frame((p.start + p.duration) * rate)
        if b <= a:
            raise ValueError(f"Prompt {p.text!r} is shorter than one frame at {fps} FPS")
        result.append(FramePrompt(p.text, a, b))
    return result


def export_prompts(prompts, native=False):
    if native:
        return {"texts": [p.text for p in prompts], "durations": [float(p.duration) for p in prompts]}
    return {"schema_version": 1, "prompts": [
        {"text": p.text, "start_seconds": float(p.start), "duration_seconds": float(p.duration)}
        for p in prompts
    ]}


def viser_bounds(prompts, fps):
    """Adapter for Kimodo Viser: final end is inclusive; others are shared boundaries."""
    frames = frame_prompts(prompts, fps)
    return [(p.text, p.start, p.end - (i == len(frames) - 1)) for i, p in enumerate(frames)]
