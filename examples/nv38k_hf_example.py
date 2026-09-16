#!/usr/bin/env python3
"""Run one Hugging Face NonVerbalSpeech38k row through NV preservation.

The default row and Korean target sentence mirror the one-row setup in
OmniVoice's ``infer.ipynb``. Use the annotated boundary or detect one with SED.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
from pathlib import Path
from typing import Any

from nv_fix.intervals import normalize_intervals
from nv_fix.pretrained_sed import DEFAULT_EVENT_LABELS, detect_waveform, event_intervals, load_model as load_sed_model


DATASET_NAME = "nonverbalspeech/nonverbalspeech38k"
DEFAULT_INDEX = 611
DEFAULT_TEXT = "응, 왜 아직도 연락이 안 오지? 답장도 안 하고, 대체 뭐 하고 있는 거야?"
TAG_PATTERN = re.compile(r"\[[^]]+\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=DEFAULT_INDEX)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--language", default="ko")
    parser.add_argument("--interval-source", choices=("dataset", "sed"), default="dataset")
    parser.add_argument("--pretrained-sed-root", type=Path, help="Required for --interval-source sed")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--min-nv-duration", type=float, default=0.10)
    parser.add_argument("--segment-seconds", type=int, default=10)
    parser.add_argument("--event-label", action="append")
    parser.add_argument("--model", help="Required unless --dry-run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sed-device")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--no-event", choices=("generate", "error"), default="generate")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def strip_nv_tags(text: str) -> str:
    return TAG_PATTERN.sub("", text).strip()


def load_row(index: int) -> tuple[dict[str, Any], Any, int]:
    if index < 0:
        raise ValueError("--index must be non-negative")
    from datasets import Audio, load_dataset
    import numpy as np
    import soundfile as sf

    dataset = load_dataset(DATASET_NAME, split=f"train[{index}:{index + 1}]")
    if len(dataset) != 1:
        raise IndexError(f"Dataset has no train row at index {index}")
    row = dataset.cast_column("audio", Audio(decode=False))[0]
    raw = row["audio"]
    audio, sample_rate = sf.read(io.BytesIO(raw["bytes"]), always_2d=False) if raw.get("bytes") else sf.read(raw["path"], always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    return row, (audio.T if audio.ndim == 2 else audio), int(sample_rate)


def write_audit(path: Path, row: dict[str, Any], intervals, source: str, events, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "dataset": DATASET_NAME, "split": "train", "index": args.index,
        "caption": row["caption"], "reference_text": strip_nv_tags(str(row["caption"])),
        "source_language": row.get("language"), "target_language": args.language,
        "target_text": args.text, "label": row.get("label"),
        "dataset_nv_position_seconds": row.get("non_verbal_region"),
        "interval_source": source, "nv_positions_seconds": intervals,
        "sed_events": [event.as_dict() for event in events],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.interval_source == "sed" and args.pretrained_sed_root is None:
        raise ValueError("--pretrained-sed-root is required with --interval-source sed")
    if not args.dry_run and not args.model:
        raise ValueError("--model is required unless --dry-run is set")
    args.output = args.output or Path("outputs") / f"nv38k_{args.index}_nv_fixed.wav"
    args.audit_json = args.audit_json or Path("outputs") / f"nv38k_{args.index}_audit.json"
    row, reference_audio, sample_rate = load_row(args.index)
    events: list[Any] = []
    if args.interval_source == "dataset":
        intervals = normalize_intervals(row.get("non_verbal_region"))
    else:
        device = args.sed_device or args.device
        sed_model, classes, torch = load_sed_model(args.pretrained_sed_root, device)
        events = detect_waveform(
            reference_audio, sample_rate, model=sed_model, class_names=classes, torch=torch,
            device=device, threshold=args.threshold, min_duration=args.min_nv_duration,
            segment_seconds=args.segment_seconds, event_labels=args.event_label or DEFAULT_EVENT_LABELS,
        )
        intervals = event_intervals(events)
    print(f"NV38K train[{args.index}] label={row.get('label')!r}; source={args.interval_source}; intervals={intervals}")
    write_audit(args.audit_json, row, intervals, args.interval_source, events, args)
    if args.dry_run:
        return
    if not intervals and args.no_event == "error":
        raise RuntimeError("No selected NV interval")

    import numpy as np
    import soundfile as sf
    import torch
    from omnivoice.models.omnivoice import OmniVoice

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} was requested for OmniVoice, but CUDA is unavailable")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    model = OmniVoice.from_pretrained(args.model, device_map=args.device, dtype=torch.float16 if args.device.startswith("cuda") else torch.float32)
    common = {"text": args.text, "language": args.language, "duration": args.duration or float(row["duration"]), "audio_chunk_threshold": 1_000_000.0}
    reference_text = strip_nv_tags(str(row["caption"]))
    if intervals:
        prompt = model.create_nv_voice_clone_prompt((reference_audio, sample_rate), reference_text, intervals)
        generated = model.generate(nv_voice_clone_prompt=prompt, **common)
    else:
        generated = model.generate(ref_audio=(reference_audio, sample_rate), ref_text=reference_text, **common)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, generated[0], model.sampling_rate)
    print(f"Wrote {args.output}; audit: {args.audit_json}")


if __name__ == "__main__":
    main()
