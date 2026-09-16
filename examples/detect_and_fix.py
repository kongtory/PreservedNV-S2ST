#!/usr/bin/env python3
"""Detect NV regions with PretrainedSED and preserve them during OmniVoice cloning."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from nv_fix.intervals import target_duration_seconds
from nv_fix.pretrained_sed import DEFAULT_EVENT_LABELS, detect_events, event_intervals, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", required=True, help="Patched OmniVoice model path or HF model ID")
    parser.add_argument("--pretrained-sed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/nv_fixed.wav"))
    parser.add_argument("--audit-json", type=Path, default=Path("outputs/audit.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sed-device", help="Defaults to --device")
    parser.add_argument("--language", help="Target language name or code")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--min-nv-duration", type=float, default=0.10)
    parser.add_argument("--segment-seconds", type=int, default=10)
    parser.add_argument("--event-label", action="append")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-event", choices=("generate", "error"), default="generate")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    sed_device = args.sed_device or args.device
    sed_model, class_names, torch = load_model(args.pretrained_sed_root, sed_device)
    events = detect_events(
        args.audio, model=sed_model, class_names=class_names, torch=torch, device=sed_device,
        threshold=args.threshold, min_duration=args.min_nv_duration,
        segment_seconds=args.segment_seconds, event_labels=args.event_label or DEFAULT_EVENT_LABELS,
    )
    intervals = event_intervals(events)
    print(f"SED selected {len(events)} event(s), merged into {len(intervals)} interval(s): {intervals}")
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps({
        "audio": str(args.audio), "threshold": args.threshold,
        "event_labels": args.event_label or sorted(DEFAULT_EVENT_LABELS),
        "sed_events": [event.as_dict() for event in events], "nv_positions_seconds": intervals,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        return
    if not intervals and args.no_event == "error":
        raise RuntimeError("No selected SED event")

    import numpy as np
    import soundfile as sf
    from omnivoice.models.omnivoice import OmniVoice

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{args.device} was requested for OmniVoice, but CUDA is unavailable")
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    model = OmniVoice.from_pretrained(args.model, device_map=args.device, dtype=dtype)
    info = sf.info(args.audio)
    duration = args.duration or target_duration_seconds(info.frames, info.samplerate, model.audio_tokenizer.config.frame_rate)
    common = {"text": args.text, "language": args.language, "duration": duration, "audio_chunk_threshold": 1_000_000.0}
    if intervals:
        prompt = model.create_nv_voice_clone_prompt(args.audio.as_posix(), args.reference_text, intervals)
        generated = model.generate(nv_voice_clone_prompt=prompt, **common)
    else:
        generated = model.generate(ref_audio=args.audio.as_posix(), ref_text=args.reference_text, **common)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, generated[0], model.sampling_rate)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
