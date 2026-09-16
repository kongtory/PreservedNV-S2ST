"""Optional adapter for the external PretrainedSED BEATs strong-event model."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .intervals import Interval, normalize_intervals


DEFAULT_EVENT_LABELS = frozenset(
    {
        "laughter", "baby laughter", "belly laugh", "chuckle, chortle", "giggle",
        "cough", "sneeze", "throat clearing", "gasp", "sigh", "sniff",
        "snoring", "hiccup", "yawn", "human voice", "human sounds",
    }
)


@dataclass(frozen=True)
class DetectedEvent:
    event_label: str
    onset: float
    offset: float
    score: float

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def load_model(pretrained_sed_root: Path, device: str) -> tuple[Any, list[str], Any]:
    """Load PretrainedSED's BEATs strong-event model from an external checkout."""

    if not pretrained_sed_root.is_dir():
        raise FileNotFoundError(f"PretrainedSED checkout not found: {pretrained_sed_root}")
    root = str(pretrained_sed_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    import torch
    from data_util import audioset_classes
    from models.beats.BEATs_wrapper import BEATsWrapper
    from models.prediction_wrapper import PredictionsWrapper

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{device} was requested for SED, but CUDA is unavailable")
    # Upstream resolves its checkpoint from the relative ``resources`` path.
    original_cwd = Path.cwd()
    try:
        os.chdir(root)
        model = PredictionsWrapper(BEATsWrapper(), checkpoint="BEATs_strong_1")
    finally:
        os.chdir(original_cwd)
    return model.eval().to(device), list(audioset_classes.as_strong_train_classes), torch


def detect_events(
    audio_path: Path,
    *,
    model: Any,
    class_names: list[str],
    torch: Any,
    device: str,
    threshold: float,
    min_duration: float,
    segment_seconds: int = 10,
    event_labels: Iterable[str] = DEFAULT_EVENT_LABELS,
) -> list[DetectedEvent]:
    """Load an audio file and return selected frame-level AudioSet events."""

    import librosa
    audio, sample_rate = librosa.load(audio_path, sr=None, mono=True)
    return detect_waveform(
        audio, sample_rate, model=model, class_names=class_names, torch=torch,
        device=device, threshold=threshold, min_duration=min_duration,
        segment_seconds=segment_seconds, event_labels=event_labels,
    )


def detect_waveform(
    audio: Any,
    sample_rate: int,
    *,
    model: Any,
    class_names: list[str],
    torch: Any,
    device: str,
    threshold: float,
    min_duration: float,
    segment_seconds: int = 10,
    event_labels: Iterable[str] = DEFAULT_EVENT_LABELS,
) -> list[DetectedEvent]:
    """Return selected AudioSet events for a mono/stereo in-memory waveform."""

    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    if min_duration < 0 or segment_seconds <= 0:
        raise ValueError("min_duration must be non-negative and segment_seconds positive")
    import librosa
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=0 if audio.shape[0] <= audio.shape[1] else 1)
    if audio.ndim != 1 or sample_rate <= 0:
        raise ValueError("SED waveform must be one- or two-dimensional with a positive sample rate")
    if sample_rate != 16_000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16_000)
        sample_rate = 16_000

    waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
    chunk_samples = segment_seconds * sample_rate
    logits = []
    for start in range(0, waveform.shape[1], chunk_samples):
        chunk = waveform[:, start : start + chunk_samples]
        if chunk.shape[1] < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.shape[1]))
        with torch.inference_mode():
            strong_logits, _ = model(model.mel_forward(chunk))
        logits.append(strong_logits)
    if not logits:
        return []

    probabilities = torch.sigmoid(torch.cat(logits, dim=2)).squeeze(0).cpu()
    frame_seconds = segment_seconds / logits[0].shape[-1]
    duration = len(audio) / sample_rate
    selected_labels = {str(label).casefold() for label in event_labels}
    events: list[DetectedEvent] = []
    for class_index, label in enumerate(class_names):
        if str(label).casefold() not in selected_labels:
            continue
        scores = probabilities[class_index]
        active = scores >= threshold
        changes = torch.diff(active.to(torch.int8), prepend=torch.zeros(1, dtype=torch.int8))
        starts = torch.where(changes == 1)[0].tolist()
        ends = torch.where(changes == -1)[0].tolist()
        if bool(active[-1]):
            ends.append(len(active))
        for start, end in zip(starts, ends):
            onset, offset = start * frame_seconds, min(end * frame_seconds, duration)
            if offset - onset >= min_duration:
                events.append(DetectedEvent(str(label), onset, offset, float(scores[start:end].max())))
    return sorted(events, key=lambda event: (event.onset, event.offset, event.event_label))


def event_intervals(events: Iterable[DetectedEvent]) -> list[Interval]:
    """Convert event detections to the disjoint intervals accepted by NV fixing."""

    return normalize_intervals([(event.onset, event.offset) for event in events])
