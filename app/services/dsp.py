"""Audio pre-processing: load + hard-dB noise gate.

This stage runs before basic-pitch inference (see CLAUDE.md "DO NOT bypass the
audio pre-processing phase"). Ambient hum and breath noise that survives this
gate becomes phantom notes downstream.
"""

from __future__ import annotations

import io
from typing import Tuple

import librosa
import numpy as np

TARGET_SAMPLE_RATE: int = 22050
DEFAULT_GATE_DB: float = -55.0
GATE_FRAME_MS: float = 20.0


def load_audio(data: bytes, sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Decode a raw audio byte buffer to a mono float32 waveform at the target rate."""
    buf = io.BytesIO(data)
    waveform, native_sr = librosa.load(buf, sr=sr, mono=True)
    return waveform.astype(np.float32, copy=False), native_sr


def apply_noise_gate(
    waveform: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    threshold_db: float = DEFAULT_GATE_DB,
    frame_ms: float = GATE_FRAME_MS,
) -> np.ndarray:
    """Zero out frames whose RMS energy falls below `threshold_db` (relative to 0 dBFS)."""
    if waveform.size == 0:
        return waveform

    frame_length = max(1, int(sr * frame_ms / 1000.0))
    hop_length = frame_length

    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    # Convert to dBFS; guard against log(0).
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-10))

    gated = waveform.copy()
    for i, db in enumerate(rms_db):
        if db < threshold_db:
            start = i * hop_length
            end = min(start + frame_length, gated.size)
            gated[start:end] = 0.0
    return gated


def preprocess(data: bytes, threshold_db: float = DEFAULT_GATE_DB) -> Tuple[np.ndarray, int]:
    """One-shot helper: decode bytes and apply the noise gate."""
    waveform, sr = load_audio(data)
    return apply_noise_gate(waveform, sr=sr, threshold_db=threshold_db), sr
