"""Shared fixtures: synthetic vocal-like audio with mandatory silences.

Per CLAUDE.md, every DSP refactor must validate against synthetic audio
containing silent regions to prove the gate is killing ghost notes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pytest
import soundfile as sf

SR = 22050


@dataclass
class ToneSpec:
    """A single tone segment."""

    freq_hz: float
    duration_s: float


@dataclass
class SilenceSpec:
    duration_s: float


Segment = ToneSpec | SilenceSpec


def _render(segments: List[Segment], sr: int = SR, amplitude: float = 0.3) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for seg in segments:
        n = int(round(seg.duration_s * sr))
        if isinstance(seg, ToneSpec):
            t = np.arange(n, dtype=np.float32) / sr
            chunks.append((amplitude * np.sin(2 * np.pi * seg.freq_hz * t)).astype(np.float32))
        else:
            chunks.append(np.zeros(n, dtype=np.float32))
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def _to_wav_bytes(waveform: np.ndarray, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture
def sr() -> int:
    return SR


@pytest.fixture
def tone_with_silence() -> Tuple[np.ndarray, int]:
    """0.5s A4 tone + 1.0s silence + 0.5s A4 tone. Silence MUST stay silent."""
    waveform = _render([
        ToneSpec(440.0, 0.5),
        SilenceSpec(1.0),
        ToneSpec(440.0, 0.5),
    ])
    return waveform, SR


@pytest.fixture
def tone_with_silence_wav(tone_with_silence) -> bytes:
    waveform, _ = tone_with_silence
    return _to_wav_bytes(waveform)


@pytest.fixture
def render_segments():
    """Factory: build a custom synthetic waveform from a segment list."""

    def _build(segments: List[Segment], amplitude: float = 0.3) -> Tuple[np.ndarray, int]:
        return _render(segments, amplitude=amplitude), SR

    return _build
