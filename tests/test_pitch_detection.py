"""Test d'int gration lent : sinusoide synth tique -> CREPE doit retrouver A4.

Marqu  ``slow`` : n cessite TensorFlow + le mod le CREPE. Lancer avec :
    pytest -m slow
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.config import settings

crepe = pytest.importorskip("crepe", reason="CREPE not installed")


pytestmark = pytest.mark.slow


def _make_sine(frequency: float, duration_sec: float, sr: int) -> np.ndarray:
    n = int(duration_sec * sr)
    t = np.arange(n, dtype=np.float32) / sr
    return (0.5 * np.sin(2.0 * math.pi * frequency * t)).astype(np.float32)


def test_crepe_detects_a4_on_sine():
    from app.services.pitch_detection import predict_pitch

    sr = settings.SAMPLE_RATE
    audio = _make_sine(440.0, duration_sec=1.0, sr=sr)
    _times, freqs, confs = predict_pitch(audio, sr)

    voiced = freqs[confs >= 0.5]
    assert voiced.size > 0
    median_hz = float(np.median(voiced))
    assert median_hz == pytest.approx(440.0, rel=0.02)
