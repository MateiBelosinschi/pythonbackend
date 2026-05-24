"""Noise gate must zero silent regions while preserving voiced segments."""

import numpy as np

from app.services.dsp import apply_noise_gate


def test_gate_zeros_silent_region(tone_with_silence):
    waveform, sr = tone_with_silence
    gated = apply_noise_gate(waveform, sr=sr, threshold_db=-40.0)

    # Silent middle region (samples 0.55s..0.95s) should be exactly zero.
    start = int(0.55 * sr)
    end = int(0.95 * sr)
    assert np.all(gated[start:end] == 0.0)


def test_gate_preserves_loud_tone(tone_with_silence):
    waveform, sr = tone_with_silence
    gated = apply_noise_gate(waveform, sr=sr, threshold_db=-40.0)

    # First tone (0.1s..0.4s) should retain meaningful energy.
    start = int(0.1 * sr)
    end = int(0.4 * sr)
    assert np.max(np.abs(gated[start:end])) > 0.1


def test_gate_handles_empty_input():
    empty = np.zeros(0, dtype=np.float32)
    assert apply_noise_gate(empty).size == 0
