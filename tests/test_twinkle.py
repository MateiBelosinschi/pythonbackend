"""End-to-end with Twinkle Twinkle — the canonical melody for amateur-friendly transcription."""

import numpy as np
import pytest

from app.services import dsp, monophonic, quantizer, transcriber

pytestmark = pytest.mark.slow

SR = 22050

# C4 C4 G4 G4 A4 A4 G4(half) — first phrase of Twinkle Twinkle.
# Real humming has breath gaps between syllables; we insert small silences so
# basic-pitch hears each note as a distinct articulation.
TWINKLE_PHRASE = [
    (261.63, 0.5),  # C4
    (261.63, 0.5),
    (392.00, 0.5),  # G4
    (392.00, 0.5),
    (440.00, 0.5),  # A4
    (440.00, 0.5),
    (392.00, 1.0),  # G4 (held)
]
INTER_NOTE_SILENCE_S = 0.06  # ~60 ms breath gap between notes


def _synth_phrase(notes_spec, sr: int = SR, amplitude: float = 0.3) -> np.ndarray:
    chunks = []
    gap = np.zeros(int(INTER_NOTE_SILENCE_S * sr), dtype=np.float32)
    for i, (freq, dur) in enumerate(notes_spec):
        n = int(round(dur * sr))
        t = np.arange(n, dtype=np.float32) / sr
        env = np.ones(n, dtype=np.float32)
        fade = max(1, int(sr * 0.01))
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        chunks.append((amplitude * env * np.sin(2 * np.pi * freq * t)).astype(np.float32))
        if i + 1 < len(notes_spec):
            chunks.append(gap)
    return np.concatenate(chunks)


def test_twinkle_phrase_recovered_as_monophonic_melody():
    waveform = _synth_phrase(TWINKLE_PHRASE)
    gated = dsp.apply_noise_gate(waveform, sr=SR)
    midi = transcriber.transcribe(gated, sr=SR)
    notes = monophonic.collapse_to_melody(quantizer.quantize(midi))

    pitches = [n.pitch for n in notes]

    # The melody has 7 articulated notes (CC GG AA G). With ~60ms breath gaps
    # basic-pitch should recover most onsets; allow some merging.
    assert len(notes) >= 6, f"Too few notes recovered: {pitches}"

    # Contour check: must visit C4, then G4, then A4, then G4 in order.
    contour_targets = [60, 67, 69, 67]
    idx = 0
    for p in pitches:
        if idx < len(contour_targets) and abs(p - contour_targets[idx]) <= 1:
            idx += 1
    assert idx == len(contour_targets), f"Twinkle contour not recovered in order: {pitches}"

    # Monophonic invariant: no two notes overlap in time.
    for i in range(len(notes) - 1):
        assert notes[i].end <= notes[i + 1].start + 1e-9, (
            f"Overlap between {notes[i]} and {notes[i + 1]}"
        )

    # All pitches in vocal range (the filter killed sub-bass octave errors).
    assert all(48 <= n.pitch <= 84 for n in notes), f"Out-of-range pitches: {pitches}"
