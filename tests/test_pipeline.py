"""End-to-end DSP → Basic-Pitch → Quantizer pipeline against synthetic audio.

This is the validation gate mandated by CLAUDE.md's Workflow section: every DSP
refactor must prove zero ghost notes are introduced during silent passages.

The basic-pitch model loads slowly (~10s cold start), so this whole module is
marked `slow` and skipped by default. Run explicitly with: pytest -m slow
"""

import pytest

from app.models import FIXED_BPM
from app.services import dsp, quantizer, transcriber
from tests.conftest import ToneSpec, SilenceSpec

pytestmark = pytest.mark.slow


def test_pipeline_produces_no_ghost_notes_in_silence(render_segments):
    """A long silent gap between two tones must yield zero notes inside that gap."""
    waveform, sr = render_segments([
        ToneSpec(440.0, 0.5),   # A4 for half a second
        SilenceSpec(2.0),       # 2 seconds of dead silence
        ToneSpec(440.0, 0.5),   # A4 again
    ])

    gated = dsp.apply_noise_gate(waveform, sr=sr)
    midi = transcriber.transcribe(gated, sr)
    notes = quantizer.quantize(midi)

    # Any note onset between 0.6s and 2.4s would be a ghost note inside silence.
    ghost = [n for n in notes if 0.6 < n.start < 2.4]
    assert ghost == [], f"Ghost notes detected in silent region: {ghost}"


def test_pipeline_recovers_tonal_content(render_segments):
    """At least one note should be detected for the voiced portion."""
    waveform, sr = render_segments([
        ToneSpec(440.0, 1.0),
    ])

    gated = dsp.apply_noise_gate(waveform, sr=sr)
    midi = transcriber.transcribe(gated, sr)
    notes = quantizer.quantize(midi)

    assert len(notes) >= 1
    # A4 is MIDI 69 — allow ±2 semitones since basic-pitch can octave-displace pure sines.
    assert any(67 <= n.pitch <= 71 for n in notes), f"No A4-ish note: {[n.pitch for n in notes]}"


def test_pipeline_emits_grid_aligned_notes(render_segments):
    waveform, sr = render_segments([ToneSpec(440.0, 1.0)])
    gated = dsp.apply_noise_gate(waveform, sr=sr)
    midi = transcriber.transcribe(gated, sr)
    notes = quantizer.quantize(midi)

    sixteenth = 60.0 / FIXED_BPM / 4
    for n in notes:
        cells_start = n.start / sixteenth
        cells_end = n.end / sixteenth
        assert abs(cells_start - round(cells_start)) < 1e-6
        assert abs(cells_end - round(cells_end)) < 1e-6
