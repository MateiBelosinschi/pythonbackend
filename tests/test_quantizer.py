"""Quantizer must snap notes exactly onto the 120 BPM 16th-note grid."""

import pretty_midi
import pytest

from app.models import FIXED_BPM, GRID_SUBDIVISION
from app.services.quantizer import notes_to_midi, quantize
from app.models import Note

SIXTEENTH_SECONDS = 60.0 / FIXED_BPM / 4  # 0.125 s at 120 BPM


def _build_pm(notes_spec):
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(FIXED_BPM))
    inst = pretty_midi.Instrument(program=0)
    for pitch, start, end in notes_spec:
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=pitch, start=start, end=end))
    pm.instruments.append(inst)
    return pm


def _is_on_grid(t: float) -> bool:
    cells = t / SIXTEENTH_SECONDS
    return abs(cells - round(cells)) < 1e-6


def test_snaps_slightly_off_notes_to_nearest_cell():
    pm = _build_pm([(60, 0.12, 0.25), (62, 0.38, 0.50)])
    notes = quantize(pm)

    assert len(notes) == 2
    for n in notes:
        assert _is_on_grid(n.start), f"start {n.start} not on grid"
        assert _is_on_grid(n.end), f"end {n.end} not on grid"


def test_collapsed_note_promoted_to_single_cell():
    # Note shorter than a 16th — should get a one-cell duration, not vanish.
    pm = _build_pm([(60, 0.0, 0.01)])
    notes = quantize(pm)
    assert len(notes) == 1
    assert notes[0].end - notes[0].start == pytest.approx(SIXTEENTH_SECONDS, abs=1e-6)


def test_round_trip_midi_export_preserves_count():
    pm = _build_pm([(60, 0.0, 0.5), (64, 0.5, 1.0), (67, 1.0, 1.5)])
    notes = quantize(pm)

    midi_bytes = notes_to_midi(notes)
    assert len(midi_bytes) > 0

    # Decode round-trip and check note count.
    import io
    reparsed = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    assert sum(len(inst.notes) for inst in reparsed.instruments) == len(notes)


def test_bpm_and_subdivision_constants():
    assert FIXED_BPM == 120
    assert GRID_SUBDIVISION == 16
