"""collapse_to_melody must produce a clean single-voice line."""

from app.models import Note
from app.services.monophonic import collapse_to_melody

SIXTEENTH = 60.0 / 120 / 4  # 0.125 s


def test_keeps_highest_velocity_per_onset_cell():
    notes = [
        Note(pitch=60, start=0.0, end=0.5, velocity=40),
        Note(pitch=72, start=0.0, end=0.5, velocity=90),  # octave-up ghost wins on velocity
        Note(pitch=48, start=0.0, end=0.5, velocity=60),
    ]
    out = collapse_to_melody(notes)
    assert len(out) == 1
    assert out[0].pitch == 72


def test_velocity_tie_prefers_lower_pitch():
    notes = [
        Note(pitch=72, start=0.0, end=0.5, velocity=80),
        Note(pitch=60, start=0.0, end=0.5, velocity=80),
    ]
    out = collapse_to_melody(notes)
    assert len(out) == 1
    assert out[0].pitch == 60


def test_distinct_onsets_all_kept():
    notes = [
        Note(pitch=60, start=0.0, end=0.5, velocity=80),
        Note(pitch=62, start=0.5, end=1.0, velocity=80),
        Note(pitch=64, start=1.0, end=1.5, velocity=80),
    ]
    out = collapse_to_melody(notes)
    assert [n.pitch for n in out] == [60, 62, 64]


def test_overhang_is_trimmed_to_next_onset():
    notes = [
        Note(pitch=60, start=0.0, end=1.0, velocity=80),   # overshoots
        Note(pitch=62, start=0.5, end=1.0, velocity=80),
    ]
    out = collapse_to_melody(notes)
    assert len(out) == 2
    assert out[0].end == 0.5
    assert out[1].start == 0.5


def test_empty_input_returns_empty():
    assert collapse_to_melody([]) == []


def test_output_is_sorted_by_start():
    notes = [
        Note(pitch=64, start=1.0, end=1.5, velocity=80),
        Note(pitch=60, start=0.0, end=0.5, velocity=80),
        Note(pitch=62, start=0.5, end=1.0, velocity=80),
    ]
    out = collapse_to_melody(notes)
    assert [n.start for n in out] == sorted(n.start for n in out)
