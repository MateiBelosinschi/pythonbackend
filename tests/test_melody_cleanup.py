"""Octave normalization + diatonic snap unit tests."""

from app.models import Note
from app.services.melody_cleanup import (
    TREBLE_HIGH,
    TREBLE_LOW,
    _detect_major_key_root,
    _nearest_scale_pitch,
    cleanup,
    normalize_octave,
    snap_to_major_key,
    transpose_to_c_major,
)


def _n(pitch: int, start: float = 0.0, end: float = 0.5) -> Note:
    return Note(pitch=pitch, start=start, end=end, velocity=80)


def _in_treble(notes):
    return [TREBLE_LOW <= n.pitch <= TREBLE_HIGH for n in notes]


def test_normalize_lifts_subbass_melody_into_treble():
    # All notes far below the staff (~C2 area) — should lift into [57, 79].
    notes = [_n(36), _n(40), _n(43)]
    out = normalize_octave(notes)
    assert all(_in_treble(out)), [n.pitch for n in out]


def test_normalize_leaves_well_placed_melody_alone():
    notes = [_n(60), _n(64), _n(67)]
    out = normalize_octave(notes)
    assert [n.pitch for n in out] == [60, 64, 67]


def test_normalize_pulls_too_high_melody_down():
    # Notes far above the staff — should pull into treble range.
    notes = [_n(96), _n(100), _n(103)]
    out = normalize_octave(notes)
    assert all(_in_treble(out)), [n.pitch for n in out]


def test_detect_c_major_key_root():
    # Pure C major pitches (no accidentals).
    notes = [_n(p) for p in [60, 62, 64, 65, 67, 69, 71]]
    assert _detect_major_key_root(notes) == 0  # C = pitch class 0


def test_detect_g_major_key_root():
    # G major: G A B C D E F# (7, 9, 11, 0, 2, 4, 6)
    notes = [_n(p) for p in [67, 69, 71, 60, 62, 64, 66]]
    assert _detect_major_key_root(notes) == 7  # G = pitch class 7


def test_nearest_scale_snaps_accidental_into_c_major():
    # C major root = 0. C#/Db (pitch class 1) is one semitone from C (0) and D (2).
    assert _nearest_scale_pitch(61, root=0) in (60, 62)  # snaps either down or up


def test_snap_eliminates_stray_accidentals():
    # A clean C-major run with one C# wobble — wobble should snap to C or D.
    notes = [_n(p) for p in [60, 61, 64, 65, 67]]
    out = snap_to_major_key(notes)
    # All output pitches must belong to the detected major scale.
    root = _detect_major_key_root(notes)
    scale_classes = {(root + step) % 12 for step in (0, 2, 4, 5, 7, 9, 11)}
    assert all((n.pitch % 12) in scale_classes for n in out)


def test_snap_preserves_timing_and_velocity():
    notes = [Note(pitch=61, start=0.5, end=1.0, velocity=90)]
    out = snap_to_major_key(notes)
    assert out[0].start == 0.5
    assert out[0].end == 1.0
    assert out[0].velocity == 90


def test_empty_inputs_pass_through():
    assert normalize_octave([]) == []
    assert snap_to_major_key([]) == []
    assert cleanup([]) == []


def test_transpose_g_major_to_c_major():
    # G major scale pitches — root = 7. +5 is the smaller shift (vs -7), so
    # G(67) lands on C5(72) and the rest of the scale follows.
    g_major = [_n(p) for p in [67, 69, 71, 72, 74, 76, 78]]
    out = transpose_to_c_major(g_major, root=7)
    expected = [72, 74, 76, 77, 79, 81, 83]
    assert [n.pitch for n in out] == expected


def test_transpose_f_major_picks_smaller_shift():
    # F major root = 5. -5 is smaller than +7, so we shift down.
    f_major = [_n(p) for p in [65, 67, 69]]
    out = transpose_to_c_major(f_major, root=5)
    assert [n.pitch for n in out] == [60, 62, 64]


def test_smooth_octave_jumps_pulls_alternating_outliers():
    from app.services.melody_cleanup import smooth_octave_jumps

    # C4 C5 C4 C5 C4 — alternating octave-up artifact. Should collapse to all C4.
    notes = [_n(60), _n(72), _n(60), _n(72), _n(60)]
    out = smooth_octave_jumps(notes)
    assert [n.pitch for n in out] == [60, 60, 60, 60, 60]


def test_smooth_octave_jumps_leaves_real_octave_leaps_alone():
    from app.services.melody_cleanup import smooth_octave_jumps

    # Real octave leap C4 -> C5 sustained -> back down. Middle note is
    # supported by being part of a sustained jump, not an alternation.
    notes = [_n(60), _n(72), _n(72), _n(72), _n(60)]
    out = smooth_octave_jumps(notes)
    assert [n.pitch for n in out] == [60, 72, 72, 72, 60]


def test_dedup_merges_consecutive_same_pitch_within_gap():
    from app.services.melody_cleanup import dedup_consecutive_repeats

    # Three C4 hits in close succession — should collapse to one held note.
    notes = [
        Note(pitch=60, start=0.0, end=0.25, velocity=80),
        Note(pitch=60, start=0.30, end=0.55, velocity=70),
        Note(pitch=60, start=0.60, end=0.90, velocity=85),
    ]
    out = dedup_consecutive_repeats(notes, max_gap=0.15)
    assert len(out) == 1
    assert out[0].pitch == 60
    assert out[0].start == 0.0
    assert out[0].end == 0.90
    # Velocity is the max of the merged notes (basic-pitch's confidence proxy).
    assert out[0].velocity == 85


def test_dedup_keeps_same_pitch_with_long_gap():
    from app.services.melody_cleanup import dedup_consecutive_repeats

    # Two C4 hits with a clear 1-second gap — these are two distinct articulations.
    notes = [
        Note(pitch=60, start=0.0, end=0.25, velocity=80),
        Note(pitch=60, start=1.50, end=1.75, velocity=80),
    ]
    out = dedup_consecutive_repeats(notes, max_gap=0.15)
    assert len(out) == 2


def test_dedup_keeps_different_pitches():
    from app.services.melody_cleanup import dedup_consecutive_repeats

    notes = [
        Note(pitch=60, start=0.0, end=0.25, velocity=80),
        Note(pitch=62, start=0.30, end=0.55, velocity=80),
        Note(pitch=64, start=0.60, end=0.85, velocity=80),
    ]
    out = dedup_consecutive_repeats(notes, max_gap=0.15)
    assert [n.pitch for n in out] == [60, 62, 64]


def test_smooth_octave_jumps_handles_short_inputs():
    from app.services.melody_cleanup import smooth_octave_jumps

    assert smooth_octave_jumps([]) == []
    assert [n.pitch for n in smooth_octave_jumps([_n(60)])] == [60]
    assert [n.pitch for n in smooth_octave_jumps([_n(60), _n(72)])] == [60, 72]


def test_cleanup_produces_only_c_major_white_keys_in_treble():
    # Simulate basic-pitch output: low octave, plus stray chromatic notes.
    raw = [_n(p) for p in [43, 45, 46, 48, 50, 52]]  # G2 A2 A#2 C3 D3 E3 etc.
    out = cleanup(raw)

    # All output pitches must be on white keys (C major scale classes).
    white_keys = {0, 2, 4, 5, 7, 9, 11}
    assert all(n.pitch % 12 in white_keys for n in out), [n.pitch for n in out]
    # And inside the treble clef range.
    assert all(TREBLE_LOW <= n.pitch <= TREBLE_HIGH for n in out), [n.pitch for n in out]
