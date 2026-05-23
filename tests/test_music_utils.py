"""Tests unitaires pour app.utils.music."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.utils.music import (
    decompose_sixteenths,
    duration_to_sixteenths,
    hz_array_to_midi,
    hz_to_midi,
    midi_to_pitch_name,
    pitch_name_to_midi,
)


class TestHzToMidi:
    def test_a4_440hz(self):
        assert hz_to_midi(440.0) == pytest.approx(69.0)

    def test_c4_middle_c(self):
        # C4 = MIDI 60 = ~261.63 Hz
        assert hz_to_midi(261.6256) == pytest.approx(60.0, abs=1e-3)

    def test_octave_doubles_frequency(self):
        assert hz_to_midi(880.0) == pytest.approx(81.0)

    def test_zero_returns_nan(self):
        assert math.isnan(hz_to_midi(0.0))

    def test_negative_returns_nan(self):
        assert math.isnan(hz_to_midi(-10.0))

    def test_vector_version(self):
        out = hz_array_to_midi(np.array([440.0, 0.0, 880.0, -1.0]))
        assert out[0] == pytest.approx(69.0)
        assert math.isnan(out[1])
        assert out[2] == pytest.approx(81.0)
        assert math.isnan(out[3])


class TestMidiToPitchName:
    @pytest.mark.parametrize(
        "midi,expected",
        [
            (60, "C4"),
            (61, "C#4"),
            (69, "A4"),
            (72, "C5"),
            (84, "C6"),
            (36, "C2"),
            (62, "D4"),
            (66, "F#4"),
            (70, "A#4"),
        ],
    )
    def test_known_values(self, midi, expected):
        assert midi_to_pitch_name(midi) == expected

    def test_always_uses_sharps(self):
        # MIDI 70 is A#4 (or Bb4). Backend must emit sharps.
        assert midi_to_pitch_name(70) == "A#4"
        assert "b" not in midi_to_pitch_name(63)


class TestPitchNameToMidi:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("C4", 60),
            ("C#4", 61),
            ("Db4", 61),  # accept flats on input
            ("A4", 69),
            ("Bb5", 82),
            ("A#5", 82),
            ("C6", 84),
        ],
    )
    def test_known_values(self, name, expected):
        assert pitch_name_to_midi(name) == expected

    def test_round_trip(self):
        for midi in range(36, 85):
            name = midi_to_pitch_name(midi)
            assert pitch_name_to_midi(name) == midi

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            pitch_name_to_midi("H4")
        with pytest.raises(ValueError):
            pitch_name_to_midi("C")


class TestDecomposeSixteenths:
    @pytest.mark.parametrize(
        "length,expected",
        [
            (0, []),
            (1, ["16"]),
            (2, ["8"]),
            (3, ["8", "16"]),
            (4, ["q"]),
            (5, ["q", "16"]),
            (6, ["q", "8"]),
            (7, ["q", "8", "16"]),
            (8, ["h"]),
            (12, ["h", "q"]),
            (16, ["w"]),
            (20, ["w", "q"]),
            (32, ["w", "w"]),
            (33, ["w", "w", "16"]),
        ],
    )
    def test_known_decompositions(self, length, expected):
        assert decompose_sixteenths(length) == expected

    def test_sum_matches_input(self):
        for length in range(0, 64):
            parts = decompose_sixteenths(length)
            assert sum(duration_to_sixteenths(p) for p in parts) == length

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            decompose_sixteenths(-1)


class TestDurationToSixteenths:
    def test_table(self):
        assert duration_to_sixteenths("w") == 16
        assert duration_to_sixteenths("h") == 8
        assert duration_to_sixteenths("q") == 4
        assert duration_to_sixteenths("8") == 2
        assert duration_to_sixteenths("16") == 1

    def test_invalid(self):
        with pytest.raises(ValueError):
            duration_to_sixteenths("32")
