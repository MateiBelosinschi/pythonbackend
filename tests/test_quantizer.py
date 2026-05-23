"""Tests pour app.services.quantizer.

On simule la sortie de CREPE avec des arrays synth tiques pour valider toute
la cha ne de quantification sans avoir besoin de TensorFlow.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pytest

from app.config import settings
from app.services.quantizer import (
    REST_CELL,
    REST_PITCH_NAME,
    QuantizeParams,
    cells_to_runs,
    quantize_cells,
    quantize_pipeline,
    quantize_to_notes,
    runs_to_notes,
    smooth_cell_pitches,
)
from app.utils.music import midi_to_pitch_name


CELL_SEC = settings.cell_seconds
STEP_SEC = settings.step_seconds
FRAMES_PER_CELL = int(round(CELL_SEC / STEP_SEC))  # ~12 frames par 16e


def _make_frames(cells_midi: List[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Construit times/freqs/conf/rms pour une suite de cellules donn e.

    Pour chaque entr e ``midi`` :
      - si == REST_CELL : frames silencieuses (conf=0, freq=0, rms=0)
      - sinon : frames vois es (freq correspondant au MIDI, conf=0.9, rms=0.1)
    """
    times = []
    freqs = []
    confs = []
    rms = []
    for cell_index, midi in enumerate(cells_midi):
        for f in range(FRAMES_PER_CELL):
            t = cell_index * CELL_SEC + (f + 0.5) * STEP_SEC
            times.append(t)
            if midi == REST_CELL:
                freqs.append(0.0)
                confs.append(0.0)
                rms.append(0.0)
            else:
                # MIDI -> Hz : 440 * 2^((m-69)/12)
                hz = 440.0 * (2.0 ** ((midi - 69) / 12.0))
                freqs.append(hz)
                confs.append(0.9)
                rms.append(0.1)
    duration_sec = len(cells_midi) * CELL_SEC
    return (
        np.array(times, dtype=np.float64),
        np.array(freqs, dtype=np.float64),
        np.array(confs, dtype=np.float64),
        np.array(rms, dtype=np.float32),
        duration_sec,
    )


class TestQuantizeCells:
    def test_single_held_note(self):
        # 4 cellules = 1 noire de C4 (MIDI 60)
        times, freqs, confs, rms, duration = _make_frames([60, 60, 60, 60])
        cells = quantize_cells(times, freqs, confs, rms, duration)
        assert cells.tolist() == [60, 60, 60, 60]

    def test_rest_cells(self):
        times, freqs, confs, rms, duration = _make_frames([REST_CELL, REST_CELL])
        cells = quantize_cells(times, freqs, confs, rms, duration)
        assert cells.tolist() == [REST_CELL, REST_CELL]

    def test_mixed(self):
        # noire C4, double-croche, croche A4, double-croche rest, ronde G4 (16)
        seq = [60, 60, 60, 60, 69, 69, REST_CELL, 67, 67, 67, 67]
        times, freqs, confs, rms, duration = _make_frames(seq)
        cells = quantize_cells(times, freqs, confs, rms, duration)
        assert cells.tolist() == seq

    def test_low_confidence_marks_rest(self):
        times, freqs, confs, rms, duration = _make_frames([60])
        confs[:] = 0.1  # en dessous du seuil 0.5
        cells = quantize_cells(times, freqs, confs, rms, duration)
        assert cells.tolist() == [REST_CELL]

    def test_sustain_with_low_ratio_but_enough_voiced_frames(self):
        """Tenue : ≥2 frames voisées suffisent même si ratio < VOICED_RATIO_THRESHOLD."""
        times, freqs, confs, rms, duration = _make_frames([60])
        # ~12 frames par cellule : on garde 2 voisées, le reste silencieux
        confs[:] = 0.0
        rms[:] = 0.0
        freqs[:] = 0.0
        confs[0] = 0.9
        confs[1] = 0.9
        rms[0] = 0.1
        rms[1] = 0.1
        hz = 440.0 * (2.0 ** ((60 - 69) / 12.0))
        freqs[0] = hz
        freqs[1] = hz
        cells = quantize_cells(times, freqs, confs, rms, duration)
        assert cells.tolist() == [60]

    def test_low_rms_marks_rest(self):
        times, freqs, confs, rms, duration = _make_frames([60])
        rms[:] = 0.001  # en dessous du seuil 0.005
        cells = quantize_cells(times, freqs, confs, rms, duration)
        assert cells.tolist() == [REST_CELL]

    def test_pitch_out_of_vocal_range_becomes_rest(self):
        # MIDI 30 (B0, hors plage [36, 84])
        # On utilise une fr quence correspondante directement.
        times, freqs, confs, rms, duration = _make_frames([60])
        # Override pour mettre une fr quence extr mement basse (MIDI 20)
        hz_low = 440.0 * (2.0 ** ((20 - 69) / 12.0))
        freqs[:] = hz_low
        cells = quantize_cells(times, freqs, confs, rms, duration)
        assert cells.tolist() == [REST_CELL]

    def test_empty_signal(self):
        cells = quantize_cells(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float32),
            duration_sec=0.0,
        )
        assert cells.size == 0


class TestCellsToRuns:
    def test_empty(self):
        assert cells_to_runs(np.array([], dtype=np.int64)) == []

    def test_all_same(self):
        runs = cells_to_runs(np.array([60, 60, 60, 60], dtype=np.int64))
        assert runs == [(60, 4)]

    def test_transitions(self):
        runs = cells_to_runs(np.array([60, 60, 62, 62, 62, REST_CELL, 64], dtype=np.int64))
        assert runs == [(60, 2), (62, 3), (REST_CELL, 1), (64, 1)]


class TestRunsToNotes:
    def test_single_quarter_note(self):
        # 4 sixteenths of C4 (MIDI 60) -> quarter note
        notes = runs_to_notes([(60, 4)])
        assert len(notes) == 1
        assert notes[0].pitch == "C4"
        assert notes[0].duration == "q"
        assert notes[0].isRest is False

    def test_long_note_decomposed(self):
        # 6 sixteenths of A4 -> q + 8 (m me pitch)
        notes = runs_to_notes([(69, 6)])
        assert [(n.pitch, n.duration, n.isRest) for n in notes] == [
            ("A4", "q", False),
            ("A4", "8", False),
        ]

    def test_rest_run(self):
        notes = runs_to_notes([(REST_CELL, 2)])
        assert len(notes) == 1
        assert notes[0].isRest is True
        assert notes[0].duration == "8"
        assert notes[0].pitch == REST_PITCH_NAME

    def test_whole_plus_quarter(self):
        notes = runs_to_notes([(60, 20)])  # > whole
        assert [n.duration for n in notes] == ["w", "q"]
        assert all(n.pitch == "C4" for n in notes)


class TestQuantizeToNotesEndToEnd:
    def test_two_quarters_separated_by_rest(self):
        # 4 cellules C4, 2 silences, 4 cellules E4 (MIDI 64)
        seq = [60] * 4 + [REST_CELL] * 2 + [64] * 4
        times, freqs, confs, rms, duration = _make_frames(seq)
        notes = quantize_to_notes(times, freqs, confs, rms, duration)
        assert [(n.pitch, n.duration, n.isRest) for n in notes] == [
            ("C4", "q", False),
            (REST_PITCH_NAME, "8", True),
            ("E4", "q", False),
        ]

    def test_whole_note(self):
        # 16 cellules de m me pitch -> ronde
        seq = [62] * 16  # D4
        times, freqs, confs, rms, duration = _make_frames(seq)
        notes = quantize_to_notes(times, freqs, confs, rms, duration)
        assert len(notes) == 1
        assert notes[0].pitch == "D4"
        assert notes[0].duration == "w"

    def test_pitch_names_use_sharps_only(self):
        # F#4 = MIDI 66
        seq = [66] * 4
        times, freqs, confs, rms, duration = _make_frames(seq)
        notes = quantize_to_notes(times, freqs, confs, rms, duration)
        assert notes[0].pitch == "F#4"
        assert "b" not in notes[0].pitch


def test_midi_to_pitch_name_used_for_runs():
    # Sanity check that pitch names match the helper exactly.
    seq = [70] * 4  # A#4
    times, freqs, confs, rms, duration = _make_frames(seq)
    notes = quantize_to_notes(times, freqs, confs, rms, duration)
    assert notes[0].pitch == midi_to_pitch_name(70)


class TestSmoothCellPitches:
    def test_unifies_minority_drift_on_long_sustain(self):
        cells = np.array([67] * 6 + [68] * 2, dtype=np.int64)
        smoothed = smooth_cell_pitches(cells)
        assert smoothed.tolist() == [67] * 8

    def test_preserves_real_interval_change(self):
        cells = np.array([60, 60, 67, 67], dtype=np.int64)
        smoothed = smooth_cell_pitches(cells)
        assert smoothed.tolist() == [60, 60, 67, 67]

    def test_preserves_equal_split_melodic_step(self):
        cells = np.array([65] * 4 + [64] * 4, dtype=np.int64)
        smoothed = smooth_cell_pitches(cells)
        assert smoothed.tolist() == [65, 65, 65, 65, 64, 64, 64, 64]


class TestGridOffset:
    def test_offset_shifts_cell_assignment(self):
        times, freqs, confs, rms, duration = _make_frames([REST_CELL, 60])
        cells_no_offset = quantize_cells(times, freqs, confs, rms, duration)
        assert cells_no_offset[1] == 60

        params = QuantizeParams(grid_offset_sec=CELL_SEC)
        cells_offset = quantize_cells(
            times, freqs, confs, rms, duration, params=params
        )
        assert cells_offset[0] == 60
        assert cells_offset[1] == REST_CELL


class TestDebugPipeline:
    def test_debug_includes_crepe_track_and_cells(self):
        times, freqs, confs, rms, duration = _make_frames([60, 60, 67, 67])
        result = quantize_pipeline(
            times, freqs, confs, rms, duration, debug=True
        )
        assert result.debug is not None
        assert len(result.debug.crepe_track) == times.size
        assert len(result.debug.cells) == 4
        assert result.debug.grid.bpm == settings.BPM
        assert result.debug.grid.offset_sec == 0.0


TWINKLE_CELLS: List[int] = (
    [60] * 4 + [60] * 4 + [67] * 4 + [67] * 4
    + [69] * 4 + [69] * 4 + [67] * 8
    + [65] * 4 + [65] * 4 + [64] * 4 + [64] * 4
    + [62] * 4 + [62] * 4 + [60] * 8
)


class TestTwinkleTwinkle:
    def test_produces_six_distinct_pitches(self):
        times, freqs, confs, rms, duration = _make_frames(TWINKLE_CELLS)
        notes = quantize_to_notes(times, freqs, confs, rms, duration)
        pitches = {n.pitch for n in notes if not n.isRest}
        assert pitches == {"C4", "D4", "E4", "F4", "G4", "A4"}

    def test_half_note_sustains_are_not_split(self):
        times, freqs, confs, rms, duration = _make_frames(TWINKLE_CELLS)
        cells = quantize_cells(times, freqs, confs, rms, duration)
        runs = cells_to_runs(cells)
        g4_half = next(
            (length for value, length in runs if value == 67 and length == 8),
            None,
        )
        c4_half = next(
            (length for value, length in runs if value == 60 and length == 8),
            None,
        )
        assert g4_half == 8
        assert c4_half == 8

    def test_sustain_survives_one_semitone_drift(self):
        seq = [67] * 8
        times, freqs, confs, rms, duration = _make_frames(seq)
        hz_g = 440.0 * (2.0 ** ((67 - 69) / 12.0))
        hz_gs = 440.0 * (2.0 ** ((68 - 69) / 12.0))
        mid = len(freqs) // 2
        freqs[mid:] = hz_gs
        freqs[:mid] = hz_g
        cells = quantize_cells(times, freqs, confs, rms, duration)
        # Dérive 50/50 : le lissage ne fusionne pas, mais le run reste continu.
        assert cells.tolist() == [67, 67, 67, 67, 68, 68, 68, 68]
        runs = cells_to_runs(cells)
        assert runs == [(67, 4), (68, 4)]

    def test_expected_merged_note_sequence(self):
        """Notes consécutives de même hauteur sont fusionnées en runs plus longs."""
        times, freqs, confs, rms, duration = _make_frames(TWINKLE_CELLS)
        notes = quantize_to_notes(times, freqs, confs, rms, duration)
        assert [(n.pitch, n.duration) for n in notes] == [
            ("C4", "h"), ("G4", "h"), ("A4", "h"), ("G4", "h"),
            ("F4", "h"), ("E4", "h"), ("D4", "h"), ("C4", "h"),
        ]
