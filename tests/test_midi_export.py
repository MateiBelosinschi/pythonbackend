"""Tests pour app.services.midi_export."""
from __future__ import annotations

import io

import pytest
from mido import MidiFile, tempo2bpm

from app.api.schemas import MusicalNoteSchema
from app.config import settings
from app.services.midi_export import (
    DEFAULT_VELOCITY,
    TICKS_PER_BEAT,
    build_midi_file,
    notes_to_midi_bytes,
)


def _quarter_c4() -> MusicalNoteSchema:
    return MusicalNoteSchema(pitch="C4", duration="q", isRest=False)


def _half_a4() -> MusicalNoteSchema:
    return MusicalNoteSchema(pitch="A4", duration="h", isRest=False)


def _eighth_rest() -> MusicalNoteSchema:
    return MusicalNoteSchema(pitch="rest", duration="8", isRest=True)


class TestBuildMidiFile:
    def test_tempo_is_set_to_settings_bpm(self):
        midi = build_midi_file([_quarter_c4()])
        tempo_msgs = [m for track in midi.tracks for m in track if m.type == "set_tempo"]
        assert len(tempo_msgs) >= 1
        assert tempo2bpm(tempo_msgs[0].tempo) == pytest.approx(float(settings.BPM))

    def test_ticks_per_beat(self):
        midi = build_midi_file([_quarter_c4()])
        assert midi.ticks_per_beat == TICKS_PER_BEAT

    def test_quarter_note_event_lengths(self):
        midi = build_midi_file([_quarter_c4()])
        note_track = midi.tracks[1]
        note_on = [m for m in note_track if m.type == "note_on"]
        note_off = [m for m in note_track if m.type == "note_off"]
        assert len(note_on) == 1
        assert len(note_off) == 1

        assert note_on[0].note == 60  # C4
        assert note_on[0].velocity == DEFAULT_VELOCITY
        # quarter @ 120 BPM avec ticks_per_beat=480 -> 480 ticks
        assert note_off[0].time == TICKS_PER_BEAT

    def test_rest_offsets_next_note_on(self):
        notes = [_eighth_rest(), _quarter_c4()]
        midi = build_midi_file(notes)
        note_on = [m for m in midi.tracks[1] if m.type == "note_on"][0]
        # 8th = TICKS_PER_BEAT // 2 = 240 ticks de retard
        assert note_on.time == TICKS_PER_BEAT // 2

    def test_consecutive_notes_have_zero_gap(self):
        notes = [_quarter_c4(), _half_a4()]
        midi = build_midi_file(notes)
        track = midi.tracks[1]
        on_off = [m for m in track if m.type in ("note_on", "note_off")]
        # Pattern: note_on(C4, t=0), note_off(C4, t=480),
        #          note_on(A4, t=0), note_off(A4, t=960)
        assert on_off[0].time == 0
        assert on_off[1].time == TICKS_PER_BEAT  # quarter
        assert on_off[2].time == 0
        assert on_off[3].time == TICKS_PER_BEAT * 2  # half

    def test_invalid_pitch_raises(self):
        notes = [MusicalNoteSchema(pitch="not_a_note", duration="q", isRest=False)]
        with pytest.raises(ValueError):
            build_midi_file(notes)


class TestNotesToMidiBytes:
    def test_round_trip_via_mido(self):
        notes = [_quarter_c4(), _eighth_rest(), _half_a4()]
        data = notes_to_midi_bytes(notes)
        assert isinstance(data, bytes)
        assert len(data) > 0

        loaded = MidiFile(file=io.BytesIO(data))
        assert loaded.ticks_per_beat == TICKS_PER_BEAT
        note_ons = [m for tr in loaded.tracks for m in tr if m.type == "note_on"]
        assert [m.note for m in note_ons] == [60, 69]

    def test_empty_notes_produces_valid_midi(self):
        data = notes_to_midi_bytes([])
        loaded = MidiFile(file=io.BytesIO(data))
        # tempo track + empty notes track
        assert len(loaded.tracks) == 2
