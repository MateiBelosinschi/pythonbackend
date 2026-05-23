"""Export MIDI stateless d'une liste ``MusicalNoteSchema``.

On  met un fichier MIDI standard format 1 :
- piste 0 : meta (tempo  ``settings.BPM``).
- piste 1 : notes (canal 0, instrument par d faut = piano).

R solution : ``ticks_per_beat = 480`` (standard de facto). Cela permet
d'exprimer un 16e de note  120 BPM avec exactement 120 ticks (entier).
"""
from __future__ import annotations

import io
from typing import Iterable, List

from mido import MetaMessage, MidiFile, MidiTrack, Message, bpm2tempo

from app.api.schemas import MusicalNoteSchema
from app.config import settings
from app.utils.music import duration_to_sixteenths, pitch_name_to_midi


TICKS_PER_BEAT: int = 480
# V locit  MIDI par d faut pour toutes les notes  mises.
DEFAULT_VELOCITY: int = 96


def _ticks_per_sixteenth() -> int:
    """Nombre de ticks par 16e (entier exact si SUBDIVISIONS_PER_BEAT divise TICKS_PER_BEAT)."""
    return TICKS_PER_BEAT // settings.SUBDIVISIONS_PER_BEAT


def _ticks_for_duration(duration: str) -> int:
    return duration_to_sixteenths(duration) * _ticks_per_sixteenth()


def build_midi_file(notes: Iterable[MusicalNoteSchema]) -> MidiFile:
    """Construit un ``MidiFile`` mido  partir des notes/silences fournis."""
    midi = MidiFile(type=1, ticks_per_beat=TICKS_PER_BEAT)

    meta_track = MidiTrack()
    meta_track.append(MetaMessage("set_tempo", tempo=bpm2tempo(settings.BPM), time=0))
    meta_track.append(MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    meta_track.append(MetaMessage("end_of_track", time=0))
    midi.tracks.append(meta_track)

    note_track = MidiTrack()
    midi.tracks.append(note_track)

    # ``pending_rest_ticks`` accumule les silences afin de les fusionner avec
    # le ``time`` (delta) du prochain note_on.
    pending_rest_ticks = 0

    for note in notes:
        ticks = _ticks_for_duration(note.duration)
        if note.isRest:
            pending_rest_ticks += ticks
            continue

        midi_number = pitch_name_to_midi(note.pitch)
        note_track.append(
            Message(
                "note_on",
                note=midi_number,
                velocity=DEFAULT_VELOCITY,
                time=pending_rest_ticks,
                channel=0,
            )
        )
        note_track.append(
            Message(
                "note_off",
                note=midi_number,
                velocity=0,
                time=ticks,
                channel=0,
            )
        )
        pending_rest_ticks = 0

    note_track.append(MetaMessage("end_of_track", time=pending_rest_ticks))
    return midi


def notes_to_midi_bytes(notes: List[MusicalNoteSchema]) -> bytes:
    """S rialise ``notes`` en bytes MIDI pr ts  renvoyer par l'API."""
    midi = build_midi_file(notes)
    buffer = io.BytesIO()
    midi.save(file=buffer)
    return buffer.getvalue()
