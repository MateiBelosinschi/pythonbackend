"""
midi_export.py — MIDI export service using mido.

Converts a list of MusicalNoteSchema-compatible dicts to a MIDI Type-1 file
and returns the raw bytes, ready to be streamed back as a file download.
"""
import io
import logging
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo

from app.utils.constants import VEXFLOW_DURATION_TO_BEATS

logger = logging.getLogger(__name__)

# MIDI ticks per quarter-note (standard resolution)
_TICKS_PER_BEAT = 480
_DEFAULT_VELOCITY = 80


def _note_name_to_midi(pitch: str) -> int:
    """
    Converts a pitch string like "C4", "G#5", "Bb3" to a MIDI note number.
    Returns 60 (C4) as a safe fallback for unrecognised strings.
    """
    NOTE_MAP = {
        "C": 0, "C#": 1, "Db": 1,
        "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "Fb": 4,
        "F": 5, "F#": 6, "Gb": 6,
        "G": 7, "G#": 8, "Ab": 8,
        "A": 9, "A#": 10, "Bb": 10,
        "B": 11, "Cb": 11,
    }
    try:
        # Split "G#5" → note_part="G#", octave=5
        if len(pitch) >= 2 and pitch[1] in ("#", "b"):
            note_part = pitch[:2]
            octave = int(pitch[2:])
        else:
            note_part = pitch[0]
            octave = int(pitch[1:])
        semitone = NOTE_MAP[note_part]
        midi = (octave + 1) * 12 + semitone
        return max(0, min(127, midi))
    except Exception:
        logger.warning(f"Could not parse pitch '{pitch}', defaulting to C4 (60).")
        return 60


def notes_to_midi_bytes(notes: list[dict], tempo: int = 120) -> bytes:
    """
    Convert a list of MusicalNoteSchema dicts to MIDI file bytes.

    Args:
        notes:  List of {"pitch": str, "duration": str, "isRest": bool}
        tempo:  BPM (30-300)

    Returns:
        Raw MIDI bytes (can be written to a .mid file or streamed to the client).

    Raises:
        ValueError: if a note has an unrecognised VexFlow duration string.
    """
    midi_file = MidiFile(type=0, ticks_per_beat=_TICKS_PER_BEAT)
    track = MidiTrack()
    midi_file.tracks.append(track)

    # Set tempo
    track.append(MetaMessage("set_tempo", tempo=bpm2tempo(tempo), time=0))

    for note in notes:
        vex_dur = note["duration"]
        if vex_dur not in VEXFLOW_DURATION_TO_BEATS:
            raise ValueError(f"Unknown VexFlow duration '{vex_dur}'.")

        beats = VEXFLOW_DURATION_TO_BEATS[vex_dur]
        ticks = int(beats * _TICKS_PER_BEAT)

        if note["isRest"]:
            # Rest: just advance time with a zero-velocity note or simple delay
            # We use a rest by emitting nothing — mido handles delta times directly.
            # Add an empty time-shift by appending a note-on at velocity 0
            track.append(Message("note_on", note=60, velocity=0, time=ticks))
        else:
            midi_note = _note_name_to_midi(note["pitch"])
            track.append(Message("note_on",  note=midi_note, velocity=_DEFAULT_VELOCITY, time=0))
            track.append(Message("note_off", note=midi_note, velocity=0, time=ticks))

    track.append(MetaMessage("end_of_track", time=0))

    buf = io.BytesIO()
    midi_file.save(file=buf)
    buf.seek(0)
    return buf.read()
