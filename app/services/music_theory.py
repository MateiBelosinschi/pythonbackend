"""
music_theory.py — Scale snapping (key correction) for detected MIDI pitches.

Per the updated claude.md architecture, instrument transposition is handled
entirely by the frontend (VexFlow). This service only corrects slightly off-pitch
notes to the nearest degree of the chosen key signature.
"""
import logging

logger = logging.getLogger(__name__)

# Relative-minor → major key mapping (for scale pitch class lookup)
_KEY_TO_RELATIVE_MAJOR = {
    "Am": "C",  "Em": "G",  "Bm": "D",  "F#m": "A", "C#m": "E",
    "G#m": "B", "D#m": "F#","A#m": "C#","Dm": "F",  "Gm": "Bb",
    "Cm": "Eb", "Fm": "Ab", "Bbm": "Db","Ebm": "Gb","Abm": "Cb",
    # Major keys map to themselves
    "C": "C",  "G": "G",  "D": "D",  "A": "A",  "E": "E",
    "B": "B",  "F#": "F#","C#": "C#","F": "F",  "Bb": "Bb",
    "Eb": "Eb","Ab": "Ab","Db": "Db","Gb": "Gb","Cb": "Cb",
}

# Pitch classes (0-11) for each major scale
_MAJOR_PITCH_CLASSES: dict[str, list[int]] = {
    "C":  [0, 2, 4, 5, 7, 9, 11],
    "G":  [7, 9, 11, 0, 2, 4, 6],
    "D":  [2, 4, 6, 7, 9, 11, 1],
    "A":  [9, 11, 1, 2, 4, 6, 8],
    "E":  [4, 6, 8, 9, 11, 1, 3],
    "B":  [11, 1, 3, 4, 6, 8, 10],
    "F#": [6, 8, 10, 11, 1, 3, 5],
    "C#": [1, 3, 5, 6, 8, 10, 0],
    "F":  [5, 7, 9, 10, 0, 2, 4],
    "Bb": [10, 0, 2, 3, 5, 7, 9],
    "Eb": [3, 5, 7, 8, 10, 0, 2],
    "Ab": [8, 10, 0, 1, 3, 5, 7],
    "Db": [1, 3, 5, 6, 8, 10, 0],
    "Gb": [6, 8, 10, 11, 1, 3, 5],
    "Cb": [11, 1, 3, 4, 6, 8, 10],
}


def _normalise_key(key_name: str) -> str:
    """Return a canonical key name, e.g. 'am' → 'Am', 'bb' → 'Bb'."""
    k = key_name.strip()
    if len(k) > 1 and k[1].lower() == "m":
        return k[0].upper() + "m"
    if len(k) > 1 and k[1] in ("b", "#"):
        return k[0].upper() + k[1]
    return k[0].upper()


def _allowed_classes(key_name: str) -> list[int]:
    rel_major = _KEY_TO_RELATIVE_MAJOR.get(_normalise_key(key_name), "C")
    return _MAJOR_PITCH_CLASSES.get(rel_major, [0, 2, 4, 5, 7, 9, 11])


def snap_to_scale(midi_note: int, key_name: str) -> int:
    """
    Snap a MIDI note to the nearest scale degree in the given key.
    MIDI 0 (rest) is returned unchanged.
    """
    if midi_note == 0:
        return 0

    pitch_class = midi_note % 12
    octave = midi_note // 12
    allowed = _allowed_classes(key_name)

    def _dist(c: int) -> int:
        diff = abs(c - pitch_class)
        return min(diff, 12 - diff)

    closest = min(allowed, key=_dist)

    # Pick candidate from same octave or adjacent octaves, take closest to original
    candidates = [
        (octave - 1) * 12 + closest,
        octave * 12 + closest,
        (octave + 1) * 12 + closest,
    ]
    best = min(candidates, key=lambda m: abs(m - midi_note))
    return max(0, min(127, best))
