"""Helpers musicaux : conversions Hz <-> MIDI <-> nom de note, tables VexFlow.

On utilise *uniquement* des di ses (jamais de b mols) pour le naming, ce qui
correspond au comportement attendu par le frontend.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


# Noms des 12 hauteurs chromatiques, di ses uniquement.
SHARP_NAMES: tuple[str, ...] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)

# Map directe nom -> classe de pitch (0..11) pour parser un input du frontend.
_PITCH_CLASS_FROM_NAME: dict[str, int] = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4,
    "F": 5, "E#": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11,
}

# Vocabulaire VexFlow (cf. CLAUDE.md). Cl  = symbole, valeur = nb de 16es.
VEXFLOW_DURATION_TO_SIXTEENTHS: dict[str, int] = {
    "w": 16,  # ronde
    "h": 8,   # blanche
    "q": 4,   # noire
    "8": 2,   # croche
    "16": 1,  # double-croche
}

# Ordre d croissant pour la d composition gourmande.
SIXTEENTHS_DESCENDING: tuple[tuple[str, int], ...] = tuple(
    sorted(VEXFLOW_DURATION_TO_SIXTEENTHS.items(), key=lambda kv: -kv[1])
)


def hz_to_midi(freq_hz: float) -> float:
    """Convertit une fr quence (Hz) en num ro MIDI float (A4 = 69, 440 Hz).

    Retourne NaN pour une entr e <= 0.
    """
    if freq_hz is None or freq_hz <= 0 or math.isnan(freq_hz):
        return float("nan")
    return 69.0 + 12.0 * math.log2(freq_hz / 440.0)


def hz_array_to_midi(freqs_hz: np.ndarray) -> np.ndarray:
    """Version vectoris e de :func:`hz_to_midi`. Les valeurs <= 0 deviennent NaN."""
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    out = np.full_like(freqs, np.nan, dtype=np.float64)
    valid = freqs > 0
    out[valid] = 69.0 + 12.0 * np.log2(freqs[valid] / 440.0)
    return out


def midi_to_pitch_name(midi_number: int) -> str:
    """Convertit un num ro MIDI entier en nom de note avec di ses.

    Exemples : 60 -> "C4", 61 -> "C#4", 69 -> "A4", 84 -> "C6".
    Convention MIDI standard : C4 = 60.
    """
    if midi_number is None:
        raise ValueError("midi_number must be an int, got None")
    midi_int = int(midi_number)
    pitch_class = midi_int % 12
    octave = midi_int // 12 - 1
    return f"{SHARP_NAMES[pitch_class]}{octave}"


def pitch_name_to_midi(pitch_name: str) -> int:
    """Convertit un nom de note (ex "C4", "Bb5", "F#3") en num ro MIDI entier.

    Accepte di ses *et* b mols en entr e pour tol rance, mais le backend  met
    toujours des di ses en sortie.
    """
    if not pitch_name:
        raise ValueError("pitch_name must be a non-empty string")

    name = pitch_name.strip()
    # On s pare la racine (lettre + accident optionnel) de l'octave.
    # L'octave peut  tre n gatif (ex "C-1"), donc on cherche la fin de la racine.
    i = 1
    if i < len(name) and name[i] in ("#", "b"):
        i += 1
    root = name[:i]
    octave_str = name[i:]

    if root not in _PITCH_CLASS_FROM_NAME:
        raise ValueError(f"Unknown pitch name root: {root!r} (in {pitch_name!r})")
    try:
        octave = int(octave_str)
    except ValueError as exc:
        raise ValueError(f"Invalid octave in pitch name {pitch_name!r}") from exc

    return (octave + 1) * 12 + _PITCH_CLASS_FROM_NAME[root]


def decompose_sixteenths(length: int) -> List[str]:
    """D compose une dur e exprim e en 16es en une suite gourmande de dur es VexFlow.

    Exemples :
      1  -> ["16"]
      2  -> ["8"]
      3  -> ["8", "16"]
      4  -> ["q"]
      6  -> ["q", "8"]
      16 -> ["w"]
      20 -> ["w", "q"]
      0  -> []
    """
    if length < 0:
        raise ValueError(f"length must be >= 0, got {length}")
    remaining = int(length)
    result: List[str] = []
    for symbol, value in SIXTEENTHS_DESCENDING:
        while remaining >= value:
            result.append(symbol)
            remaining -= value
    return result


def duration_to_sixteenths(duration: str) -> int:
    """Retourne le nombre de 16es repr sent  par un symbole VexFlow."""
    try:
        return VEXFLOW_DURATION_TO_SIXTEENTHS[duration]
    except KeyError as exc:
        raise ValueError(f"Unknown VexFlow duration: {duration!r}") from exc
