"""Collapse a quantized note list down to a single-voice melody.

basic-pitch routinely emits overlapping octave-error candidates per onset —
fine for polyphonic audio, fatal for clean humming output. For a single
hummed line we keep one note per 16th-note start cell (highest velocity
wins, lowest pitch breaks ties) and trim overhangs so nothing overlaps.
"""

from __future__ import annotations

from typing import Dict, List

from app.models import FIXED_BPM, GRID_SUBDIVISION, Note


def _cell(seconds: float, bpm: int, subdivision: int) -> int:
    """Map a quantized timestamp to its integer grid-cell index."""
    cell_duration = 60.0 / bpm / (subdivision / 4)
    return int(round(seconds / cell_duration))


def collapse_to_melody(
    notes: List[Note],
    bpm: int = FIXED_BPM,
    subdivision: int = GRID_SUBDIVISION,
) -> List[Note]:
    """Reduce overlapping notes to a single-voice melody on the 16th-note grid.

    Pass 1 — one note per start cell: among notes that snap to the same grid
    onset, keep the highest-velocity candidate (basic-pitch's proxy for model
    confidence); on a tie, take the lowest pitch (octave-up errors are far
    more common than octave-down for vocals).

    Pass 2 — trim overhangs: if a kept note runs past the next note's onset,
    shorten it so the melody is strictly sequential.
    """
    if not notes:
        return []

    # Pass 1: one note per start cell. When a collision is the SAME pitch (a
    # held syllable re-detected as two onsets), keep the louder one. When the
    # collision is a DIFFERENT pitch (two real hummed notes that happened to
    # snap to the same 16th cell because the user's tempo ≠ 120 BPM), bump
    # the loser forward to the next free cell instead of deleting it —
    # silently dropping notes here is the biggest source of "missing notes"
    # complaints from end users.
    winners: Dict[int, Note] = {}
    for n in sorted(notes, key=lambda x: x.start):
        key = _cell(n.start, bpm, subdivision)
        current = winners.get(key)
        if current is None:
            winners[key] = n
            continue
        if current.pitch == n.pitch:
            # Same pitch: dedupe, keep the louder candidate.
            if n.velocity > current.velocity:
                winners[key] = n
            continue
        # Different pitch: pick the louder for this cell, push the other to
        # the next empty cell so it survives.
        loser = current if n.velocity > current.velocity else n
        winner = n if n.velocity > current.velocity else current
        winners[key] = winner
        next_key = key + 1
        while next_key in winners:
            next_key += 1
        winners[next_key] = loser

    melody = sorted(winners.values(), key=lambda x: x.start)

    # Pass 2: trim overhangs so the melody is monophonic in time as well.
    cleaned: List[Note] = []
    for i, note in enumerate(melody):
        if i + 1 < len(melody) and note.end > melody[i + 1].start:
            note = Note(
                pitch=note.pitch,
                start=note.start,
                end=melody[i + 1].start,
                velocity=note.velocity,
            )
        cleaned.append(note)

    return cleaned
