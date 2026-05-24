"""Collapse a quantized note list down to a single-voice melody.

basic-pitch routinely emits overlapping octave-error candidates per onset —
fine for polyphonic audio, fatal for clean humming output. For a single
hummed line we drop any note that overlaps an already-kept one (the louder
candidate wins via sort order) and trim overhangs so the result is strictly
sequential in time.
"""

from __future__ import annotations

from typing import List

from app.models import Note


_ONSET_DEDUP_WINDOW = 0.08  # 80 ms — shorter than any real humming gap, and
# shorter than a 16th note at 120 BPM (125 ms), so consecutive 16ths survive.


def collapse_to_melody(notes: List[Note]) -> List[Note]:
    """Reduce overlapping notes to a single-voice melody."""
    if not notes:
        return []

    # Pass 1 — onset dedupe: basic-pitch's #1 failure mode on hummed audio is
    # emitting the same onset as two overlapping notes at different pitches
    # (fundamental + octave-harmonic confusion, or octave-error pairs). Any
    # note whose start falls within `_ONSET_DEDUP_WINDOW` of an already-kept
    # note is treated as a duplicate of that same syllable and dropped. The
    # louder candidate wins because we sort by descending velocity within each
    # start time (lowest pitch breaks velocity ties — octave-up errors are far
    # more common than octave-down for vocals).
    #
    # Notes that start LATER than the window but before the kept note ends are
    # real sequential notes whose predecessor's release tail got over-extended
    # by basic-pitch — those are handled by the Pass 2 overhang trim, not
    # dropped here.
    sorted_notes = sorted(notes, key=lambda n: (n.start, -n.velocity, n.pitch))
    melody: List[Note] = []
    for n in sorted_notes:
        if melody and (n.start - melody[-1].start) < _ONSET_DEDUP_WINDOW:
            continue
        melody.append(n)

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
