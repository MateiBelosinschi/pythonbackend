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


def monophonize_for_playback(
    notes: List[Note],
    max_repeat_gap: float = 0.15,
) -> List[Note]:
    """Reduce raw basic-pitch output to a single voice for piano playback.

    Unlike `collapse_to_melody` (which targets quantized notes destined for the
    sheet), this preserves the original timings — no grid snap, no key snap.
    The goal is just: at any moment, only ONE note may be ringing.

    Pass 1 — overlap drop: any note that starts while an already-kept note is
    still ringing is dropped. Sort key (start ↑, velocity ↓, pitch ↑) means
    the loudest candidate per onset wins, with the lower pitch breaking ties
    (octave-up harmonic errors are far more common than octave-down for hum).
    This is the textbook monophonic invariant and kills basic-pitch's
    fundamental+harmonic clusters regardless of how spread out their onset
    times are — anything that overlaps a still-ringing note is by definition
    polyphonic and gets cut.

    Pass 2 — same-pitch merge: basic-pitch re-fires onsets several times per
    second during a sustained held syllable, producing many short same-pitch
    fragments. Adjacent same-pitch notes separated by <= `max_repeat_gap`
    seconds are merged into one continuous note — without this the piano
    re-strikes the same key several times during what was meant to be one
    held note.
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: (n.start, -n.velocity, n.pitch))
    kept: List[Note] = []
    for n in sorted_notes:
        if kept and n.start < kept[-1].end:
            continue
        kept.append(n)

    merged: List[Note] = [kept[0]]
    for n in kept[1:]:
        prev = merged[-1]
        if n.pitch == prev.pitch and (n.start - prev.end) <= max_repeat_gap:
            merged[-1] = Note(
                pitch=prev.pitch,
                start=prev.start,
                end=max(prev.end, n.end),
                velocity=max(prev.velocity, n.velocity),
            )
        else:
            merged.append(n)
    return merged


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
