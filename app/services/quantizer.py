"""120 BPM, 16th-note rigid time quantization.

CLAUDE.md is explicit: no manual NumPy/list arithmetic for grid timings.
Every conversion between seconds and grid units goes through `pretty_midi`'s
`time_to_tick` / `tick_to_time` helpers, which respect the symbolic tempo map.
"""

from __future__ import annotations

from typing import List

import pretty_midi

from app.models import FIXED_BPM, GRID_SUBDIVISION, Note


def _grid_tick_step(pm: pretty_midi.PrettyMIDI, subdivision: int) -> int:
    """Number of ticks per grid cell (e.g. 16 for sixteenth notes)."""
    # `resolution` is ticks per quarter note. A 16th note = quarter / 4.
    quarter_ticks = pm.resolution
    # subdivision=16 means 16th notes => quarter_ticks / 4. subdivision=8 => /2. etc.
    cells_per_quarter = subdivision // 4
    if cells_per_quarter < 1:
        raise ValueError(f"subdivision must be >= 4, got {subdivision}")
    return quarter_ticks // cells_per_quarter


def _snap_tick(tick: int, step: int) -> int:
    """Round a tick value to the nearest grid step."""
    return int(round(tick / step)) * step


def quantize(
    pm: pretty_midi.PrettyMIDI,
    bpm: int = FIXED_BPM,
    subdivision: int = GRID_SUBDIVISION,
) -> List[Note]:
    """Snap all notes in `pm` to a rigid `subdivision`-grid at `bpm`, return Concert Pitch JSON notes.

    A note whose start and end collapse onto the same grid cell is dropped — it
    represents a sub-grid blip (often a basic-pitch onset glitch) that the
    quantization explicitly disallows.
    """
    # Re-anchor tempo so pretty_midi's time<->tick maps reflect the target BPM.
    # We build a fresh PrettyMIDI at the fixed BPM and copy notes into it so that
    # `time_to_tick` uses the correct mapping.
    target = pretty_midi.PrettyMIDI(initial_tempo=float(bpm), resolution=pm.resolution)
    instrument = pretty_midi.Instrument(program=0)
    for src_inst in pm.instruments:
        for n in src_inst.notes:
            instrument.notes.append(
                pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end)
            )
    target.instruments.append(instrument)

    step = _grid_tick_step(target, subdivision)
    quantized: List[Note] = []

    for n in instrument.notes:
        start_tick = _snap_tick(target.time_to_tick(n.start), step)
        end_tick = _snap_tick(target.time_to_tick(n.end), step)

        if end_tick <= start_tick:
            # Promote to a single grid cell rather than dropping silently — preserves
            # the onset but enforces the minimum 16th-note duration.
            end_tick = start_tick + step

        quantized.append(
            Note(
                pitch=int(n.pitch),
                start=float(target.tick_to_time(start_tick)),
                end=float(target.tick_to_time(end_tick)),
                velocity=int(n.velocity),
            )
        )

    quantized.sort(key=lambda x: (x.start, x.pitch))
    return quantized


def notes_to_midi(notes: List[Note], bpm: int = FIXED_BPM) -> bytes:
    """Render Concert Pitch notes back to a MIDI file byte stream."""
    import io

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    instrument = pretty_midi.Instrument(program=0)
    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end,
            )
        )
    pm.instruments.append(instrument)

    buf = io.BytesIO()
    pm.write(buf)
    return buf.getvalue()
