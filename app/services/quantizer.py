"""
quantizer.py — NumPy-based rhythm quantizer.

Takes a list of per-frame MIDI pitches (one per 10 ms hop) and a tempo in BPM,
groups consecutive identical pitches, then snaps each segment's duration to the
nearest VexFlow note value using NumPy vectorised operations.

Returns a list of dicts compatible with MusicalNoteSchema:
    [{"pitch": "C4", "duration": "q", "isRest": False}, ...]
"""
import numpy as np
import logging
from app.utils.constants import BEATS_TO_VEXFLOW
from app.utils.helpers import midi_to_note_name

logger = logging.getLogger(__name__)

# Allowed beat durations in ascending order (NumPy array for fast searchsorted)
_ALLOWED_BEATS = np.array(sorted(BEATS_TO_VEXFLOW.keys()))  # [0.25, 0.5, 1.0, 2.0, 4.0]


def _snap_to_grid(raw_beats: float) -> float:
    """
    Snap a raw beat duration to the nearest allowed VexFlow bucket using
    NumPy searchsorted for O(log n) lookup.
    """
    idx = np.searchsorted(_ALLOWED_BEATS, raw_beats)
    # Clamp index to valid range
    idx = int(np.clip(idx, 0, len(_ALLOWED_BEATS) - 1))
    # Check left neighbour for closer match
    if idx > 0 and abs(_ALLOWED_BEATS[idx - 1] - raw_beats) <= abs(_ALLOWED_BEATS[idx] - raw_beats):
        idx -= 1
    return float(_ALLOWED_BEATS[idx])


def quantize(midi_frames: list[int], tempo: int) -> list[dict]:
    """
    Quantize a list of per-frame MIDI pitch values to MusicalNoteSchema dicts.

    Args:
        midi_frames: MIDI note per 10 ms frame (0 = silence/rest).
        tempo:       Tempo in BPM used to convert frame counts → beats.

    Returns:
        List of dicts with keys: pitch (str), duration (str), isRest (bool).
    """
    if not midi_frames:
        return []

    hop_duration_s = 0.010          # 10 ms per frame
    seconds_per_beat = 60.0 / tempo

    # ── Step 1: run-length encode consecutive identical pitches ──────────────
    arr = np.array(midi_frames, dtype=np.int32)
    # Find positions where the value changes
    change_indices = np.where(np.diff(arr) != 0)[0] + 1
    split_groups = np.split(arr, change_indices)

    # ── Step 2: convert frame count → beats, snap to grid ───────────────────
    notes = []
    for group in split_groups:
        midi = int(group[0])
        n_frames = len(group)
        raw_seconds = n_frames * hop_duration_s
        raw_beats = raw_seconds / seconds_per_beat
        snapped_beats = _snap_to_grid(raw_beats)
        vexflow_dur = BEATS_TO_VEXFLOW[snapped_beats]

        is_rest = (midi == 0)
        pitch_str = "R" if is_rest else midi_to_note_name(midi)

        notes.append({
            "pitch": pitch_str,
            "duration": vexflow_dur,
            "isRest": is_rest,
        })

    return notes
