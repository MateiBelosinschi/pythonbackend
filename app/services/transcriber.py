"""basic-pitch inference wrapper.

Returns a `pretty_midi.PrettyMIDI` so downstream code can lean on the symbolic
music library for all timing math (see CLAUDE.md — no manual array arithmetic
for note timing).

basic-pitch's `predict()` only accepts a file path, so we write the
already-pre-processed waveform to a short-lived WAV in the OS temp dir.
"""

from __future__ import annotations

import gc
import os
import tempfile
from typing import List, Optional

import numpy as np
import pretty_midi
import soundfile as sf
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

from app.models import Note

_MODEL_PATH = ICASSP_2022_MODEL_PATH


def transcribe(
    waveform: np.ndarray,
    sr: int,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 11.0,
    minimum_frequency: Optional[float] = None,
    maximum_frequency: Optional[float] = 3000.0,
) -> pretty_midi.PrettyMIDI:
    """Run basic-pitch on a pre-processed waveform and return the resulting MIDI.

    Defaults match the Spotify hosted demo exactly so the raw output (used for
    piano playback) sounds the same as what the user already validated there:
    onset 0.5, frame 0.3, min-note 11 ms, no min-freq filter, max 3000 Hz.
    Downstream cleanup (quantize/monophonic/melody_cleanup) only feeds the
    *sheet-music* view — playback uses these raw notes directly via
    `pretty_midi_to_notes`.
    """
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="musicme_")
    os.close(fd)
    try:
        sf.write(path, waveform, sr, subtype="PCM_16")
        _, midi_data, _ = predict(
            audio_path=path,
            model_or_model_path=_MODEL_PATH,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    gc.collect()
    if not isinstance(midi_data, pretty_midi.PrettyMIDI):
        raise TypeError(f"Unexpected basic-pitch return type: {type(midi_data)!r}")
    return midi_data


def pretty_midi_to_notes(pm: pretty_midi.PrettyMIDI) -> List[Note]:
    """Flatten a PrettyMIDI to a sorted list of `Note` with timings unchanged.

    This is the "playback" path: no quantization, no monophonic collapse, no
    key-snapping — the same shape Spotify's basic-pitch demo plays back, just
    re-encoded in our JSON contract so the frontend's piano sampler can play
    it directly.
    """
    out: List[Note] = []
    for inst in pm.instruments:
        for n in inst.notes:
            out.append(
                Note(
                    pitch=int(n.pitch),
                    start=float(n.start),
                    end=float(n.end),
                    velocity=int(n.velocity),
                )
            )
    out.sort(key=lambda n: (n.start, n.pitch))
    return out
