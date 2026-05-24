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

import numpy as np
import pretty_midi
import soundfile as sf
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

_MODEL_PATH = ICASSP_2022_MODEL_PATH


def transcribe(
    waveform: np.ndarray,
    sr: int,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 120.0,
    minimum_frequency: float = 65.0,
    maximum_frequency: float = 1000.0,
    melodia_trick: bool = True,
) -> pretty_midi.PrettyMIDI:
    """Run basic-pitch on a pre-processed waveform and return the resulting MIDI.

    Tuned for humming, not the general polyphonic case: `melodia_trick=True`
    enables basic-pitch's monophonic post-processor (suppresses harmonic
    siblings that otherwise show up as a stack of simultaneous notes), and
    `minimum_note_length` is raised to basic-pitch's own default of 120 ms to
    cut sub-grid filler blips from breath and vibrato wobble. The frequency
    band is widened to 65 Hz so low male humming (~A2/B2) isn't dropped.
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
            melodia_trick=melodia_trick,
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
