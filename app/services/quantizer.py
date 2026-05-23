"""Quantification d'une piste de fréquences CREPE vers la grille rythmique.

Pipeline :
1. Pour chaque cellule de la grille (16e de note), on regroupe les frames CREPE.
2. On décide si la cellule est "voix présente" via confiance CREPE + RMS.
3. Le pitch de la cellule est la médiane (en demi-tons) des frames voisées,
   arrondie au demi-ton entier puis filtrée par la plage vocale autorisée.
4. Lissage inter-cellules (hystérésis) pour stabiliser les notes tenues.
5. On fusionne les cellules adjacentes de même pitch en runs ``(pitch, length)``.
6. Chaque run est décomposé en durées VexFlow (w, h, q, 8, 16).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from app.api.schemas import (
    CrepeFrameDebug,
    GridMetadata,
    MusicalNoteSchema,
    TranscriptionDebugInfo,
)
from app.config import settings
from app.utils.music import (
    decompose_sixteenths,
    hz_array_to_midi,
    midi_to_pitch_name,
)


# Valeur sentinelle dans le tableau de cellules pour représenter un silence.
REST_CELL: int = -1

# Convention de nom utilisée pour le champ ``pitch`` quand ``isRest=True``.
REST_PITCH_NAME: str = "rest"


@dataclass(frozen=True)
class QuantizeParams:
    """Paramètres de grille surchargables par requête."""

    bpm: int = settings.BPM
    grid_offset_sec: float = 0.0

    @property
    def cell_seconds(self) -> float:
        return 60.0 / self.bpm / settings.SUBDIVISIONS_PER_BEAT


@dataclass
class QuantizeResult:
    notes: List[MusicalNoteSchema]
    cells: np.ndarray
    debug: Optional[TranscriptionDebugInfo] = None


def _align_rms_to_times(rms: np.ndarray, times_sec: np.ndarray) -> np.ndarray:
    """Pour chaque timestamp CREPE, retourne la valeur RMS du frame correspondant."""
    if rms.size == 0:
        return np.zeros_like(times_sec, dtype=np.float32)
    if rms.size == times_sec.size:
        return np.asarray(rms, dtype=np.float32)
    step_sec = settings.step_seconds
    idx = np.rint(times_sec / step_sec).astype(np.int64)
    idx = np.clip(idx, 0, rms.size - 1)
    return rms[idx]


def _cell_is_voiced(voiced_count: int, total: int) -> bool:
    """Décide si une cellule compte comme voisée (vote assoupli pour sustains)."""
    if total == 0:
        return False
    if voiced_count >= settings.MIN_VOICED_FRAMES_PER_CELL:
        return True
    return voiced_count / total >= settings.VOICED_RATIO_THRESHOLD


def quantize_cells(
    times_sec: np.ndarray,
    freqs_hz: np.ndarray,
    confidences: np.ndarray,
    rms: np.ndarray,
    duration_sec: float,
    params: Optional[QuantizeParams] = None,
) -> np.ndarray:
    """Calcule le tableau de cellules (1 valeur par 16e de note)."""
    qp = params or QuantizeParams()
    cell_sec = qp.cell_seconds
    n_cells = max(0, int(math.ceil(duration_sec / cell_sec - 1e-9)))
    cells = np.full(n_cells, REST_CELL, dtype=np.int64)
    if n_cells == 0 or times_sec.size == 0:
        return cells

    cell_idx = np.floor((times_sec - qp.grid_offset_sec) / cell_sec).astype(np.int64)
    in_range = (cell_idx >= 0) & (cell_idx < n_cells)
    if not np.any(in_range):
        return cells

    cell_idx = cell_idx[in_range]
    freqs = freqs_hz[in_range]
    conf = confidences[in_range]
    times_in = times_sec[in_range]

    rms_per_frame = _align_rms_to_times(rms, times_in)

    voiced_mask = (
        (conf >= settings.CONFIDENCE_THRESHOLD)
        & (rms_per_frame >= settings.RMS_THRESHOLD)
        & (freqs > 0)
    )

    order = np.argsort(cell_idx, kind="stable")
    sorted_cells = cell_idx[order]
    sorted_freqs = freqs[order]
    sorted_voiced = voiced_mask[order]

    unique_cells, start_idx = np.unique(sorted_cells, return_index=True)
    end_idx = np.append(start_idx[1:], sorted_cells.size)

    for cell_id, lo, hi in zip(unique_cells, start_idx, end_idx):
        total = hi - lo
        if total == 0:
            continue
        voiced_in_cell = sorted_voiced[lo:hi]
        voiced_count = int(voiced_in_cell.sum())
        if not _cell_is_voiced(voiced_count, total):
            continue

        cell_freqs = sorted_freqs[lo:hi][voiced_in_cell]
        if cell_freqs.size == 0:
            continue

        midi_values = hz_array_to_midi(cell_freqs)
        midi_values = midi_values[~np.isnan(midi_values)]
        if midi_values.size == 0:
            continue

        midi_int = int(round(float(np.median(midi_values))))
        if midi_int < settings.MIDI_MIN or midi_int > settings.MIDI_MAX:
            continue

        cells[cell_id] = midi_int

    return smooth_cell_pitches(cells)


def smooth_cell_pitches(cells: np.ndarray) -> np.ndarray:
    """Stabilise les notes tenues en corrigeant une dérive minoritaire de pitch."""
    if cells.size == 0:
        return cells

    result = cells.copy()
    tolerance = settings.PITCH_HYSTERESIS_SEMITONES
    min_sustain_cells = 4

    i = 0
    while i < result.size:
        if result[i] == REST_CELL:
            i += 1
            continue

        run_start = i
        while i < result.size and result[i] != REST_CELL:
            if i > run_start and abs(int(result[i]) - int(result[i - 1])) > tolerance:
                break
            i += 1

        run = result[run_start:i]
        if run.size < min_sustain_cells:
            continue

        pitch_range = int(run.max()) - int(run.min())
        if pitch_range > tolerance:
            continue

        anchor = int(result[run_start])
        drift_count = int(np.sum(run != anchor))
        if 0 < drift_count < run.size // 2:
            result[run_start:i] = anchor

    return result


def cells_to_runs(cells: np.ndarray) -> List[Tuple[int, int]]:
    """Fusionne les cellules adjacentes égales en runs ``(value, length_16ths)``."""
    runs: List[Tuple[int, int]] = []
    if cells.size == 0:
        return runs

    current_value = int(cells[0])
    current_length = 1
    for v in cells[1:]:
        v_int = int(v)
        if v_int == current_value:
            current_length += 1
        else:
            runs.append((current_value, current_length))
            current_value = v_int
            current_length = 1
    runs.append((current_value, current_length))
    return runs


def runs_to_notes(runs: List[Tuple[int, int]]) -> List[MusicalNoteSchema]:
    """Convertit des runs en une liste plate de ``MusicalNoteSchema``."""
    notes: List[MusicalNoteSchema] = []
    for value, length in runs:
        if length <= 0:
            continue
        is_rest = value == REST_CELL
        pitch_name = REST_PITCH_NAME if is_rest else midi_to_pitch_name(value)
        for symbol in decompose_sixteenths(length):
            notes.append(
                MusicalNoteSchema(
                    pitch=pitch_name,
                    duration=symbol,  # type: ignore[arg-type]
                    isRest=is_rest,
                )
            )
    return notes


def build_crepe_debug_track(
    times_sec: np.ndarray,
    freqs_hz: np.ndarray,
    confidences: np.ndarray,
) -> List[CrepeFrameDebug]:
    """Construit la piste CREPE frame par frame pour le mode debug."""
    midi_values = hz_array_to_midi(freqs_hz)
    track: List[CrepeFrameDebug] = []
    for t, freq, conf, midi in zip(times_sec, freqs_hz, confidences, midi_values):
        midi_rounded: Optional[int] = None
        if not math.isnan(midi) and freq > 0:
            midi_rounded = int(round(float(midi)))
        track.append(
            CrepeFrameDebug(
                time=float(t),
                freq_hz=float(freq),
                confidence=float(conf),
                midi_rounded=midi_rounded,
            )
        )
    return track


def _cells_to_debug_list(cells: np.ndarray) -> List[Optional[int]]:
    return [None if int(v) == REST_CELL else int(v) for v in cells]


def quantize_pipeline(
    times_sec: np.ndarray,
    freqs_hz: np.ndarray,
    confidences: np.ndarray,
    rms: np.ndarray,
    duration_sec: float,
    params: Optional[QuantizeParams] = None,
    debug: bool = False,
) -> QuantizeResult:
    """Pipeline complet : frames CREPE -> notes (+ debug optionnel)."""
    qp = params or QuantizeParams()
    cells = quantize_cells(
        times_sec, freqs_hz, confidences, rms, duration_sec, params=qp
    )
    runs = cells_to_runs(cells)
    notes = runs_to_notes(runs)

    debug_info: Optional[TranscriptionDebugInfo] = None
    if debug:
        debug_info = TranscriptionDebugInfo(
            crepe_track=build_crepe_debug_track(times_sec, freqs_hz, confidences),
            cells=_cells_to_debug_list(cells),
            grid=GridMetadata(
                bpm=qp.bpm,
                cell_seconds=qp.cell_seconds,
                offset_sec=qp.grid_offset_sec,
            ),
        )

    return QuantizeResult(notes=notes, cells=cells, debug=debug_info)


def quantize_to_notes(
    times_sec: np.ndarray,
    freqs_hz: np.ndarray,
    confidences: np.ndarray,
    rms: np.ndarray,
    duration_sec: float,
    params: Optional[QuantizeParams] = None,
) -> List[MusicalNoteSchema]:
    """Helper de bout en bout : frames CREPE -> liste de ``MusicalNoteSchema``."""
    return quantize_pipeline(
        times_sec, freqs_hz, confidences, rms, duration_sec, params=params
    ).notes
