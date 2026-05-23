"""Quantification d'une piste de fr quences CREPE vers la grille rythmique 120 BPM.

Pipeline :
1. Pour chaque cellule de la grille (16e de note  120 BPM = 125 ms), on
   r colte les frames CREPE qui tombent dedans.
2. On d cide si la cellule est "voix pr sente" via un vote majoritaire sur
   la confiance CREPE et le RMS du frame correspondant.
3. Le pitch de la cellule est la m diane (en demi-tons) des frames vois es,
   arrondie au demi-ton entier puis "clamp e"  la plage vocale autoris e
   (hors plage => silence).
4. On fusionne les cellules adjacentes de m me pitch en runs ``(pitch, length)``.
5. Chaque run est d compos  en une suite de dur es VexFlow (w, h, q, 8, 16) ;
   un run plus long qu'une ronde produit plusieurs ``MusicalNoteSchema``
   cons cutives de m me pitch (le frontend peut ensuite lier visuellement).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from app.api.schemas import MusicalNoteSchema
from app.config import settings
from app.utils.music import (
    decompose_sixteenths,
    hz_array_to_midi,
    midi_to_pitch_name,
)


# Valeur sentinelle dans le tableau de cellules pour repr senter un silence.
REST_CELL: int = -1

# Convention de nom utilis e pour le champ ``pitch`` quand ``isRest=True``.
REST_PITCH_NAME: str = "rest"

# Seuil de vote majoritaire : il faut au moins 50% de frames "vois es"
# dans une cellule pour la consid rer comme une note.
VOICED_RATIO_THRESHOLD: float = 0.5


def _align_rms_to_times(rms: np.ndarray, times_sec: np.ndarray) -> np.ndarray:
    """Pour chaque time stamp CREPE, retourne la valeur RMS du frame correspondant.

    Les deux arrays sont cens s utiliser le m me ``STEP_MS`` mais peuvent avoir
    une longueur l g rement diff rente (effets de bord / centrage). On fait
    donc un mapping par index direct avec clipping aux bornes.
    """
    if rms.size == 0:
        return np.zeros_like(times_sec, dtype=np.float32)
    step_sec = settings.step_seconds
    idx = np.rint(times_sec / step_sec).astype(np.int64)
    idx = np.clip(idx, 0, rms.size - 1)
    return rms[idx]


def quantize_cells(
    times_sec: np.ndarray,
    freqs_hz: np.ndarray,
    confidences: np.ndarray,
    rms: np.ndarray,
    duration_sec: float,
) -> np.ndarray:
    """Calcule le tableau de cellules (1 valeur par 16e de note).

    Args:
        times_sec: timestamps CREPE (1D, secondes).
        freqs_hz:  fr quences CREPE (1D, Hz, 0 = non d tect ).
        confidences: confiances CREPE (1D, dans [0, 1]).
        rms: enveloppe RMS du signal (1D, m me hop que ``STEP_MS``).
        duration_sec: dur e totale du signal en secondes (utilis e pour
            d terminer le nombre de cellules).

    Returns:
        ``np.ndarray`` 1D ``int64`` de longueur ``n_cells``. Chaque entr e est
        soit un num ro MIDI valide (dans la plage vocale), soit ``REST_CELL``.
    """
    cell_sec = settings.cell_seconds
    n_cells = max(0, int(math.floor(duration_sec / cell_sec + 1e-9)))
    cells = np.full(n_cells, REST_CELL, dtype=np.int64)
    if n_cells == 0 or times_sec.size == 0:
        return cells

    # Frame -> index de cellule
    cell_idx = np.floor(times_sec / cell_sec).astype(np.int64)
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

    # Tri stable des frames par cellule pour pouvoir bucketiser efficacement.
    order = np.argsort(cell_idx, kind="stable")
    sorted_cells = cell_idx[order]
    sorted_freqs = freqs[order]
    sorted_voiced = voiced_mask[order]

    # Bornes de chaque bucket dans le tableau tri .
    unique_cells, start_idx = np.unique(sorted_cells, return_index=True)
    end_idx = np.append(start_idx[1:], sorted_cells.size)

    for cell_id, lo, hi in zip(unique_cells, start_idx, end_idx):
        total = hi - lo
        if total == 0:
            continue
        voiced_in_cell = sorted_voiced[lo:hi]
        voiced_count = int(voiced_in_cell.sum())
        if voiced_count / total < VOICED_RATIO_THRESHOLD:
            continue  # reste REST_CELL

        cell_freqs = sorted_freqs[lo:hi][voiced_in_cell]
        if cell_freqs.size == 0:
            continue

        midi_values = hz_array_to_midi(cell_freqs)
        midi_values = midi_values[~np.isnan(midi_values)]
        if midi_values.size == 0:
            continue

        midi_int = int(round(float(np.median(midi_values))))
        if midi_int < settings.MIDI_MIN or midi_int > settings.MIDI_MAX:
            continue  # hors plage vocale -> on garde le silence

        cells[cell_id] = midi_int

    return cells


def cells_to_runs(cells: np.ndarray) -> List[Tuple[int, int]]:
    """Fusionne les cellules adjacentes  gales en runs ``(value, length_16ths)``.

    ``value`` est soit un num ro MIDI, soit ``REST_CELL`` pour un silence.
    """
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
    """Convertit des runs en une liste plate de ``MusicalNoteSchema``.

    Une dur e non repr sentable par un seul symbole VexFlow (ex. 6 = q + 8,
    20 = w + q) est rendue par plusieurs ``MusicalNoteSchema`` cons cutives
    avec le m me pitch — convention partag e avec le frontend.
    """
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


def quantize_to_notes(
    times_sec: np.ndarray,
    freqs_hz: np.ndarray,
    confidences: np.ndarray,
    rms: np.ndarray,
    duration_sec: float,
) -> List[MusicalNoteSchema]:
    """Helper de bout en bout : frames CREPE -> liste de ``MusicalNoteSchema``."""
    cells = quantize_cells(times_sec, freqs_hz, confidences, rms, duration_sec)
    runs = cells_to_runs(cells)
    return runs_to_notes(runs)
