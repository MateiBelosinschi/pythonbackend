"""Chargement et pr -traitement de l'audio entrant.

L' tape consiste  :
1. D coder n'importe quel format support  par `audioread`/`ffmpeg`
   (WAV, WebM/Opus depuis MediaRecorder, MP3, MP4/AAC, ...).
2. Convertir en mono.
3. Resampler  ``settings.SAMPLE_RATE`` (16 kHz par d faut, requis par CREPE).
4. Calculer une enveloppe RMS sur la m me grille temporelle (10 ms) que CREPE,
   ce qui permet ensuite de fusionner les masques "voix pr sente".
"""
from __future__ import annotations

import io
from typing import Tuple

import librosa
import numpy as np

from app.config import settings


class AudioDecodeError(ValueError):
    """L ve  e quand l'audio re u ne peut pas  tre d cod ."""


def load_audio(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """D code un blob audio arbitraire en mono float32 16 kHz.

    Args:
        audio_bytes: contenu binaire brut tel qu'envoy  par le frontend.

    Returns:
        (samples, sample_rate) o  samples est un ``np.ndarray`` mono float32
        dans [-1, 1], et sample_rate vaut ``settings.SAMPLE_RATE``.

    Raises:
        AudioDecodeError: si le d codage  choue ou si l'audio est vide.
    """
    if not audio_bytes:
        raise AudioDecodeError("Empty audio payload")

    import os
    import tempfile

    # On crit les octets dans un fichier temporaire car librosa (via audioread/ffmpeg)
    # a besoin d'un chemin de fichier r el pour d coder les formats compress s
    # comme WebM ou MP3 envoy s par les navigateurs.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    try:
        samples, sr = librosa.load(
            temp_path,
            sr=settings.SAMPLE_RATE,
            mono=True,
            dtype=np.float32,
        )
    except Exception as exc:  # noqa: BLE001 — on enveloppe en erreur m tier
        raise AudioDecodeError(f"Failed to decode audio: {exc}") from exc
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass

    if samples.size == 0:
        raise AudioDecodeError("Decoded audio is empty")

    return samples, int(sr)


def compute_rms_frames(
    samples: np.ndarray,
    sample_rate: int,
    step_ms: int | None = None,
) -> np.ndarray:
    """Calcule l'enveloppe RMS frame par frame, sur une grille r guli re.

    On utilise une fen tre = hop = ``step_ms`` (10 ms par d faut). Le r sultat
    contient une valeur par hop, ce qui doit s'aligner avec la grille de CREPE
    qui utilise aussi ``step_size=10 ms``.

    Args:
        samples: signal mono float32.
        sample_rate: fr quence d' chantillonnage de ``samples``.
        step_ms: pas en millisecondes (par d faut ``settings.STEP_MS``).

    Returns:
        ``np.ndarray`` 1D float32, longueur ~ ``len(samples) * step_ms / 1000``.
    """
    step = step_ms if step_ms is not None else settings.STEP_MS
    hop_length = max(1, int(round(sample_rate * step / 1000.0)))

    # On utilise frame_length == hop_length pour des fen tres non recouvrantes,
    # une RMS frame correspond donc  exactement step_ms d'audio.
    rms = librosa.feature.rms(
        y=samples,
        frame_length=hop_length,
        hop_length=hop_length,
        center=True,
    )[0]
    return rms.astype(np.float32, copy=False)
