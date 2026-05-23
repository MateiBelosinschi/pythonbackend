"""Wrapper autour de CREPE pour la d tection de hauteur.

CREPE charge son mod le Keras paresseusement  la premi re inf rence. Pour
 viter une premi re requ te tr s lente, on expose :func:`warmup` qui force
la construction du graphe d s le d marrage de l'app FastAPI.
"""
from __future__ import annotations

import logging
import threading
from typing import Tuple

import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


# CREPE n'est pas import  au top-level : ainsi les tests qui ne touchent pas
# au pitch peuvent tourner sans avoir TensorFlow install . On l'importe
# paresseusement et on garde la r f rence du module.
_crepe_module = None
_crepe_lock = threading.Lock()
_warmed_up = False


def _get_crepe():
    global _crepe_module
    if _crepe_module is None:
        with _crepe_lock:
            if _crepe_module is None:
                import crepe  # type: ignore
                _crepe_module = crepe
    return _crepe_module


def predict_pitch(
    samples: np.ndarray,
    sample_rate: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lance la d tection de hauteur CREPE sur le signal donn .

    Args:
        samples: signal mono float32 (idealement d j   ``settings.SAMPLE_RATE``).
        sample_rate: fr quence d' chantillonnage de ``samples``.

    Returns:
        ``(times, freqs_hz, confidences)``, trois arrays 1D de m me longueur.
        - ``times`` (s) : timestamp du centre de chaque frame.
        - ``freqs_hz`` : fr quence estim e en Hz (0 si non d tect ).
        - ``confidences`` : score de confiance dans [0, 1].
    """
    crepe = _get_crepe()
    # CREPE attend du float ; on s'assure du dtype pour  viter une copie interne.
    audio = np.asarray(samples, dtype=np.float32)
    time, frequency, confidence, _activation = crepe.predict(
        audio,
        sample_rate,
        model_capacity=settings.CREPE_MODEL,
        viterbi=settings.CREPE_VITERBI,
        step_size=settings.STEP_MS,
        verbose=0,
    )
    return (
        np.asarray(time, dtype=np.float64),
        np.asarray(frequency, dtype=np.float64),
        np.asarray(confidence, dtype=np.float64),
    )


def warmup() -> None:
    """Force le chargement du mod le CREPE pour viter la latence sur la 1re requ te.

    Inf re sur 1 seconde de silence. Idempotent.
    """
    global _warmed_up
    if _warmed_up:
        return
    with _crepe_lock:
        if _warmed_up:
            return
        try:
            sr = settings.SAMPLE_RATE
            dummy = np.zeros(sr, dtype=np.float32)
            predict_pitch(dummy, sr)
            _warmed_up = True
            logger.info("CREPE model '%s' warmed up", settings.CREPE_MODEL)
        except Exception:  # noqa: BLE001
            # En d veloppement on peut vouloir d marrer sans TF install ;
            # le warmup est best-effort.
            logger.exception("CREPE warmup failed (continuing without warmup)")
