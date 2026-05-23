"""Configuration centrale du backend de transcription.

Toutes les constantes "musicales" et les seuils du pipeline vivent ici.
Les valeurs surchargables via variables d'environnement sont lues au chargement
du module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw else default


@dataclass(frozen=True)
class Settings:
    # Tempo et grille rythmique
    BPM: int = 120
    # Plus petite subdivision exprim e dans le vocabulaire VexFlow accept .
    # 4 => 16e de note (4 par temps).
    SUBDIVISIONS_PER_BEAT: int = 4

    # Audio
    SAMPLE_RATE: int = 16000  # CREPE est entra n  sur 16 kHz
    STEP_MS: int = 10         # Hop de CREPE et du RMS (frames toutes les 10 ms)

    # Plage vocale autoris e (MIDI). Hors plage => isRest.
    MIDI_MIN: int = 36  # C2
    MIDI_MAX: int = 84  # C6

    # Seuils de d tection "voix pr sente" par frame
    CONFIDENCE_THRESHOLD: float = 0.5
    # RMS normalis  (audio en float32 [-1, 1]). 0.01 ~ -40 dBFS.
    RMS_THRESHOLD: float = 0.01

    # Mod le CREPE: "tiny" | "small" | "medium" | "large" | "full".
    # tiny est largement suffisant pour de la voix humm e sur Railway CPU.
    CREPE_MODEL: str = "tiny"

    # Activer le lissage Viterbi de CREPE (plus pr cis, ~2x plus lent).
    CREPE_VITERBI: bool = True

    # Taille maximum de fichier audio accept e (octets). 25 Mo par d faut.
    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024

    # CORS
    CORS_ORIGINS: tuple[str, ...] = ("*",)

    @property
    def cell_seconds(self) -> float:
        """Dur e d'une cellule de la grille rythmique en secondes."""
        return 60.0 / self.BPM / self.SUBDIVISIONS_PER_BEAT

    @property
    def step_seconds(self) -> float:
        return self.STEP_MS / 1000.0


def _load_settings() -> Settings:
    return Settings(
        BPM=_env_int("BPM", 120),
        SUBDIVISIONS_PER_BEAT=_env_int("SUBDIVISIONS_PER_BEAT", 4),
        SAMPLE_RATE=_env_int("SAMPLE_RATE", 16000),
        STEP_MS=_env_int("STEP_MS", 10),
        MIDI_MIN=_env_int("MIDI_MIN", 36),
        MIDI_MAX=_env_int("MIDI_MAX", 84),
        CONFIDENCE_THRESHOLD=_env_float("CONFIDENCE_THRESHOLD", 0.5),
        RMS_THRESHOLD=_env_float("RMS_THRESHOLD", 0.01),
        CREPE_MODEL=_env_str("CREPE_MODEL", "tiny"),
        CREPE_VITERBI=_env_str("CREPE_VITERBI", "1") not in ("0", "false", "False"),
        MAX_UPLOAD_BYTES=_env_int("MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
    )


settings = _load_settings()
