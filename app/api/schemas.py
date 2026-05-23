"""Mod les Pydantic — strict respect du data contract du `CLAUDE.md`."""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


VexFlowDuration = Literal["w", "h", "q", "8", "16"]


class MusicalNoteSchema(BaseModel):
    pitch: str = Field(
        ...,
        description='Hauteur de note en notation standard (di ses uniquement), p.ex. "C4", "G#5". '
                    'Pour un silence, ce champ vaut conventionnellement "rest".',
    )
    duration: VexFlowDuration = Field(
        ...,
        description='Dur e style VexFlow: "w" (ronde), "h" (blanche), "q" (noire), '
                    '"8" (croche), "16" (double-croche).',
    )
    isRest: bool = Field(
        ...,
        description="True si ce bloc repr sente un silence/bruit.",
    )


class CrepeFrameDebug(BaseModel):
    time: float = Field(..., description="Timestamp du frame CREPE en secondes.")
    freq_hz: float = Field(..., description="Fréquence estimée en Hz (0 si non voisée).")
    confidence: float = Field(..., description="Confiance CREPE dans [0, 1].")
    midi_rounded: Optional[int] = Field(
        None,
        description="Numéro MIDI arrondi au demi-ton, ou null si non voisée.",
    )


class GridMetadata(BaseModel):
    bpm: int = Field(..., description="Tempo utilisé pour la grille.")
    cell_seconds: float = Field(..., description="Durée d'une cellule (16e de note) en secondes.")
    offset_sec: float = Field(
        ...,
        description="Décalage appliqué entre t=0 du fichier et le temps 1 du métronome.",
    )


class TranscriptionDebugInfo(BaseModel):
    crepe_track: List[CrepeFrameDebug] = Field(
        ...,
        description="Piste CREPE frame par frame pour diagnostic.",
    )
    cells: List[Optional[int]] = Field(
        ...,
        description="Pitch MIDI par cellule de grille (null = silence).",
    )
    grid: GridMetadata = Field(..., description="Paramètres de la grille rythmique.")


class TranscriptionResponse(BaseModel):
    status: Literal["success", "error"]
    data: Optional[List[MusicalNoteSchema]] = None
    error: Optional[str] = None
    debug: Optional[TranscriptionDebugInfo] = None


class MidiExportRequest(BaseModel):
    notes: List[MusicalNoteSchema] = Field(
        ...,
        description="Liste ordonn e de notes/silences  exporter en MIDI.",
    )
