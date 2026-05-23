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


class TranscriptionResponse(BaseModel):
    status: Literal["success", "error"]
    data: Optional[List[MusicalNoteSchema]] = None
    error: Optional[str] = None


class MidiExportRequest(BaseModel):
    notes: List[MusicalNoteSchema] = Field(
        ...,
        description="Liste ordonn e de notes/silences  exporter en MIDI.",
    )
