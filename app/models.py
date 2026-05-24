"""Pydantic schemas for the Concert Pitch JSON contract exchanged with the frontend."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


FIXED_BPM: int = 120
GRID_SUBDIVISION: int = 16  # 16th-note grid


class Note(BaseModel):
    """A single quantized note in Concert Pitch (MIDI pitch number)."""

    pitch: int = Field(..., ge=0, le=127, description="MIDI note number, 0-127.")
    start: float = Field(..., ge=0.0, description="Onset in seconds, snapped to the 16th-note grid.")
    end: float = Field(..., gt=0.0, description="Offset in seconds, snapped to the 16th-note grid.")
    velocity: int = Field(80, ge=1, le=127)


class TranscriptionResult(BaseModel):
    """Output of /api/transcribe.

    Two parallel views of the same hum:

    - `notes` — quantized, monophonic, key-snapped. Drives the sheet music
      (VexFlow) and the rhythmic grid.
    - `raw_notes` — direct basic-pitch output, original timings preserved,
      possibly polyphonic. Drives the piano playback so it sounds like
      Spotify's basic-pitch demo.

    Frontend renders `notes` on the staff and plays `raw_notes` on the piano.
    """

    bpm: int = Field(FIXED_BPM, description="Always 120 — multi-tempo tracking is out of scope.")
    subdivision: int = Field(GRID_SUBDIVISION, description="Grid subdivision (16 = sixteenth notes).")
    time_signature: Literal["4/4"] = "4/4"
    notes: List[Note] = Field(default_factory=list)
    raw_notes: List[Note] = Field(default_factory=list)


class ExportRequest(BaseModel):
    """Input for /api/export-midi — accepts the same shape returned by /api/transcribe."""

    notes: List[Note]
    bpm: int = FIXED_BPM
