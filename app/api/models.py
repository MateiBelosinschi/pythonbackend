from pydantic import BaseModel, Field
from typing import List, Optional


class MusicalNoteSchema(BaseModel):
    pitch: str          # e.g., "C4", "G#5"
    duration: str       # VexFlow style: "w", "h", "q", "8", "16"
    isRest: bool        # True if this chunk represents silence/noise


class TranscriptionResponse(BaseModel):
    status: str                                   # "success" | "error"
    data: Optional[List[MusicalNoteSchema]] = None
    error: Optional[str] = None


class MidiExportRequest(BaseModel):
    notes: List[MusicalNoteSchema] = Field(..., description="List of notes to export as MIDI")
    tempo: int = Field(120, description="Tempo in BPM (30-300)")
    title: Optional[str] = Field("export", description="Filename stem for the MIDI file")
