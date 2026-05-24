"""Pydantic schemas for the Concert Pitch JSON contract exchanged with the frontend."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


FIXED_BPM: int = 120
GRID_SUBDIVISION: int = 16  # 16th-note grid


RepeatStrategy = Literal["merge", "tie", "split"]
CleanupPreset = Literal["beginner", "standard", "expert"]


class Note(BaseModel):
    """A single quantized note in Concert Pitch (MIDI pitch number)."""

    pitch: int = Field(..., ge=0, le=127, description="MIDI note number, 0-127.")
    start: float = Field(..., ge=0.0, description="Onset in seconds, snapped to the 16th-note grid.")
    end: float = Field(..., gt=0.0, description="Offset in seconds, snapped to the 16th-note grid.")
    velocity: int = Field(80, ge=1, le=127)
    tied_to_next: bool = Field(
        False,
        description="If True, render a tie (VexFlow liaison) into the following note in sequence.",
    )


class CleanupOptions(BaseModel):
    """Knobs that change how consecutive same-pitch notes are grouped.

    A preset picks sensible defaults; explicit fields override the preset.
    """

    preset: CleanupPreset = "standard"
    repeat_strategy: RepeatStrategy | None = Field(
        None,
        description="merge=one held note, tie=two notes joined by a liaison, split=leave as-is.",
    )
    max_gap: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Max gap (s) between two same-pitch notes for them to be grouped.",
    )

    def resolved(self) -> tuple[RepeatStrategy, float]:
        """Return (strategy, max_gap) with preset defaults filled in."""
        preset_defaults: dict[CleanupPreset, tuple[RepeatStrategy, float]] = {
            "beginner": ("merge", 0.30),
            "standard": ("merge", 0.15),
            "expert": ("split", 0.0),
        }
        default_strategy, default_gap = preset_defaults[self.preset]
        return (
            self.repeat_strategy or default_strategy,
            self.max_gap if self.max_gap is not None else default_gap,
        )


class TranscriptionResult(BaseModel):
    """Output of /api/transcribe."""

    bpm: int = Field(FIXED_BPM, description="Always 120 — multi-tempo tracking is out of scope.")
    subdivision: int = Field(GRID_SUBDIVISION, description="Grid subdivision (16 = sixteenth notes).")
    time_signature: Literal["4/4"] = "4/4"
    notes: List[Note] = Field(default_factory=list)
    raw_notes: List[Note] = Field(
        default_factory=list,
        description="Notes after quantize+monophonic but BEFORE cleanup. Cache client-side to re-process via /api/recleanup without re-uploading audio.",
    )


class RecleanupRequest(BaseModel):
    """Input for /api/recleanup — replay cleanup on cached raw notes with different options."""

    raw_notes: List[Note]
    options: CleanupOptions = Field(default_factory=CleanupOptions)


class ExportRequest(BaseModel):
    """Input for /api/export-midi — accepts the same shape returned by /api/transcribe."""

    notes: List[Note]
    bpm: int = FIXED_BPM
