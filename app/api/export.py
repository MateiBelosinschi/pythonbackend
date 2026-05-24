"""POST /api/export-midi — Concert Pitch JSON in, MIDI file bytes out."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from app.models import ExportRequest
from app.services.quantizer import notes_to_midi

router = APIRouter()


@router.post("/export-midi")
def export_midi(payload: ExportRequest) -> Response:
    data = notes_to_midi(payload.notes, bpm=payload.bpm)
    return Response(
        content=data,
        media_type="audio/midi",
        headers={"Content-Disposition": 'attachment; filename="transcription.mid"'},
    )
