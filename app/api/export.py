"""Endpoint POST /api/export/midi : ``MusicalNote[]`` -> fichier ``.mid``."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response

from app.api.schemas import MidiExportRequest
from app.services.midi_export import notes_to_midi_bytes

logger = logging.getLogger(__name__)
router = APIRouter(tags=["export"])


@router.post(
    "/export/midi",
    summary="Exporte une liste de notes en fichier MIDI",
    responses={200: {"content": {"audio/midi": {}}}},
)
async def export_midi(payload: MidiExportRequest) -> Response:
    """Endpoint stateless : on re oit le tableau de notes,
    on renvoie les octets MIDI dans la r ponse.
    """
    try:
        midi_bytes = notes_to_midi_bytes(payload.notes)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("MIDI export failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MIDI export failed: {exc}",
        ) from exc

    return Response(
        content=midi_bytes,
        media_type="audio/midi",
        headers={"Content-Disposition": 'attachment; filename="transcription.mid"'},
    )
