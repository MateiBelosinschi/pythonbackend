"""Endpoint POST /api/transcribe : audio multipart -> JSON ``TranscriptionResponse``."""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.schemas import TranscriptionResponse
from app.config import settings
from app.services.audio_loader import AudioDecodeError, compute_rms_frames, load_audio
from app.services.pitch_detection import predict_pitch
from app.services.quantizer import quantize_to_notes

logger = logging.getLogger(__name__)
router = APIRouter(tags=["transcription"])


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcrit une prise audio fredonn e en notes quantif es 120 BPM",
)
async def transcribe(audio: UploadFile = File(...)) -> JSONResponse:
    """Re oit un blob audio (n'importe quel format support  par ffmpeg) et
    renvoie le tableau de ``MusicalNote`` correspondant.

    Le client doit envoyer le fichier en ``multipart/form-data`` sous le champ
    ``audio``.
    """
    try:
        raw = await audio.read()
    finally:
        await audio.close()

    if len(raw) > settings.MAX_UPLOAD_BYTES:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=TranscriptionResponse(
                status="error",
                error=(
                    f"Audio file too large: {len(raw)} bytes "
                    f"(max {settings.MAX_UPLOAD_BYTES})"
                ),
            ).model_dump(),
        )

    try:
        samples, sr = load_audio(raw)
    except AudioDecodeError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=TranscriptionResponse(status="error", error=str(exc)).model_dump(),
        )

    try:
        times, freqs, confidences = predict_pitch(samples, sr)
        rms = compute_rms_frames(samples, sr)
        duration_sec = len(samples) / float(sr)
        notes = quantize_to_notes(times, freqs, confidences, rms, duration_sec)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription pipeline failed")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=TranscriptionResponse(
                status="error",
                error=f"Transcription failed: {exc}",
            ).model_dump(),
        )

    payload = TranscriptionResponse(status="success", data=notes)
    return JSONResponse(content=payload.model_dump())
