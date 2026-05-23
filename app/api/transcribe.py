"""Endpoint POST /api/transcribe : audio multipart -> JSON ``TranscriptionResponse``."""
from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, Query, UploadFile, status
from fastapi.responses import JSONResponse

from app.api.schemas import TranscriptionResponse
from app.config import settings
from app.services.audio_loader import AudioDecodeError, compute_rms_frames, load_audio
from app.services.pitch_detection import predict_pitch
from app.services.quantizer import QuantizeParams, quantize_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(tags=["transcription"])


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcrit une prise audio fredonnée en notes quantifiées",
)
async def transcribe(
    audio: UploadFile = File(...),
    bpm: int = Form(default=settings.BPM, description="Tempo de la grille en BPM."),
    grid_offset_sec: float = Form(
        default=0.0,
        description="Décalage (s) entre le début du fichier et le temps 1 du métronome.",
    ),
    debug: bool = Query(
        default=False,
        description="Inclure crepe_track, cells et grid dans la réponse.",
    ),
) -> JSONResponse:
    """Reçoit un blob audio (n'importe quel format supporté par ffmpeg) et
    renvoie le tableau de ``MusicalNote`` correspondant.

    Le client doit envoyer le fichier en ``multipart/form-data`` sous le champ
    ``audio``. Les champs ``bpm`` et ``grid_offset_sec`` sont optionnels.
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

    if bpm <= 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=TranscriptionResponse(
                status="error",
                error=f"Invalid bpm: {bpm} (must be > 0)",
            ).model_dump(),
        )

    try:
        samples, sr = load_audio(raw)
    except AudioDecodeError as exc:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=TranscriptionResponse(status="error", error=str(exc)).model_dump(),
        )

    params = QuantizeParams(bpm=bpm, grid_offset_sec=grid_offset_sec)

    try:
        times, freqs, confidences = predict_pitch(samples, sr)
        rms = compute_rms_frames(samples, sr)
        duration_sec = len(samples) / float(sr)
        result = quantize_pipeline(
            times,
            freqs,
            confidences,
            rms,
            duration_sec,
            params=params,
            debug=debug,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Transcription pipeline failed")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=TranscriptionResponse(
                status="error",
                error=f"Transcription failed: {exc}",
            ).model_dump(),
        )

    payload = TranscriptionResponse(
        status="success",
        data=result.notes,
        debug=result.debug,
    )
    return JSONResponse(content=payload.model_dump())
