"""POST /api/transcribe — audio bytes in, Concert Pitch JSON out.

Also POST /api/recleanup — replay the cleanup stage on cached raw notes
with a different CleanupOptions payload, no re-upload, no basic-pitch re-run.
"""

from __future__ import annotations

import gc
import json

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models import CleanupOptions, RecleanupRequest, TranscriptionResult
from app.services import dsp, melody_cleanup, monophonic, quantizer, transcriber

router = APIRouter()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB — keeps Railway free-tier RAM happy.


def _parse_options(raw: str | None) -> CleanupOptions:
    if not raw:
        return CleanupOptions()
    try:
        return CleanupOptions.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid cleanup options: {exc}") from exc


@router.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_endpoint(
    file: UploadFile = File(...),
    options: str | None = Form(default=None),
) -> TranscriptionResult:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio exceeds 25 MB limit.")

    cleanup_options = _parse_options(options)

    try:
        waveform, sr = dsp.preprocess(raw)
    except Exception as exc:  # librosa raises a variety of decode errors
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc

    midi = transcriber.transcribe(waveform, sr)
    quantized = quantizer.quantize(midi)
    melody = monophonic.collapse_to_melody(quantized)
    cleaned = melody_cleanup.cleanup(melody, cleanup_options)

    # Free heavy tensors before returning — Railway free tier is RAM-constrained.
    del waveform, midi, quantized
    gc.collect()

    return TranscriptionResult(notes=cleaned, raw_notes=melody)


@router.post("/recleanup", response_model=TranscriptionResult)
def recleanup_endpoint(payload: RecleanupRequest) -> TranscriptionResult:
    """Re-run the cleanup stage on cached raw notes with new options.

    The frontend caches ``raw_notes`` (from a prior /transcribe response) in
    IndexedDB and posts them back here when the user changes preset. Skips the
    heavy DSP + basic-pitch path entirely.
    """
    cleaned = melody_cleanup.cleanup(payload.raw_notes, payload.options)
    return TranscriptionResult(notes=cleaned, raw_notes=payload.raw_notes)
