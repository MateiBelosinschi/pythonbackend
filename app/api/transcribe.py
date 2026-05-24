"""POST /api/transcribe — audio bytes in, Concert Pitch JSON out."""

from __future__ import annotations

import gc

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models import TranscriptionResult
from app.services import dsp, melody_cleanup, monophonic, quantizer, transcriber

router = APIRouter()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB — keeps Railway free-tier RAM happy.


@router.post("/transcribe", response_model=TranscriptionResult)
async def transcribe_endpoint(file: UploadFile = File(...)) -> TranscriptionResult:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio exceeds 25 MB limit.")

    try:
        waveform, sr = dsp.preprocess(raw)
    except Exception as exc:  # librosa raises a variety of decode errors
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}") from exc

    midi = transcriber.transcribe(waveform, sr)

    # Two views of the same transcription:
    # - raw_notes: original basic-pitch output, drives piano playback so it
    #   sounds like Spotify's hosted demo (the user's reference quality).
    # - notes: quantized/monophonic/key-snapped, drives the sheet music.
    raw_notes = transcriber.pretty_midi_to_notes(midi)
    quantized = quantizer.quantize(midi)
    melody = monophonic.collapse_to_melody(quantized)
    cleaned = melody_cleanup.cleanup(melody)

    # Free heavy tensors before returning — Railway free tier is RAM-constrained.
    del waveform, midi, quantized, melody
    gc.collect()

    return TranscriptionResult(notes=cleaned, raw_notes=raw_notes)
