"""
routes.py — FastAPI router.

Endpoints:
  GET  /health         → sanity check
  POST /transcribe     → audio → List[MusicalNoteSchema]  (concert pitch, no transposition)
  POST /export/midi    → List[MusicalNoteSchema] → .mid file download
"""
import logging
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, status
from fastapi.responses import StreamingResponse
import io

from app.api.models import TranscriptionResponse, MusicalNoteSchema, MidiExportRequest
from app.config import settings
from app.services.audio_service import AudioService
from app.services.pitch_detector import PitchDetector
from app.services.quantizer import quantize
from app.services.midi_export import notes_to_midi_bytes
from app.services.music_theory import snap_to_scale
from app.utils.helpers import hz_to_midi

logger = logging.getLogger(__name__)

router = APIRouter()
pitch_detector = PitchDetector()

# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@router.get("/health")
def health():
    """Sanity check — always returns 200 if the server is running."""
    return {
        "status": "success",
        "data": {"status": "ok", "app": settings.APP_TITLE, "version": settings.APP_VERSION},
        "error": None,
    }


# ---------------------------------------------------------------------------
# /transcribe
# ---------------------------------------------------------------------------

# Mock: "Au clair de la lune" — concert pitch, VexFlow durations
_MOCK_NOTES = [
    MusicalNoteSchema(pitch="C4", duration="q", isRest=False),
    MusicalNoteSchema(pitch="C4", duration="q", isRest=False),
    MusicalNoteSchema(pitch="C4", duration="q", isRest=False),
    MusicalNoteSchema(pitch="D4", duration="q", isRest=False),
    MusicalNoteSchema(pitch="E4", duration="h", isRest=False),
    MusicalNoteSchema(pitch="D4", duration="h", isRest=False),
    MusicalNoteSchema(pitch="C4", duration="q", isRest=False),
    MusicalNoteSchema(pitch="E4", duration="q", isRest=False),
    MusicalNoteSchema(pitch="D4", duration="q", isRest=False),
    MusicalNoteSchema(pitch="D4", duration="q", isRest=False),
    MusicalNoteSchema(pitch="C4", duration="w", isRest=False),
]


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, …)"),
    tempo: int = Form(120, description="Tempo in BPM (30-300)"),
    key: str = Form("C", description="Key signature for scale snapping (C, G, Am, …)"),
    mock: bool = Form(False, description="Return mock data instantly (useful for frontend dev)"),
):
    """
    Transcribe hummed/sung audio into a list of concert-pitch musical notes.

    Returns MusicalNoteSchema objects with:
      - pitch    : standard pitch string, e.g. "C4", "G#5"  (concert pitch, no transposition)
      - duration : VexFlow duration string "w" | "h" | "q" | "8" | "16"
      - isRest   : True when the frame was silence / low-confidence noise
    """
    # 1. Validate tempo bounds
    if not (settings.MIN_TEMPO <= tempo <= settings.MAX_TEMPO):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tempo {tempo} BPM is out of bounds. Must be between {settings.MIN_TEMPO} and {settings.MAX_TEMPO}.",
        )

    # 2. Mock path (for frontend integration before real audio is ready)
    if mock:
        logger.info("Returning mock transcription.")
        return TranscriptionResponse(status="success", data=_MOCK_NOTES, error=None)

    # 3. Read file bytes
    try:
        content = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Could not read uploaded file: {exc}")

    # 4. Persist to temp file (guaranteed cleanup via context manager)
    try:
        temp_path = AudioService.save_temporary_file(content, file.filename or "upload.wav")
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    # 5. Full pipeline inside the temp-file context manager
    try:
        with AudioService.manage_temp_file(temp_path):
            y, sr = AudioService.load_and_resample(temp_path)

            # Pitch detection → list of {"time", "frequency", "confidence"} at 10 ms
            frames = pitch_detector.detect_pitch(y, sr)
            if not frames:
                raise ValueError("No audio content detected in the uploaded file.")

            # Convert Hz → MIDI integers (0 = rest/silence)
            midi_frames = [hz_to_midi(f["frequency"]) for f in frames]

            # Snap pitches to the chosen key signature (concert pitch, no transposition)
            midi_frames = [snap_to_scale(m, key) for m in midi_frames]

            # NumPy quantization → MusicalNoteSchema-compatible dicts
            note_dicts = quantize(midi_frames, tempo)

            notes = [MusicalNoteSchema(**n) for n in note_dicts]
            return TranscriptionResponse(status="success", data=notes, error=None)

    except Exception as exc:
        logger.error(f"Transcription error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Transcription failed: {exc}",
        )


# ---------------------------------------------------------------------------
# /export/midi
# ---------------------------------------------------------------------------

@router.post("/export/midi")
def export_midi(payload: MidiExportRequest):
    """
    Convert a list of MusicalNoteSchema notes to a downloadable MIDI file.

    Accepts the same note objects returned by /transcribe.
    Returns a binary .mid file as an attachment.
    """
    if not (settings.MIN_TEMPO <= payload.tempo <= settings.MAX_TEMPO):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tempo {payload.tempo} BPM is out of bounds ({settings.MIN_TEMPO}-{settings.MAX_TEMPO}).",
        )

    try:
        note_dicts = [n.model_dump() for n in payload.notes]
        midi_bytes = notes_to_midi_bytes(note_dicts, tempo=payload.tempo)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        logger.error(f"MIDI export error: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"MIDI export failed: {exc}")

    filename = f"{payload.title or 'export'}.mid"
    return StreamingResponse(
        io.BytesIO(midi_bytes),
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
