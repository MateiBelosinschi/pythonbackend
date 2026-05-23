# CLAUDE.md

## Project
Backend API for humming-to-sheet-music transcription: users record audio to a click track → system detects pitch (CREPE) → quantizes to a strict 120 BPM grid via NumPy → returns Concert Pitch notes JSON. (Frontend handles VexFlow visual rendering and instant instrument transposition). Backend also provides endpoints for MIDI export.

## Stack
- **Language**: Python 3.9+
- **Framework**: FastAPI
- **Server**: Uvicorn
- **Pitch Detection**: Crepe (TensorFlow)
- **Audio Processing**: librosa, scipy, numpy
- **Export**: mido (MIDI)
- **Deployment**: Railway
- **Package Manager**: pip

## Commands
- Dev: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- Build: `docker build -t melody-scribe-api .`
- Test single: `pytest tests/test_export.py -v`
- Test all: `pytest`
- Lint: `flake8 app/ --max-line-length=100`
- Type check: `mypy app/`

## Architecture
- `app/api/` → API endpoint definitions, request/response models (Pydantic)
- `app/services/` → Core business logic (pitch detection, 120 BPM numpy quantization, MIDI export)
- `app/utils/` → Helper functions, constants (MIDI mappings, 120 BPM duration buckets, note names)
- `app/config.py` → Configuration management (environment variables, API settings)
- `main.py` → FastAPI app initialization, middleware, route mounting
- `requirements.txt` → Python dependencies
- `Dockerfile` → Docker container setup (used by Railway)
- `tests/` → Unit tests for each service

## Data Contracts

### 1. Backend API Response Model (Pydantic)
from pydantic import BaseModel
from typing import List, Optional

class MusicalNoteSchema(BaseModel):
    pitch: str         # e.g., "C4", "G#5"
    duration: str      # VexFlow style: "w", "h", "q", "8", "16"
    isRest: bool       # True if this chunk represents silence/noise

class TranscriptionResponse(BaseModel):
    status: str        # "success" | "error"
    data: Optional[List[MusicalNoteSchema]] = None
    error: Optional[str] = None

### 2. Frontend State / Response Types (TypeScript)

export interface MusicalNote {
  pitch: string;         // Standard pitch e.g., "C4", "Bb5"
  duration: string;      // VexFlow format: "w", "h", "q", "8", "16"
  isRest: boolean;       // True if this block was silence/noise
}

export interface TranscriptionResponse {
  status: "success" | "error";
  data: MusicalNote[] | null;
  error: string | null;
}