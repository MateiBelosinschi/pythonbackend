# CLAUDE.md

## Project
Backend API for humming-to-sheet-music transcription: users upload audio → system detects pitch → transcribes to notes → transposes for different instruments → exports to MIDI/MusicXML.

## Stack
- **Language**: Python 3.9+
- **Framework**: FastAPI
- **Server**: Uvicorn
- **Pitch Detection**: Crepe (TensorFlow)
- **Music Theory**: Music21 (MIT)
- **Audio Processing**: librosa, scipy
- **Export**: music21 (MusicXML), mido (MIDI)
- **Deployment**: Docker, Heroku/AWS Lambda
- **Package Manager**: pip

## Commands
- Dev: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`
- Build: `docker build -t melody-scribe-api .`
- Test single: `pytest tests/test_transposer.py -v`
- Test all: `pytest`
- Lint: `flake8 app/ --max-line-length=100`
- Type check: `mypy app/`

## Architecture
- `app/api/` → API endpoint definitions, request/response models (Pydantic)
- `app/services/` → Core business logic (pitch detection, quantization, transposition, export)
- `app/utils/` → Helper functions, constants (instrument definitions, MIDI mappings, note names)
- `app/config.py` → Configuration management (environment variables, API settings)
- `main.py` → FastAPI app initialization, middleware, route mounting
- `requirements.txt` → Python dependencies
- `Dockerfile` → Docker container setup
- `tests/` → Unit tests for each service

## Rules
- **Music Theory Accuracy**: Always validate MIDI numbers are in valid range (0-127). Never allow out-of-bounds MIDI values to be exported.
- **Instrument Range Checking**: Before transposing, verify that transposed notes fit within the instrument's documented range. Warn but don't fail—still export if out of range.
- **Floating Point Precision**: Round all frequency-to-MIDI conversions to nearest integer semitone. Never export unquantized decimal pitches.
- **File Handling**: Always clean up temporary audio files after processing. Use `try/finally` blocks to ensure temp files are deleted even if processing fails.
- **API Response Format**: Every response must include `status`, `data`, and `error` fields (even if error is null). Maintain consistent JSON structure across all endpoints.
- **Tempo Bounds**: Validate tempo is between 30-300 BPM. Reject requests outside this range with a clear error message.
- **IMPORTANT**: When exporting to MIDI, always verify duration_beats is positive and non-zero. A zero-duration note will cause the MIDI file to be malformed and unplayable.

## Workflow
- **Approach**: When adding a feature, start with the service (app/services/), then add the API route (app/api/routes.py), then add tests.
- **Commit Conventions**: `feat: add X`, `fix: resolve X`, `docs: update X`, `test: add tests for X`, `refactor: simplify X`
- **Testing Expectations**: Every service class method should have at least one unit test. Happy path + error case minimum. Test with realistic audio values (frequencies 60-2000 Hz).
- **When to Ask vs Act**: Ask before modifying API contracts (routes, request/response models). Act on service improvements, test additions, or bug fixes. Ask before adding new external dependencies.

## Out of scope
- Frontend code (React, Vue, HTML/CSS)
- Tone.js or any browser-side audio playback
- Vexflow or sheet music rendering (that's frontend)
- Deployment infrastructure (DevOps/terraform)
- User authentication or database models
- Production monitoring/logging setup (basic logging is fine)
- Modifying `Dockerfile` without Docker knowledge
- AI model retraining or fine-tuning Crepe