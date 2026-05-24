# CLAUDE.md

## Project
Real-time audio humming-to-sheet-music transcription platform featuring strict 120 BPM quantization, VexFlow compatibility, and MIDI export.

## Stack  
- **Backend:** Python (FastAPI, basic-pitch, pretty_midi/music21, librosa)
- **Frontend:** TypeScript, React, Next.js, VexFlow, Web Audio API
- **Database:** None (Stateless, memory-efficient streaming/file ingestion)
- **Deployment Target:** Railway (Backend - Free Tier), Vercel (Frontend - Free Tier)

## Commands
- Dev (Backend): `uvicorn app.main:app --reload --port 8000`
- Dev (Frontend): `npm run dev`
- Build (Frontend): `npm run build`
- Test single: `pytest -- -v tests/test_transcription.py`
- Test all: `pytest`
- Lint: `flake8 app/ && npm run lint`
- Type check: `mypy app/ && npm run type-check`

## Architecture
- `backend/app/main.py` → FastAPI entry point, routing, CORS, and global middleware configuration.
- `backend/app/services/dsp.py` → Audio pre-processing layer using librosa (handles sample rate normalization and noise-gating to strip silent noise).
- `backend/app/services/transcriber.py` → Inference engine utilizing Spotify's `basic-pitch` to extract structured MIDI notes.
- `backend/app/services/quantizer.py` → Time-alignment grid execution using `pretty_midi` to force-snap notes to a rigid 120 BPM 16th-note grid.
- `backend/app/api/` → Endpoints for audio upload (`/api/transcribe`) and MIDI compilation (`/api/export-midi`).
- `frontend/src/hooks/useAudioRecorder.ts` → React hook managing user Web Audio streams, matching recording constraints with a visual or auditory 120 BPM click track.
- `frontend/src/components/SheetMusicRenderer.tsx` → VexFlow visualization wrapper accepting structured Concert Pitch JSON from the backend.

## Rules
- **NEVER use continuous frame-by-frame pitch metrics (like CREPE/Hz tracking) for note division.** You must use discrete MIDI events extracted via `basic-pitch` or event-based thresholding to avoid generation of spurious note fragments.
- **NEVER implement manual multi-dimensional math arrays for timing grid snapshots.** Time-quantization and offset boundaries must always be delegated to symbolic music libraries (`pretty_midi` or `music21`) to prevent floating-point rounding drifts.
- **DO NOT bypass the audio pre-processing phase.** Every raw byte block from the client must be clamped down using a hard dB noise gate before passing it to inference models to eliminate ambient hum and breath-induced notes during pauses.
- **OPTIMIZE FOR LOW RAM COMPUTE.** Because we are running on Railway's free tier, you must clear unmanaged memory objects after inference runs. Minimize reliance on heavy external neural network dependencies outside `basic-pitch` and process audio in short bursts.
- **IMPORTANT:** AI agents persistently default to building custom matrix structures via NumPy or vanilla lists to handle rhythm subdivisions. **STOP.** Do not calculate grid timings by dividing arrays. Use `pretty_midi` time-snapping and rhythm division primitives exclusively.

## Workflow
- **Incremental Architecture Execution:** Build and rigorously verify the isolated backend execution path (`DSP -> Basic-Pitch -> Quantizer`) using mock synthetic audio files before attempting frontend assembly or UI linking.
- **Commit Conventions:** Follow strict conventional commits: `feat(backend):`, `fix(quantization):`, `feat(frontend):`, `test:`.
- **Testing Expectations:** Every single DSP pipeline refactor requires validation against a standard synthetic audio file featuring clear silences to prove that zero "ghost notes" are introduced.
- **When to ask vs when to act:** Act autonomously when mapping static schemas or scaffolding boilerplate REST hooks. Stop and ask for explicit direction if the default `basic-pitch` classification thresholds require deep heuristic adjustment for complex user vocals.

## Out of scope
- **Multi-tempo tracking:** The entire application is statically optimized for a hardcoded 120 BPM execution layout. Do not write variable tempo tracking parameters.
- **On-server Audio Transposition:** The backend strictly outputs Standard Concert Pitch JSON. Instrumental transpositions (e.g., Bb Clarinet, Eb Alto Sax alterations) must remain entirely client-side tasks within the frontend VexFlow layer.
- **User Authentication or Storage:** The pipeline is intentionally stateless. All session memory and state management reside purely in frontend React memory structures to stay within resource limitations.