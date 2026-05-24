"""Lightweight FastAPI route smoke tests (no model inference)."""

import pretty_midi
from fastapi.testclient import TestClient

from app.main import app
from app.models import Note
from app.services.quantizer import notes_to_midi

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_export_midi_returns_valid_midi():
    notes = [
        Note(pitch=60, start=0.0, end=0.5, velocity=80),
        Note(pitch=64, start=0.5, end=1.0, velocity=80),
    ]
    r = client.post("/api/export-midi", json={"notes": [n.model_dump() for n in notes], "bpm": 120})
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/midi"
    assert r.content[:4] == b"MThd"  # MIDI file magic header


def test_export_midi_round_trip_matches_notes():
    notes = [Note(pitch=72, start=0.0, end=0.25, velocity=90)]
    midi_bytes = notes_to_midi(notes)
    import io
    pm = pretty_midi.PrettyMIDI(io.BytesIO(midi_bytes))
    decoded = [n for inst in pm.instruments for n in inst.notes]
    assert len(decoded) == 1
    assert decoded[0].pitch == 72


def test_transcribe_rejects_empty_upload():
    r = client.post("/api/transcribe", files={"file": ("empty.wav", b"", "audio/wav")})
    assert r.status_code == 400
