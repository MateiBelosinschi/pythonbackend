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


def test_recleanup_replays_cleanup_with_new_strategy():
    # Three same-pitch onsets ~0.05s apart — under standard merge gap they
    # collapse to one note; with expert/split they stay as three.
    raw = [
        Note(pitch=60, start=0.0, end=0.20, velocity=80),
        Note(pitch=60, start=0.25, end=0.45, velocity=80),
        Note(pitch=60, start=0.50, end=0.70, velocity=80),
    ]
    payload = {
        "raw_notes": [n.model_dump() for n in raw],
        "options": {"preset": "standard"},
    }
    r = client.post("/api/recleanup", json=payload)
    assert r.status_code == 200
    merged = r.json()["notes"]
    assert len(merged) == 1

    payload["options"] = {"preset": "expert"}
    r = client.post("/api/recleanup", json=payload)
    split = r.json()["notes"]
    assert len(split) == 3


def test_recleanup_tie_strategy_flags_tied_to_next():
    raw = [
        Note(pitch=60, start=0.0, end=0.20, velocity=80),
        Note(pitch=60, start=0.25, end=0.45, velocity=80),
    ]
    payload = {
        "raw_notes": [n.model_dump() for n in raw],
        "options": {"preset": "standard", "repeat_strategy": "tie"},
    }
    r = client.post("/api/recleanup", json=payload)
    assert r.status_code == 200
    notes = r.json()["notes"]
    assert len(notes) == 2
    assert notes[0]["tied_to_next"] is True
    assert notes[1]["tied_to_next"] is False


def test_recleanup_rejects_unknown_preset():
    payload = {"raw_notes": [], "options": {"preset": "wizard"}}
    r = client.post("/api/recleanup", json=payload)
    assert r.status_code == 422
