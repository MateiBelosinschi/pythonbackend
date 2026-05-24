"""FastAPI entry point: CORS, routing, health check."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import export, transcribe

app = FastAPI(
    title="musicMe transcription API",
    description="Humming-to-sheet-music transcription pinned at 120 BPM.",
    version="0.1.0",
)

_allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(transcribe.router, prefix="/api", tags=["transcription"])
app.include_router(export.router, prefix="/api", tags=["export"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
