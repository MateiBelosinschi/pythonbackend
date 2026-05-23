"""Point d'entr e FastAPI du backend de transcription humming -> notes."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.export import router as export_router
from app.api.transcribe import router as transcribe_router
from app.config import settings

# TensorFlow / CREPE sont tr s verbeux par d faut ; on calme les logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Skipped warmup to prevent Railway timeout/OOM during startup
    yield


app = FastAPI(
    title="melody-scribe-api",
    version="0.1.0",
    description=(
        "Backend de transcription audio fredonn  -> notes quantif es 120 BPM. "
        "Pipeline: CREPE (pitch detection) -> quantification numpy -> JSON."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(transcribe_router, prefix="/api")
app.include_router(export_router, prefix="/api")
