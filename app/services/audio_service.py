import os
import shutil
import uuid
import contextlib
import logging
from typing import Generator
import librosa
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)

class AudioService:
    @staticmethod
    def save_temporary_file(file_content: bytes, filename: str) -> str:
        """
        Saves binary audio data to a temporary file with a unique name.
        """
        ext = os.path.splitext(filename)[1]
        # Keep extension but use UUID to avoid collisions
        temp_filename = f"{uuid.uuid4()}{ext}"
        temp_path = os.path.join(settings.TEMP_DIR, temp_filename)
        
        try:
            with open(temp_path, "wb") as buffer:
                buffer.write(file_content)
            logger.info(f"Saved temporary file to {temp_path}")
            return temp_path
        except Exception as e:
            logger.error(f"Failed to save temporary file: {e}")
            raise RuntimeError(f"Could not save uploaded audio file: {e}")

    @staticmethod
    @contextlib.contextmanager
    def manage_temp_file(file_path: str) -> Generator[str, None, None]:
        """
        Context manager to ensure a temporary file is cleaned up after processing.
        """
        try:
            yield file_path
        finally:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Successfully deleted temporary file: {file_path}")
                except Exception as e:
                    logger.error(f"Error deleting temporary file {file_path}: {e}")

    @staticmethod
    def load_and_resample(file_path: str) -> tuple[np.ndarray, int]:
        """
        Loads an audio file and resamples it to the configured target sample rate (mono).
        Returns a tuple of (audio_time_series, sample_rate).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        try:
            # Load as mono and resample to settings.SAMPLE_RATE (usually 16000 Hz)
            y, sr = librosa.load(file_path, sr=settings.SAMPLE_RATE, mono=True)
            return y, sr
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise ValueError(f"Invalid or unsupported audio format: {e}")
