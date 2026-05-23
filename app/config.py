import os

class Settings:
    # App Settings
    APP_TITLE: str = os.getenv("MELODY_SCRIBE_APP_TITLE", "Melody Scribe API")
    APP_VERSION: str = os.getenv("MELODY_SCRIBE_APP_VERSION", "1.0.0")
    
    # Audio Processing Settings
    SAMPLE_RATE: int = int(os.getenv("MELODY_SCRIBE_SAMPLE_RATE", "16000"))  # 16kHz recommended for CREPE
    CONFIDENCE_THRESHOLD: float = float(os.getenv("MELODY_SCRIBE_CONFIDENCE_THRESHOLD", "0.5"))
    
    # Music Theory Settings
    DEFAULT_TEMPO: int = int(os.getenv("MELODY_SCRIBE_DEFAULT_TEMPO", "120"))
    MIN_TEMPO: int = 30
    MAX_TEMPO: int = 300
    DEFAULT_KEY: str = os.getenv("MELODY_SCRIBE_DEFAULT_KEY", "C")
    
    # Storage
    TEMP_DIR: str = os.getenv(
        "MELODY_SCRIBE_TEMP_DIR", 
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "temp_audio")
    )

settings = Settings()

# Ensure temp directory exists
os.makedirs(settings.TEMP_DIR, exist_ok=True)
