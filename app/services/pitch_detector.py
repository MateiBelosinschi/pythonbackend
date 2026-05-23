import numpy as np
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class PitchDetector:
    def __init__(self):
        self.has_crepe = False
        try:
            import crepe
            import tensorflow
            self.has_crepe = True
            logger.info("CREPE and TensorFlow are successfully loaded.")
        except ImportError:
            logger.warning("CREPE/TensorFlow not available. Falling back to librosa.pyin.")

    def detect_pitch(self, y: np.ndarray, sr: int) -> list[dict]:
        """
        Extracts the pitch (f0) and confidence every 10ms.
        Returns a list of dicts: [{'time': float, 'frequency': float, 'confidence': float}]
        Low confidence/energy frames are set to 0.0 Hz (to represent rests/silences).
        """
        if len(y) == 0:
            return []

        # 10ms step size in seconds
        hop_duration = 0.010
        hop_length = int(sr * hop_duration)

        # Calculate energy (RMS) to filter out noise/silence in quiet parts
        # If signal is too quiet, pitch detection should be treated as silence.
        try:
            import librosa
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
            # Normalize RMS to [0, 1] range
            rms_max = np.max(rms) if np.max(rms) > 0 else 1.0
            rms_normalized = rms / rms_max
        except Exception as e:
            logger.error(f"Failed to calculate RMS energy: {e}")
            rms_normalized = np.ones(int(np.ceil(len(y) / hop_length)))

        times = []
        frequencies = []
        confidences = []

        # Attempt to run CREPE
        if self.has_crepe:
            try:
                import crepe
                # Run CREPE prediction
                # step_size=10 is 10ms hop
                # model_capacity='tiny' is lightweight and fast for hackathons
                logger.info("Running pitch detection using CREPE (tiny model)...")
                t, f, c, _ = crepe.predict(y, sr, step_size=10, model_capacity='tiny', verbose=0)
                
                times = t
                frequencies = f
                confidences = c
            except Exception as e:
                logger.error(f"CREPE execution failed, falling back to librosa.pyin: {e}")
                self.has_crepe = False  # Reset flag so we run pyin fallback

        # Fallback to librosa.pyin
        if not self.has_crepe:
            try:
                import librosa
                logger.info("Running pitch detection using librosa.pyin...")
                # Human vocal range: C2 (~65Hz) to C7 (~2093Hz)
                fmin = librosa.note_to_hz('C2')
                fmax = librosa.note_to_hz('C7')
                
                f0, voiced_flag, voiced_probs = librosa.pyin(
                    y,
                    fmin=fmin,
                    fmax=fmax,
                    sr=sr,
                    hop_length=hop_length,
                    fill_value=0.0
                )
                
                # Align lengths
                n_frames = len(f0)
                times = np.arange(n_frames) * hop_duration
                frequencies = f0
                confidences = voiced_probs
                # In pyin, voiced_probs can be NaN for unvoiced frames
                confidences = np.nan_to_num(confidences, nan=0.0)
            except Exception as e:
                logger.error(f"librosa.pyin failed: {e}")
                raise RuntimeError(f"All pitch detection methods failed: {e}")

        # Post-process frames: filter by confidence and RMS energy
        result = []
        n_frames = min(len(times), len(frequencies), len(confidences))
        
        # We also align rms length with frame count
        rms_len = len(rms_normalized)

        for i in range(n_frames):
            t = float(times[i])
            freq = float(frequencies[i])
            conf = float(confidences[i])
            
            # Get RMS value for this frame (boundary safe)
            r_val = float(rms_normalized[i]) if i < rms_len else 1.0

            # Signal is considered silence if:
            # - Confidence is below the threshold
            # - OR RMS energy is extremely low (below 3% of maximum energy, filtering out background hum)
            # - OR frequency is 0
            if conf < settings.CONFIDENCE_THRESHOLD or r_val < 0.03 or freq <= 0:
                final_freq = 0.0
            else:
                final_freq = freq
                
            result.append({
                "time": t,
                "frequency": final_freq,
                "confidence": conf
            })
            
        return result
