import math
from app.utils.constants import PITCH_CLASS_TO_NAME


def hz_to_midi(frequency: float) -> int:
    """
    Converts a frequency in Hz to the nearest MIDI note number (integer semitone).
    Returns 0 if frequency is invalid (<= 0). Range is clamped to [0, 127].
    """
    if frequency <= 0:
        return 0
    midi = 69 + 12 * math.log2(frequency / 440.0)
    return clamp_midi(int(round(midi)))


def clamp_midi(midi: int) -> int:
    """Ensures MIDI note number is within valid range [0, 127]."""
    return max(0, min(127, midi))


def midi_to_note_name(midi: int) -> str:
    """
    Converts a MIDI note number to a standard pitch string e.g. 60 → "C4", 69 → "A4".
    Returns "Rest" for out-of-range values.
    """
    if not (0 <= midi <= 127):
        return "Rest"
    octave = (midi // 12) - 1
    note = PITCH_CLASS_TO_NAME[midi % 12]
    return f"{note}{octave}"
