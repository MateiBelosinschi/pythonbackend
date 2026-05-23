# VexFlow duration string → number of beats (at quarter-note = 1 beat)
VEXFLOW_DURATION_TO_BEATS: dict[str, float] = {
    "w":  4.0,   # whole note
    "h":  2.0,   # half note
    "q":  1.0,   # quarter note
    "8":  0.5,   # eighth note
    "16": 0.25,  # sixteenth note
}

# Beats → VexFlow duration string (used for quantization output)
BEATS_TO_VEXFLOW: dict[float, str] = {v: k for k, v in VEXFLOW_DURATION_TO_BEATS.items()}

# MIDI pitch class (0-11) → note name (using sharps by convention)
PITCH_CLASS_TO_NAME = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Valid key signatures for scale snapping
VALID_KEYS = [
    "C", "G", "D", "A", "E", "B", "F#", "C#",
    "F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb",
    "Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "A#m",
    "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm", "Abm"
]
