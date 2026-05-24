"""Amateur-friendly post-processing for a monophonic melody.

Runs after `monophonic.collapse_to_melody`. Three stages, in order:

1. `snap_to_major_key` — vocal pitch drifts a quarter-tone between syllables;
   that drift turns into accidentals on the staff. Detect the most likely
   major key from the pitch-class histogram, then quantize every note to the
   nearest scale degree.

2. `transpose_to_c_major` — the renderer always uses a default (C major) key
   signature, so non-C scale notes would still appear as accidentals. After
   snapping, shift the whole melody by `-root` semitones so the tonic is C —
   every output note now sits on a white key, no key-signature handling needed.

3. `normalize_octave` — basic-pitch routinely picks a sub-harmonic for hummed
   audio, putting the melody several ledger lines below the staff. Choose the
   whole-octave shift that maximises notes inside the treble clef range
   (A3-G5), ties broken by proximity to E4.
"""

from __future__ import annotations

from statistics import median
from typing import List

from app.models import CleanupOptions, Note, RepeatStrategy

# Treble clef sweet-spot: A3 (lowest comfortable ledger line below) to G5
# (just above the top staff line). Anchor on E4 — middle of the staff.
TREBLE_LOW = 57   # A3
TREBLE_HIGH = 79  # G5
TREBLE_ANCHOR = 64  # E4

# Semitone offsets of a major scale.
_MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)


def _shift(notes: List[Note], semitones: int) -> List[Note]:
    if semitones == 0:
        return notes
    return [
        Note(
            pitch=max(0, min(127, n.pitch + semitones)),
            start=n.start,
            end=n.end,
            velocity=n.velocity,
        )
        for n in notes
    ]


def normalize_octave(
    notes: List[Note],
    low: int = TREBLE_LOW,
    high: int = TREBLE_HIGH,
    anchor: int = TREBLE_ANCHOR,
) -> List[Note]:
    """Octave-shift the melody so as many notes as possible fall in [low, high]."""
    if not notes:
        return notes

    pitches = [n.pitch for n in notes]
    med = median(pitches)

    best_shift = 0
    best_in_range = -1
    best_distance = 1_000

    for octaves in range(-5, 6):
        shift = octaves * 12
        in_range = sum(1 for p in pitches if low <= p + shift <= high)
        distance = abs(med + shift - anchor)
        if (in_range, -distance) > (best_in_range, -best_distance):
            best_in_range = in_range
            best_distance = distance
            best_shift = shift

    return _shift(notes, best_shift)


def _detect_major_key_root(notes: List[Note]) -> int:
    """Return the MIDI pitch class (0-11) of the major key that best fits `notes`."""
    histogram = [0] * 12
    for n in notes:
        histogram[n.pitch % 12] += 1

    best_root = 0
    best_score = -1
    for root in range(12):
        scale_classes = {(root + step) % 12 for step in _MAJOR_SCALE}
        score = sum(histogram[pc] for pc in scale_classes)
        if score > best_score:
            best_score = score
            best_root = root
    return best_root


def _nearest_scale_pitch(pitch: int, root: int) -> int:
    """Snap `pitch` to the closest member of the major scale rooted at `root`."""
    pitch_class = pitch % 12
    # Distance to each scale degree (both up and down, take min absolute).
    best_offset = 0
    best_distance = 12
    for step in _MAJOR_SCALE:
        scale_class = (root + step) % 12
        diff = (scale_class - pitch_class) % 12
        # Choose the shorter direction: e.g. +1 or -11.
        signed = diff if diff <= 6 else diff - 12
        if abs(signed) < best_distance:
            best_distance = abs(signed)
            best_offset = signed
    return pitch + best_offset


def snap_to_major_key(notes: List[Note]) -> List[Note]:
    """Detect the best-fit major key and snap every note to the nearest scale degree."""
    if not notes:
        return notes

    root = _detect_major_key_root(notes)
    snapped: List[Note] = []
    for n in notes:
        snapped.append(
            Note(
                pitch=_nearest_scale_pitch(n.pitch, root),
                start=n.start,
                end=n.end,
                velocity=n.velocity,
            )
        )
    return snapped


def transpose_to_c_major(notes: List[Note], root: int) -> List[Note]:
    """Shift the melody so the detected tonic becomes C.

    After `snap_to_major_key`, every note belongs to the detected major scale.
    Shifting by `-root` (preferring the smaller of -root vs 12-root to stay
    near the original register) lands the whole melody on the white keys.
    """
    shift = -root if root <= 6 else 12 - root
    return _shift(notes, shift)


def smooth_octave_jumps(notes: List[Note], max_passes: int = 4) -> List[Note]:
    """Pull single-note octave outliers toward their neighbours.

    basic-pitch routinely emits the same hummed pitch one octave away on
    alternate detections (e.g. C4 C5 C4 C5 instead of C4 C4 C4 C4). For each
    note, if shifting it by ±12 semitones brings it closer to the mean of its
    two neighbours, do so. Iterate until stable (or up to `max_passes`).
    """
    if len(notes) < 3:
        return notes

    current = list(notes)
    for _ in range(max_passes):
        changed = False
        next_pass = list(current)
        for i in range(1, len(current) - 1):
            prev_p = current[i - 1].pitch
            next_p = current[i + 1].pitch
            neighbour_mean = (prev_p + next_p) / 2.0
            here = current[i].pitch
            best_pitch = here
            best_distance = abs(here - neighbour_mean)
            for shift in (-12, 12):
                candidate = here + shift
                if 0 <= candidate <= 127 and abs(candidate - neighbour_mean) < best_distance - 0.5:
                    best_pitch = candidate
                    best_distance = abs(candidate - neighbour_mean)
            if best_pitch != here:
                next_pass[i] = Note(
                    pitch=best_pitch,
                    start=current[i].start,
                    end=current[i].end,
                    velocity=current[i].velocity,
                )
                changed = True
        current = next_pass
        if not changed:
            break
    return current


def group_consecutive_repeats(
    notes: List[Note],
    strategy: RepeatStrategy = "merge",
    max_gap: float = 0.15,
) -> List[Note]:
    """Decide what to do with consecutive same-pitch notes within `max_gap` seconds.

    basic-pitch re-fires onsets during breath release on sustained humming,
    producing C4 C4 C4 sequences from a single held syllable. The right
    interpretation is musical, not algorithmic — so we let the caller choose:

    - ``merge``  collapse the run into a single held note (sum the durations).
    - ``tie``    keep each onset as its own note but flag every note except the
                 last in a run with ``tied_to_next=True`` so the renderer draws
                 a VexFlow liaison.
    - ``split``  do nothing, treat each onset as a distinct articulation.

    A pair is considered "consecutive" when they share a pitch AND the second
    starts within ``max_gap`` seconds of the first ending.
    """
    if not notes or strategy == "split":
        return notes

    out: List[Note] = [notes[0].model_copy()]
    for n in notes[1:]:
        prev = out[-1]
        adjacent = n.pitch == prev.pitch and n.start - prev.end <= max_gap

        if not adjacent:
            out.append(n.model_copy())
            continue

        if strategy == "merge":
            out[-1] = Note(
                pitch=prev.pitch,
                start=prev.start,
                end=max(prev.end, n.end),
                velocity=max(prev.velocity, n.velocity),
                tied_to_next=False,
            )
        else:  # tie
            out[-1] = prev.model_copy(update={"tied_to_next": True})
            out.append(n.model_copy())

    return out


# Back-compat shim — older call sites and tests use the merge-only name.
def dedup_consecutive_repeats(notes: List[Note], max_gap: float = 0.15) -> List[Note]:
    """Deprecated alias — equivalent to ``group_consecutive_repeats(strategy='merge')``."""
    return group_consecutive_repeats(notes, strategy="merge", max_gap=max_gap)


def cleanup(notes: List[Note], options: CleanupOptions | None = None) -> List[Note]:
    """Apply the full amateur-friendly pipeline."""
    if not notes:
        return notes
    strategy, max_gap = (options or CleanupOptions()).resolved()
    root = _detect_major_key_root(notes)
    snapped = snap_to_major_key(notes)
    transposed = transpose_to_c_major(snapped, root)
    normalized = normalize_octave(transposed)
    smoothed = smooth_octave_jumps(normalized)
    return group_consecutive_repeats(smoothed, strategy=strategy, max_gap=max_gap)
