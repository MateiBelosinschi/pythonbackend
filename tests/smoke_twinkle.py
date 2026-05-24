"""Live POST of synthetic Twinkle Twinkle to the running server."""
import io
import sys
import urllib.request

import numpy as np
import soundfile as sf

SR = 22050
TWINKLE = [
    (261.63, 0.5), (261.63, 0.5), (392.00, 0.5), (392.00, 0.5),
    (440.00, 0.5), (440.00, 0.5), (392.00, 1.0),
    (349.23, 0.5), (349.23, 0.5), (329.63, 0.5), (329.63, 0.5),
    (293.66, 0.5), (293.66, 0.5), (261.63, 1.0),
]

chunks = []
gap = np.zeros(int(0.06 * SR), dtype=np.float32)
for i, (f, d) in enumerate(TWINKLE):
    n = int(d * SR)
    t = np.arange(n, dtype=np.float32) / SR
    env = np.ones(n, dtype=np.float32)
    fade = int(SR * 0.01)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    chunks.append((0.3 * env * np.sin(2 * np.pi * f * t)).astype(np.float32))
    if i + 1 < len(TWINKLE):
        chunks.append(gap)
waveform = np.concatenate(chunks)

buf = io.BytesIO()
sf.write(buf, waveform, SR, format="WAV", subtype="PCM_16")
wav = buf.getvalue()

boundary = "----twinklesmoke"
body = b"".join([
    f"--{boundary}\r\n".encode(),
    b'Content-Disposition: form-data; name="file"; filename="twinkle.wav"\r\n',
    b"Content-Type: audio/wav\r\n\r\n",
    wav,
    f"\r\n--{boundary}--\r\n".encode(),
])

req = urllib.request.Request(
    "http://127.0.0.1:8000/api/transcribe",
    data=body,
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as r:
    import json
    result = json.loads(r.read())
    print(f"Status: {r.status}")
    print(f"Notes: {len(result['notes'])}")
    print("Pitches:", [n["pitch"] for n in result["notes"]])
    print("Median pitch:", sorted(n["pitch"] for n in result["notes"])[len(result["notes"]) // 2])
