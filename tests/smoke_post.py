"""Manual end-to-end smoke test: synth a 1s A4 tone, POST to /api/transcribe."""

import io
import sys

import numpy as np
import soundfile as sf
import urllib.request


def main(base_url: str = "http://127.0.0.1:8000") -> int:
    sr = 22050
    t = np.arange(int(sr * 1.0), dtype=np.float32) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

    wav_buf = io.BytesIO()
    sf.write(wav_buf, tone, sr, format="WAV", subtype="PCM_16")
    wav_bytes = wav_buf.getvalue()

    boundary = "----musicmesmoke"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="file"; filename="tone.wav"\r\n',
        b"Content-Type: audio/wav\r\n\r\n",
        wav_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ])

    req = urllib.request.Request(
        f"{base_url}/api/transcribe",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        print(f"Status: {resp.status}")
        print(resp.read().decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
