# melody-scribe-api

Backend FastAPI pour la transcription de mélodies fredonnées en notation musicale.

Pipeline : audio (n'importe quel format) → CREPE (détection de hauteur) → quantification NumPy sur une grille stricte 120 BPM → JSON `MusicalNote[]`. Endpoint d'export MIDI stateless inclus.

## Stack

- Python 3.11
- FastAPI + Uvicorn
- CREPE (TensorFlow CPU) pour la détection de pitch
- librosa / scipy / numpy pour le traitement audio
- mido pour l'export MIDI
- ffmpeg (via le Dockerfile) pour décoder WebM/Opus, MP3, MP4, WAV, …

## Pré-requis (dev local)

- Python 3.11
- ffmpeg installé et accessible dans le `PATH` (pour décoder autre chose que WAV)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install "setuptools<81" wheel
pip install -r requirements.txt
```

> Note : sur Windows, l'installation de `tensorflow-cpu` et `crepe` est lourde (~500 Mo). Si vous voulez itérer uniquement sur la logique métier (quantif, MIDI, schémas), `pytest` tourne déjà avec `pip install pytest numpy mido librosa scipy soundfile python-multipart fastapi uvicorn`.

## Lancer le serveur

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

- API : http://127.0.0.1:8000
- Docs OpenAPI : http://127.0.0.1:8000/docs

Au démarrage, le modèle CREPE est warmé sur 1 seconde de silence pour éviter la latence de chargement sur la première requête.

## Endpoints

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/health` | Vérification de santé |
| POST | `/api/transcribe` | `multipart/form-data` (champ `audio`) → `TranscriptionResponse` JSON |
| POST | `/api/export/midi` | JSON `{ notes: MusicalNote[] }` → fichier `.mid` (`audio/midi`) |

### Modèles de données

`MusicalNote` :

```json
{ "pitch": "C4", "duration": "q", "isRest": false }
```

- `pitch` : nom de note avec dièses uniquement (`C4`, `F#5`, …). Pour un silence : `"rest"`.
- `duration` : VexFlow (`w` = ronde, `h` = blanche, `q` = noire, `8` = croche, `16` = double-croche).
- `isRest` : `true` pour les silences/bruits.

`TranscriptionResponse` :

```json
{ "status": "success", "data": [ /* MusicalNote[] */ ], "error": null }
```

### Exemple `curl`

```bash
curl -X POST http://127.0.0.1:8000/api/transcribe \
  -F "audio=@recording.webm"
```

```bash
curl -X POST http://127.0.0.1:8000/api/export/midi \
  -H "Content-Type: application/json" \
  -d '{"notes":[{"pitch":"C4","duration":"q","isRest":false}]}' \
  --output transcription.mid
```

## Architecture

```
app/
  api/
    schemas.py        # Modèles Pydantic (data contract)
    transcribe.py     # POST /api/transcribe
    export.py         # POST /api/export/midi
  services/
    audio_loader.py   # Décodage audio + RMS frame-par-frame
    pitch_detection.py# Wrapper CREPE (singleton + warmup)
    quantizer.py      # Coeur numpy : grille 16es, merge runs, durations VexFlow
    midi_export.py    # mido MidiFile @ 120 BPM
  utils/
    music.py          # Hz↔MIDI↔nom, table durations
  config.py           # Constantes (BPM, seuils, sample rate, …)
  main.py             # FastAPI app + CORS + warmup CREPE
tests/                # Tests unitaires (cf. ci-dessous)
```

## Configuration via variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `BPM` | `120` | Tempo de la grille de quantification |
| `SAMPLE_RATE` | `16000` | Fréquence cible pour CREPE |
| `STEP_MS` | `10` | Hop des frames CREPE + RMS |
| `MIDI_MIN` / `MIDI_MAX` | `36` / `84` | Plage vocale autorisée |
| `CONFIDENCE_THRESHOLD` | `0.5` | Seuil de confiance CREPE pour "voix présente" |
| `RMS_THRESHOLD` | `0.01` | Seuil RMS pour "voix présente" |
| `CREPE_MODEL` | `tiny` | `tiny` / `small` / `medium` / `large` / `full` |
| `CREPE_VITERBI` | `1` | Activer le lissage Viterbi |
| `MAX_UPLOAD_BYTES` | `26214400` | Taille max d'upload (25 Mo) |

## Tests

```powershell
# Tests unitaires rapides (sans CREPE/TensorFlow)
pytest

# Tests "slow" incluant l'inférence CREPE sur une sinusoide
pytest -m slow
```

Les tests rapides couvrent : conversions Hz/MIDI/nom, décomposition de durées, quantification synthétique end-to-end, round-trip MIDI.

## Docker / Railway

```bash
docker build -t melody-scribe-api .
docker run --rm -p 8000:8000 melody-scribe-api
```

Le `Dockerfile` ajoute `ffmpeg` et `libsndfile1`, et démarre `uvicorn` sur le port `$PORT` (compatible Railway). Le `Procfile` est conservé pour le fallback buildpack.

## Hors scope (extensions possibles)

- Authentification / quotas
- Persistance des transcriptions
- Triolets, notes pointées, signatures ≠ 4/4 ou tempos variables
- Détection automatique de tonalité (la quantif ne produit que des dièses)
- Streaming temps réel
