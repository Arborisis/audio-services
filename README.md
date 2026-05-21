# Arborisis Audio Services

<p align="center">
  <img src="https://raw.githubusercontent.com/Arborisis/.github/main/profile/logo.svg" alt="Arborisis Logo" width="150" />
</p>

<p align="center">
  <em>Services d'analyse audio pour la plateforme Arborisis.</em>
</p>

<p align="center">
  <a href="https://github.com/Arborisis/audio-services/actions"><img src="https://img.shields.io/github/actions/workflow/status/Arborisis/audio-services/ci.yml?branch=main&style=flat-square&label=CI" alt="CI" /></a>
  <a href="https://github.com/Arborisis/audio-services/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Arborisis/audio-services?style=flat-square" alt="License" /></a>
</p>

---

## Overview

Services Python pour l'analyse audio automatisee du pipeline Arborisis. Ce repository contient deux services complementaires :

### 1. Audio Analyzer (`audio-analyzer/`)

Service **FastAPI** pour l'analyse audio automatisee. Il recoit des requetes du Worker Cloudflare, telecharge le fichier audio depuis R2, execute l'analyse, upload les resultats dans R2, et notifie Laravel via callback.

**Pipeline :**
```
Worker Cloudflare → POST /analyze → FastAPI Background Task
    → Download R2 → FFmpeg / FFprobe → Librosa → BirdNET
    → Upload resultats R2 → Callback Laravel
```

**Features :**
- Classification des especes via BirdNET
- Generation de spectrogrammes WebP
- Extraction de features audio (MFCC, centroid, etc.)
- Preview MP3 automatique
- Resume et metadonnees

### 2. Python Analysis (`python-analysis/`)

Module de **data science audio** pour la plateforme. Extraction de features et generation de visualisations avec librosa, scipy, matplotlib.

**Features :**
- Chargement et preprocessing audio
- Features temporelles (ZCR, RMS, enveloppe)
- Features spectrales (centroid, bandwidth, rolloff)
- Features cepstrales (MFCC, delta)
- Visualisations (spectrogrammes, heatmaps)

## Architecture

```
audio-services/
├── audio-analyzer/          # Service FastAPI
│   ├── app/
│   │   ├── core/           # Fondations (exceptions, logger, security)
│   │   ├── models/         # Modeles Pydantic
│   │   ├── routers/        # Endpoints FastAPI
│   │   └── services/       # Logique metier
│   │       ├── audio_downloader.py
│   │       ├── birdnet_runner.py
│   │       ├── feature_extractor.py
│   │       ├── preview_generator.py
│   │       └── spectrogram_generator.py
│   ├── Dockerfile
│   └── requirements.txt
│
└── python-analysis/         # Module data science
    ├── core/               # Chargement, preprocessing, segmentation
    ├── features/           # Extraction de features
    ├── visualizations/     # Generation de figures
    └── cli.py             # Point d'entree CLI
```

## Stack technique

- **FastAPI** + Uvicorn
- **boto3** (R2/S3 compatible)
- **librosa** + **numpy** + **scipy**
- **matplotlib** + **Pillow** (visualisations)
- **BirdNET Analyzer** (classification especes)
- **FFmpeg** / **FFprobe**

## Installation

### Audio Analyzer

```bash
cd audio-analyzer
cp .env.example .env
pip install -r requirements.txt

# Lancer
uvicorn app.main:app --reload --port 8000
```

### Python Analysis

```bash
cd python-analysis
pip install -r requirements.txt

# CLI
python cli.py --input audio.wav --output ./out
```

## Tests

```bash
# Audio Analyzer
cd audio-analyzer
pytest -v

# Python Analysis
cd python-analysis
python -m pytest tests/ -v
```

## Deploiement

### Docker (production)

```bash
# Audio Analyzer
cd audio-analyzer
docker build -t arborisis-audio-analyzer .
docker run -d --env-file .env -p 8000:8000 arborisis-audio-analyzer
```

### Multi-instance avec load balancer

Pour scaler horizontalement :

```bash
cd infrastructure/audio-analyzer-worker  # Voir repo infrastructure
docker compose up -d --build
```

## Integration avec Arborisis

```
Laravel App → Request Analysis → Cloudflare Worker
    → Audio Analyzer (ce repo) → R2 Storage
    → Callback Laravel → Update Sound Analysis
```

## License

[MIT License](LICENSE)
