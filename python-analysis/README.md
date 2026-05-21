# Arborisis Audio Analysis — Module Python

Module de data science audio pour la plateforme Arborisis. Extraction de features et génération de visualisations avec librosa, scipy, matplotlib.

## Prérequis

- Python 3.10+
- ffmpeg (pour le décodage MP3/FLAC)

### Installation de ffmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

**Production (Contabo):**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## Installation

```bash
cd arborisis/python
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Architecture

```
python/
├── cli.py              # Point d'entrée CLI
├── config.py           # Configuration globale
├── pipeline.py         # Pipeline end-to-end
├── core/               # Fondations
│   ├── loader.py       # Chargement audio
│   ├── preprocessing.py # Filtrage, normalisation, VAD
│   ├── segmentation.py # Segmentation temporelle
│   └── utils.py        # I/O, plots
├── features/           # Extraction de features
│   ├── temporal.py     # ZCR, RMS, enveloppe
│   ├── spectral.py     # Centroid, bandwidth, rolloff, contrast, flatness
│   └── cepstral.py     # MFCC, delta, delta-delta
└── visualizations/     # Génération de figures
    ├── spectrograms.py # STFT, Mel, MFCC
    ├── heatmaps.py     # Corrélation, confusion
    └── activation_maps.py # Placeholder DL
```

## Utilisation CLI

```bash
# Analyse avec paramètres par défaut
python cli.py --input audio.wav --output ./out

# Analyse avec config personnalisée
python cli.py --input audio.wav --output ./out --config '{"n_fft":4096,"n_mels":256}'

# Output formaté
python cli.py --input audio.wav --output ./out --pretty
```

## Config JSON

```json
{
  "sample_rate": 22050,
  "n_fft": 2048,
  "hop_length": 512,
  "n_mels": 128,
  "n_mfcc": 13,
  "preprocessing": {
    "normalize": true,
    "normalize_method": "peak",
    "trim_silence": true,
    "top_db": 20,
    "filter": null,
    "vad": false
  },
  "visualizations": {
    "spectrograms": ["stft", "mel", "mfcc"],
    "heatmaps": ["feature_correlation"],
    "dpi": 150,
    "colormap": "viridis"
  }
}
```

## Tests

```bash
pytest tests/
```

## Limites

- Durée max : 10 minutes (`MAX_DURATION_SECONDS`)
- Formats supportés : WAV, MP3, FLAC, OGG, M4A
- RAM : les fichiers longs peuvent consommer beaucoup de mémoire
