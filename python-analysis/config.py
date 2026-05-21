"""
Configuration globale pour le pipeline d'analyse audio Arborisis.
"""

import os

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Paramètres audio par défaut
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_N_MELS = 128
DEFAULT_N_MFCC = 13

# Paramètres de visualisation
DEFAULT_DPI = 150
DEFAULT_FIGURE_FORMAT = "png"
DEFAULT_COLORMAP = "viridis"

# Limites de sécurité
MAX_DURATION_SECONDS = 600  # 10 minutes
MAX_FILE_SIZE_MB = 500

# Paramètres de prétraitement
DEFAULT_TOP_DB = 20
DEFAULT_ENERGY_THRESHOLD = 0.01

# Paramètres de segmentation
DEFAULT_FRAME_LENGTH = 2048

# Ensembles de paramètres complets
DEFAULT_CONFIG = {
    "sample_rate": DEFAULT_SAMPLE_RATE,
    "n_fft": DEFAULT_N_FFT,
    "hop_length": DEFAULT_HOP_LENGTH,
    "n_mels": DEFAULT_N_MELS,
    "n_mfcc": DEFAULT_N_MFCC,
    "preprocessing": {
        "normalize": True,
        "normalize_method": "peak",
        "trim_silence": True,
        "top_db": DEFAULT_TOP_DB,
        "filter": None,
        "vad": False,
    },
    "visualizations": {
        "spectrograms": ["stft", "mel", "mfcc"],
        "heatmaps": ["feature_correlation"],
        "dpi": DEFAULT_DPI,
        "colormap": DEFAULT_COLORMAP,
    },
}


from typing import Optional

def merge_config(user_config: Optional[dict]) -> dict:
    """Fusionne la config utilisateur avec les valeurs par défaut."""
    if user_config is None or not isinstance(user_config, dict):
        return DEFAULT_CONFIG.copy()

    merged = DEFAULT_CONFIG.copy()
    for key, value in user_config.items():
        if isinstance(value, dict) and key in merged:
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
