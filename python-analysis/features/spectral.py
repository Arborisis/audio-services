"""
Extraction de features spectrales : centroid, bandwidth, rolloff, contrast, flatness.
"""

import numpy as np
import librosa


def extract_spectral_centroid(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Spectral Centroid par frame."""
    return librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]


def extract_spectral_bandwidth(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Spectral Bandwidth par frame."""
    return librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]


def extract_spectral_rolloff(
    y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, roll_percent: float = 0.85
) -> np.ndarray:
    """Spectral Rolloff par frame."""
    return librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=roll_percent)[0]


def extract_spectral_contrast(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Spectral Contrast par frame."""
    return librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)


def extract_spectral_flatness(y: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Spectral Flatness par frame."""
    return librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]


def _aggregate_feature(values: np.ndarray) -> dict:
    """Calcule les statistiques agrégées."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def extract_spectral_features(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> dict:
    """
    Extrait et agrège toutes les features spectrales.

    Returns:
        Dict avec centroid, bandwidth, rolloff, contrast, flatness.
    """
    centroid = extract_spectral_centroid(y, sr, n_fft, hop_length)
    bandwidth = extract_spectral_bandwidth(y, sr, n_fft, hop_length)
    rolloff = extract_spectral_rolloff(y, sr, n_fft, hop_length)
    contrast = extract_spectral_contrast(y, sr, n_fft, hop_length)
    flatness = extract_spectral_flatness(y, n_fft, hop_length)

    result = {
        "centroid": {
            "values": centroid.tolist(),
            "stats": _aggregate_feature(centroid),
        },
        "bandwidth": {
            "values": bandwidth.tolist(),
            "stats": _aggregate_feature(bandwidth),
        },
        "rolloff": {
            "values": rolloff.tolist(),
            "stats": _aggregate_feature(rolloff),
        },
        "flatness": {
            "values": flatness.tolist(),
            "stats": _aggregate_feature(flatness),
        },
    }

    # Spectral contrast a plusieurs bandes
    contrast_stats = {}
    for i in range(contrast.shape[0]):
        band_values = contrast[i]
        contrast_stats[f"band_{i}"] = _aggregate_feature(band_values)
    result["contrast"] = {
        "values": contrast.tolist(),
        "stats": contrast_stats,
    }

    return result
