"""
Extraction de features cepstrales : MFCC, delta, delta-delta.
"""

import numpy as np
import librosa


def extract_mfcc(
    y: np.ndarray, sr: int, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128
) -> np.ndarray:
    """MFCC par frame."""
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)


def extract_mfcc_delta(mfccs: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Calcule les delta (dérivées) des MFCC.

    Args:
        mfccs: Matrice MFCC (n_mfcc, n_frames)
        order: 1 pour delta, 2 pour delta-delta

    Returns:
        Matrice delta de même forme
    """
    return librosa.feature.delta(mfccs, order=order)


def _aggregate_feature(values: np.ndarray) -> dict:
    """Calcule les statistiques agrégées."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def extract_cepstral_features(
    y: np.ndarray, sr: int, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512
) -> dict:
    """
    Extrait et agrège toutes les features cepstrales.

    Returns:
        Dict avec mfcc, delta, delta2 et leurs statistiques.
    """
    mfcc = extract_mfcc(y, sr, n_mfcc, n_fft, hop_length)
    delta = extract_mfcc_delta(mfcc, order=1)
    delta2 = extract_mfcc_delta(mfcc, order=2)

    mfcc_stats = {}
    for i in range(mfcc.shape[0]):
        mfcc_stats[f"mfcc_{i}"] = _aggregate_feature(mfcc[i])

    return {
        "mfcc": {
            "values": mfcc.tolist(),
            "stats": mfcc_stats,
        },
        "delta": {
            "values": delta.tolist(),
        },
        "delta2": {
            "values": delta2.tolist(),
        },
    }
