"""
Extraction de features temporelles : ZCR, RMS, enveloppe.
"""

import numpy as np
import librosa


def extract_zcr(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Zero-Crossing Rate par frame."""
    return librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]


def extract_rms(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Root Mean Square energy par frame."""
    return librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]


def extract_envelope(y: np.ndarray, sr: int, frame_length: int = 2048) -> np.ndarray:
    """
    Enveloppe d'amplitude via la moyenne glissante.

    Args:
        y: Signal audio
        sr: Sample rate
        frame_length: Taille de la fenêtre de moyennage

    Returns:
        Enveloppe du signal
    """
    hop = frame_length // 4
    frames = np.abs(y)
    n_frames = 1 + (len(frames) - frame_length) // hop
    envelope = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        envelope[i] = np.mean(frames[start : start + frame_length])
    return envelope


def _aggregate_feature(values: np.ndarray) -> dict:
    """Calcule les statistiques agrégées d'une feature."""
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
    }


def extract_temporal_features(y: np.ndarray, sr: int) -> dict:
    """
    Extrait et agrège toutes les features temporelles.

    Returns:
        Dict avec 'zcr', 'rms', 'envelope' et leurs statistiques.
    """
    frame_length = 2048
    hop_length = 512

    zcr = extract_zcr(y, frame_length, hop_length)
    rms = extract_rms(y, frame_length, hop_length)
    envelope = extract_envelope(y, sr, frame_length)

    return {
        "zcr": {
            "values": zcr.tolist(),
            "stats": _aggregate_feature(zcr),
        },
        "rms": {
            "values": rms.tolist(),
            "stats": _aggregate_feature(rms),
        },
        "envelope": {
            "values": envelope.tolist(),
            "stats": _aggregate_feature(envelope),
        },
        "duration_seconds": float(len(y) / sr),
    }
