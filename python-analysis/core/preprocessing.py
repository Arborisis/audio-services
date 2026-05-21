from typing import Union
"""
Prétraitement audio : filtrage, normalisation, VAD, trimming.
"""

import numpy as np
import scipy.signal
import librosa

from config import DEFAULT_ENERGY_THRESHOLD


def apply_filter(y: np.ndarray, sr: int, filter_type: str, cutoff: Union[float, tuple]) -> np.ndarray:
    """
    Applique un filtre numérique au signal.

    Args:
        y: Signal audio
        sr: Sample rate
        filter_type: 'lowpass', 'highpass', 'bandpass'
        cutoff: Fréquence de coupure (float pour low/high, tuple pour bandpass)

    Returns:
        Signal filtré
    """
    nyquist = sr / 2.0

    if filter_type == "lowpass":
        normalized_cutoff = cutoff / nyquist
        b, a = scipy.signal.butter(4, normalized_cutoff, btype="low")
    elif filter_type == "highpass":
        normalized_cutoff = cutoff / nyquist
        b, a = scipy.signal.butter(4, normalized_cutoff, btype="high")
    elif filter_type == "bandpass":
        if not isinstance(cutoff, (list, tuple)) or len(cutoff) != 2:
            raise ValueError("bandpass nécessite un tuple (low, high)")
        low, high = cutoff
        normalized_cutoff = [low / nyquist, high / nyquist]
        b, a = scipy.signal.butter(4, normalized_cutoff, btype="band")
    else:
        raise ValueError(f"Type de filtre inconnu: {filter_type}")

    return scipy.signal.filtfilt(b, a, y)


def normalize(y: np.ndarray, method: str = "peak") -> np.ndarray:
    """
    Normalise le signal audio.

    Args:
        y: Signal audio
        method: 'peak', 'rms', 'zscore'

    Returns:
        Signal normalisé
    """
    if method == "peak":
        peak = np.max(np.abs(y))
        if peak == 0:
            return y
        return y / peak
    elif method == "rms":
        rms = np.sqrt(np.mean(y ** 2))
        if rms == 0:
            return y
        return y / rms
    elif method == "zscore":
        mean = np.mean(y)
        std = np.std(y)
        if std == 0:
            return y - mean
        return (y - mean) / std
    else:
        raise ValueError(f"Méthode de normalisation inconnue: {method}")


def detect_voice_activity(
    y: np.ndarray,
    sr: int,
    frame_length: int = 2048,
    hop_length: int = 512,
    energy_threshold: float = DEFAULT_ENERGY_THRESHOLD,
    use_zcr: bool = False,
    zcr_threshold: float = 0.1,
) -> np.ndarray:
    """
    Détection d'activité vocale (VAD) basée sur l'énergie.

    Args:
        y: Signal audio
        sr: Sample rate
        frame_length: Taille de la fenêtre
        hop_length: Pas de la fenêtre
        energy_threshold: Seuil d'énergie relatif (0-1)
        use_zcr: Utiliser aussi le zero-crossing rate
        zcr_threshold: Seuil ZCR

    Returns:
        Masque booléen par frame (True = actif)
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    max_rms = np.max(rms)
    if max_rms == 0:
        return np.ones_like(rms, dtype=bool)

    energy_mask = rms > (energy_threshold * max_rms)

    if use_zcr:
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        zcr_mask = zcr < zcr_threshold
        return energy_mask & zcr_mask

    return energy_mask


def trim_silence(y: np.ndarray, sr: int, top_db: int = 20) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Supprime le silence au début et à la fin.

    Args:
        y: Signal audio
        sr: Sample rate
        top_db: Seuil en dB sous le pic

    Returns:
        (signal_trimmed, (start_frame, end_frame))
    """
    if y.size == 0:
        return y, (0, 0)

    peak = np.max(np.abs(y))
    if peak == 0:
        return y, (0, len(y))

    threshold = peak * (10 ** (-top_db / 20))
    non_silent = np.flatnonzero(np.abs(y) > threshold)

    if non_silent.size == 0:
        return y, (0, len(y))

    start = int(non_silent[0])
    end = int(non_silent[-1] + 1)

    return y[start:end], (start, end)
