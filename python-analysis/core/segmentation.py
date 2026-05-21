"""
Segmentation temporelle du signal audio.
"""

import numpy as np
import librosa


def segment_frames(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Segmente le signal en fenêtres glissantes.

    Args:
        y: Signal audio
        sr: Sample rate
        frame_length: Taille de chaque frame
        hop_length: Pas entre frames

    Returns:
        Tableau 2D (n_frames, frame_length)
    """
    n_frames = 1 + (len(y) - frame_length) // hop_length
    if n_frames <= 0:
        return np.array([y])

    frames = np.zeros((n_frames, frame_length), dtype=y.dtype)
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        if end <= len(y):
            frames[i] = y[start:end]
        else:
            frames[i, : len(y) - start] = y[start:]

    return frames


def segment_onsets(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """
    Détecte les onsets et retourne les indices de début de segment.

    Args:
        y: Signal audio
        sr: Sample rate
        hop_length: Pas pour la détection

    Returns:
        Indices d'onset en échantillons
    """
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)
    return onset_samples
