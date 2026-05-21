"""
Features liées à la biodiversité acoustique et à l'activité sonore.
"""

import numpy as np
import librosa


def extract_acoustic_events(y, sr, hop_length=512, threshold_db=-40):
    """
    Détecte les événements sonores par énergie frame-level.

    Returns:
        dict avec event_count, event_density_per_min, mean_event_duration_s
    """
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    threshold = librosa.db_to_amplitude(threshold_db)
    events = rms > threshold

    # Détection des fronts montants (débuts d'événements)
    event_starts = np.where(np.diff(events.astype(int)) == 1)[0]

    duration_minutes = len(y) / sr / 60

    mean_duration = 0.0
    if len(event_starts) > 1:
        mean_duration = float(np.mean(np.diff(event_starts)) * hop_length / sr)

    return {
        "event_count": int(len(event_starts)),
        "event_density_per_min": float(len(event_starts) / duration_minutes) if duration_minutes > 0 else 0.0,
        "mean_event_duration_s": mean_duration,
    }


def compute_acoustic_diversity_index(y, sr, n_bands=10, fmin=100, fmax=11025):
    """
    ADI simplifié : Shannon diversity sur l'énergie par bande de fréquence.

    Returns:
        dict avec adi, adi_normalized, n_bands
    """
    band_edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bands + 1)

    stft = np.abs(librosa.stft(y))
    frequencies = librosa.fft_frequencies(sr=sr)

    band_energies = []
    for i in range(n_bands):
        mask = (frequencies >= band_edges[i]) & (frequencies < band_edges[i + 1])
        if np.any(mask):
            band_energies.append(np.sum(stft[mask, :]))

    energies = np.array(band_energies, dtype=np.float64)

    if energies.sum() == 0:
        return {"adi": 0.0, "adi_normalized": 0.0, "n_bands": n_bands}

    proportions = energies / energies.sum()
    proportions = proportions[proportions > 0]

    shannon = -np.sum(proportions * np.log(proportions))
    max_shannon = np.log(n_bands)

    return {
        "adi": float(shannon),
        "adi_normalized": float(shannon / max_shannon) if max_shannon > 0 else 0.0,
        "n_bands": n_bands,
    }
