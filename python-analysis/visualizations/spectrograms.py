from typing import Optional
"""
Génération de spectrogrammes : STFT, Mel, MFCC.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

from core.utils import save_figure, setup_plot_style


def plot_stft(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    scale: str = "linear",
    output_path: str = None,
    cmap: str = "viridis",
    dpi: int = 150,
) -> Optional[str]:
    """
    Génère un spectrogramme STFT.

    Args:
        scale: 'linear', 'log'
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    if scale == "log":
        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", ax=ax, cmap=cmap
        )
    else:
        img = librosa.display.specshow(
            S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="linear", ax=ax, cmap=cmap
        )

    ax.set_title("Spectrogramme STFT", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Fréquence (Hz)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Amplitude (dB)")

    if output_path:
        return save_figure(fig, output_path, dpi=dpi)
    plt.show()
    return None


def plot_mel_spectrogram(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    output_path: str = None,
    cmap: str = "viridis",
    dpi: int = 150,
) -> Optional[str]:
    """Génère un spectrogramme Mel."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)

    img = librosa.display.specshow(
        S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="mel", ax=ax, cmap=cmap
    )

    ax.set_title("Spectrogramme Mel", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Fréquence Mel")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", label="Amplitude (dB)")

    if output_path:
        return save_figure(fig, output_path, dpi=dpi)
    plt.show()
    return None


def plot_mfcc(
    mfccs: np.ndarray,
    sr: int,
    hop_length: int = 512,
    output_path: str = None,
    cmap: str = "viridis",
    dpi: int = 150,
) -> Optional[str]:
    """Génère une visualisation des MFCC."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    img = librosa.display.specshow(
        mfccs, sr=sr, hop_length=hop_length, x_axis="time", ax=ax, cmap=cmap
    )

    ax.set_title("MFCC", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Coefficients MFCC")
    fig.colorbar(img, ax=ax, label="Valeur")

    if output_path:
        return save_figure(fig, output_path, dpi=dpi)
    plt.show()
    return None


def plot_combined_spectrograms(
    y: np.ndarray, sr: int, config: dict, output_dir: str
) -> dict:
    """
    Génère les 3 spectrogrammes et retourne les chemins.

    Returns:
        Dict {type: chemin_fichier}
    """
    n_fft = config.get("n_fft", 2048)
    hop_length = config.get("hop_length", 512)
    n_mels = config.get("n_mels", 128)
    cmap = config.get("colormap", "viridis")
    dpi = config.get("dpi", 150)

    results = {}

    # STFT
    stft_scale = config.get("stft_scale", "linear")
    results["stft"] = plot_stft(
        y, sr, n_fft, hop_length, scale=stft_scale,
        output_path=f"{output_dir}/spectrogram_stft", cmap=cmap, dpi=dpi
    )

    # Mel
    results["mel"] = plot_mel_spectrogram(
        y, sr, n_fft, hop_length, n_mels=n_mels,
        output_path=f"{output_dir}/spectrogram_mel", cmap=cmap, dpi=dpi
    )

    # MFCC
    n_mfcc = config.get("n_mfcc", 13)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    results["mfcc"] = plot_mfcc(
        mfccs, sr, hop_length,
        output_path=f"{output_dir}/spectrogram_mfcc", cmap=cmap, dpi=dpi
    )

    return results
