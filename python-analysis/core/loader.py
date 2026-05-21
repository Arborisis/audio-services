from typing import Optional, Union
"""
Chargement audio pour WAV/MP3/FLAC via librosa/soundfile.
"""

import io
import mimetypes
import os

import librosa
import numpy as np
import soundfile as sf

from config import DEFAULT_SAMPLE_RATE

SUPPORTED_MIMETYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/x-flac",
    "audio/ogg",
    "audio/x-m4a",
    "audio/mp4",
}

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4"}


def get_mime_type(path: str) -> Optional[str]:
    """Détecte le MIME type d'un fichier audio."""
    mime, _ = mimetypes.guess_type(path)
    return mime


def is_supported_format(path: str) -> bool:
    """Vérifie si le format est supporté."""
    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_EXTENSIONS:
        return True
    mime = get_mime_type(path)
    return mime in SUPPORTED_MIMETYPES


def load_audio(path: Union[str, bytes], sr: Optional[int] = DEFAULT_SAMPLE_RATE, mono: bool = True) -> tuple[np.ndarray, int]:
    """
    Charge un fichier audio ou un flux de bytes.

    Args:
        path: Chemin fichier ou bytes/array numpy
        sr: Sample rate cible (None pour conserver l'original)
        mono: Convertir en mono

    Returns:
        (y, sr_loaded): Signal audio et sample rate
    """
    if isinstance(path, (bytes, bytearray)):
        return _load_from_bytes(path, sr=sr, mono=mono)

    if isinstance(path, np.ndarray):
        return path, sr or DEFAULT_SAMPLE_RATE

    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier audio introuvable: {path}")

    if not is_supported_format(path):
        mime = get_mime_type(path)
        raise ValueError(f"Format audio non supporté: {mime or 'inconnu'} ({path})")

    y, sr_loaded = librosa.load(path, sr=sr, mono=mono)
    return y, sr_loaded


def _load_from_bytes(data: bytes, sr: Optional[int], mono: bool) -> tuple[np.ndarray, int]:
    """Charge depuis un buffer bytes (streaming)."""
    buffer = io.BytesIO(data)
    y, sr_loaded = librosa.load(buffer, sr=sr, mono=mono)
    return y, sr_loaded


def get_audio_info(path: str) -> dict:
    """Récupère les métadonnées d'un fichier audio sans le charger entièrement."""
    info = sf.info(path)
    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
    }
