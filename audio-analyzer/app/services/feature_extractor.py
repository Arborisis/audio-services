import json
import os
from typing import Any

import librosa
import numpy as np

from app.core.logger import get_logger
from app.services.r2_storage import R2Storage

logger = get_logger(__name__)

# Target sample rate for analysis — reduces data size and speeds up processing
TARGET_SR = 22050
HOP_LENGTH = 2048
N_FFT = 2048


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


class FeatureExtractor:
    def __init__(self, storage: R2Storage) -> None:
        self.storage = storage

    def extract_and_upload(self, local_path: str, sound_id: int, metadata: dict[str, Any]) -> tuple[dict[str, Any], str]:
        # Load at target SR to reduce data size and speed up all downstream processing
        y, sr = librosa.load(local_path, sr=TARGET_SR, mono=True)
        duration = metadata.get("duration_seconds", librosa.get_duration(y=y, sr=sr))
        hop_length = HOP_LENGTH
        n_fft = N_FFT

        # ── Pre-compute STFT (reused for spectral features) ──
        stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
        freqs_stft = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # ── Temporal features ──
        rms = librosa.feature.rms(S=stft, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=hop_length)[0]

        # ── Spectral features (from STFT) ──
        spectral_centroids = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
        spectral_rolloffs = librosa.feature.spectral_rolloff(S=stft, sr=sr)[0]
        spectral_bandwidths = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
        spectral_flatnesses = librosa.feature.spectral_flatness(S=stft)[0]

        # ── MFCC (fast, from STFT) ──
        mfccs = librosa.feature.mfcc(S=librosa.power_to_db(stft), sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfccs, width=3)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2, width=3)

        # ── Chroma (fast, from STFT) ──
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)

        # ── Heavy features only for shorter files (< 90s) to avoid CPU saturation ──
        compute_heavy = duration < 90
        if compute_heavy:
            contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
            harmonic = librosa.effects.harmonic(y, margin=8)
            tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        else:
            contrast = np.zeros((7, stft.shape[1]))
            tonnetz = np.zeros((6, stft.shape[1]))
            logger.info("heavy_features_skipped", sound_id=sound_id, duration=duration)

        # ── Tempo & beats ──
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
        tempo_float = float(tempo) if isinstance(tempo, (int, float, np.number)) else float(tempo.item() if hasattr(tempo, 'item') else 0)

        # ── Harmonic / Percussive ──
        y_harmonic, y_percussive = librosa.effects.hpss(y, margin=8)
        harmonic_energy = float(np.sum(y_harmonic ** 2))
        percussive_energy = float(np.sum(y_percussive ** 2))
        total_energy = harmonic_energy + percussive_energy + 1e-10
        harmonic_ratio = round(harmonic_energy / total_energy, 4)
        percussive_ratio = round(percussive_energy / total_energy, 4)

        # ── Dominant frequencies (top 5 from mean spectrum) ──
        spec = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), 1 / sr)
        peak_indices = np.argsort(spec)[-5:][::-1]
        dominant_frequencies = [int(freqs[i]) for i in peak_indices if freqs[i] > 0]

        # ── Energy in frequency bands ──
        total_spec_energy = np.sum(stft ** 2) + 1e-10
        low_energy = float(np.sum(stft[freqs_stft < 500] ** 2)) / total_spec_energy
        mid_energy = float(np.sum(stft[(freqs_stft >= 500) & (freqs_stft < 4000)] ** 2)) / total_spec_energy
        high_energy = float(np.sum(stft[freqs_stft >= 4000] ** 2)) / total_spec_energy

        def _band_ratio(low: int, high: int) -> float:
            mask = (freqs_stft >= low) & (freqs_stft < high)
            return float(np.sum(stft[mask] ** 2)) / total_spec_energy

        sub_bass_energy = _band_ratio(20, 120)
        bass_energy = _band_ratio(120, 500)
        low_mid_energy = _band_ratio(500, 1500)
        presence_energy = _band_ratio(1500, 4000)
        bird_band_energy = _band_ratio(2000, 8000)
        insect_band_energy = _band_ratio(6000, 11000)

        # Spectral flux captures rapid acoustic changes such as birdsong syllables
        # without storing frame-level data.
        stft_norm = stft / (np.sum(stft, axis=0, keepdims=True) + 1e-10)
        spectral_flux = np.sqrt(np.sum(np.diff(stft_norm, axis=1) ** 2, axis=0))

        # ── Peak / RMS / Noise floor (dB) ──
        peak = float(np.max(np.abs(y)))
        peak_db = round(20 * np.log10(peak + 1e-10), 2) if peak > 0 else -120.0
        rms_mean = float(np.mean(rms))
        rms_db = round(20 * np.log10(rms_mean + 1e-10), 2) if rms_mean > 0 else -120.0
        rms_std = float(np.std(rms))

        noise_floor = float(np.percentile(rms, 10))
        noise_floor_db = round(20 * np.log10(noise_floor + 1e-10), 2) if noise_floor > 0 else -120.0
        loudness_lufs = round(rms_db - 14.0, 2)

        # ── Dynamic range & crest factor ──
        dynamic_range_db = round(peak_db - noise_floor_db, 2)
        crest_factor = round(peak / (rms_mean + 1e-10), 2)

        # ── Silence ratio ──
        silence_threshold = rms_mean * 0.1
        silence_frames = np.sum(rms < silence_threshold)
        silence_ratio = round(float(silence_frames / len(rms)), 4)

        # ── Event density ──
        rms_norm = rms / (np.max(rms) + 1e-10)
        burst_threshold = 0.3
        above = rms_norm > burst_threshold
        bursts = np.diff(above.astype(int))
        event_count = int(np.sum(bursts == 1))
        event_density = round(event_count / duration, 2) if duration > 0 else 0

        # ── Spectral entropy ──
        mean_spec = np.mean(stft, axis=1)
        mean_spec = mean_spec / (np.sum(mean_spec) + 1e-10)
        spectral_entropy = round(float(-np.sum(mean_spec * np.log2(mean_spec + 1e-10))), 4)

        # ── Periodicity ──
        # Full autocorrelation is O(n^2) and can stall workers on multi-minute files.
        if duration <= 30:
            autocorr = np.correlate(y, y, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            periodicity = round(float(np.max(autocorr[1:min(len(autocorr), int(sr))])), 4)
        else:
            periodicity = 0.0
            logger.info("periodicity_skipped", sound_id=sound_id, duration=duration)

        def _stats(arr: np.ndarray) -> dict[str, float]:
            return {
                "mean": round(float(np.mean(arr)), 4),
                "std": round(float(np.std(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
            }

        features = {
            "duration_seconds": round(duration, 2),
            "sample_rate": metadata.get("sample_rate", sr),
            "channels": metadata.get("channels", 1),
            "format": metadata.get("format", ""),
            "bitrate": metadata.get("bitrate"),
            "loudness_lufs": loudness_lufs,
            "peak_db": peak_db,
            "rms_db": rms_db,
            "rms_std": round(rms_std, 4),
            "noise_floor_db": noise_floor_db,
            "dynamic_range_db": dynamic_range_db,
            "crest_factor": crest_factor,
            "silence_ratio": silence_ratio,
            "event_density": event_density,
            "zero_crossing_rate": round(float(np.mean(zcr)), 4),
            "zcr_std": round(float(np.std(zcr)), 4),
            "spectral_centroid": round(float(np.mean(spectral_centroids)), 2),
            "spectral_rolloff": round(float(np.mean(spectral_rolloffs)), 2),
            "spectral_bandwidth": round(float(np.mean(spectral_bandwidths)), 2),
            "spectral_flatness": round(float(np.mean(spectral_flatnesses)), 4),
            "spectral_entropy": spectral_entropy,
            "dominant_frequencies": dominant_frequencies,
            "low_freq_ratio": round(low_energy, 4),
            "mid_freq_ratio": round(mid_energy, 4),
            "high_freq_ratio": round(high_energy, 4),
            "sub_bass_ratio": round(sub_bass_energy, 4),
            "bass_ratio": round(bass_energy, 4),
            "low_mid_ratio": round(low_mid_energy, 4),
            "presence_ratio": round(presence_energy, 4),
            "bird_band_ratio": round(bird_band_energy, 4),
            "insect_band_ratio": round(insect_band_energy, 4),
            "spectral_flux": round(float(np.mean(spectral_flux)), 4) if spectral_flux.size else 0.0,
            "spectral_flux_std": round(float(np.std(spectral_flux)), 4) if spectral_flux.size else 0.0,
            "tempo_bpm": round(tempo_float, 1),
            "beat_count": len(beat_frames),
            "harmonic_ratio": harmonic_ratio,
            "percussive_ratio": percussive_ratio,
            "periodicity": periodicity,
            "mfcc": {f"mfcc_{i+1}": _stats(mfccs[i]) for i in range(13)},
            "chroma": {f"chroma_{i+1}": _stats(chroma[i]) for i in range(12)},
            "spectral_contrast": {f"band_{i}": _stats(contrast[i]) for i in range(contrast.shape[0])},
            "tonnetz": {f"tonnetz_{i+1}": _stats(tonnetz[i]) for i in range(tonnetz.shape[0])},
        }
        features = _json_safe(features)

        temp_json = local_path.replace(os.path.splitext(local_path)[1], "_features.json")
        with open(temp_json, "w") as f:
            json.dump(features, f, indent=2)

        r2_key = f"sounds/analysis/{sound_id}/features.json"
        self.storage.upload(temp_json, r2_key, content_type="application/json")
        os.remove(temp_json)

        logger.info(
            "features_extracted",
            sound_id=sound_id,
            r2_key=r2_key,
            duration=features["duration_seconds"],
            tempo=features["tempo_bpm"],
            events=features["event_density"],
            heavy=compute_heavy,
        )
        return features, r2_key
