import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.loader import load_audio, is_supported_format, get_audio_info
from core.preprocessing import normalize, trim_silence, apply_filter
from core.segmentation import segment_frames, segment_onsets
from features.temporal import extract_zcr, extract_rms, extract_temporal_features
from features.spectral import extract_spectral_features
from features.cepstral import extract_mfcc, extract_cepstral_features


class TestLoader:
    def test_is_supported_format_wav(self):
        assert is_supported_format("audio.wav") is True

    def test_is_supported_format_mp3(self):
        assert is_supported_format("audio.mp3") is True

    def test_is_supported_format_txt(self):
        assert is_supported_format("audio.txt") is False


class TestPreprocessing:
    def test_normalize_peak(self):
        y = np.array([0.5, -0.3, 0.8])
        result = normalize(y, method="peak")
        assert np.max(np.abs(result)) == pytest.approx(1.0)

    def test_normalize_rms(self):
        y = np.array([0.5, -0.3, 0.8])
        result = normalize(y, method="rms")
        rms = np.sqrt(np.mean(result ** 2))
        assert rms == pytest.approx(1.0)

    def test_trim_silence(self):
        y = np.concatenate([np.zeros(1000), np.ones(500) * 0.5, np.zeros(1000)])
        trimmed, _ = trim_silence(y, sr=22050, top_db=20)
        assert len(trimmed) < len(y)

    def test_apply_lowpass_filter(self):
        sr = 22050
        t = np.linspace(0, 1, sr)
        y = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 5000 * t)
        filtered = apply_filter(y, sr, "lowpass", 2000)
        assert len(filtered) == len(y)


class TestSegmentation:
    def test_segment_frames(self):
        y = np.random.randn(22050)
        frames = segment_frames(y, sr=22050, frame_length=2048, hop_length=512)
        assert frames.ndim == 2
        assert frames.shape[1] == 2048


class TestTemporalFeatures:
    def test_extract_zcr_shape(self):
        y = np.random.randn(22050)
        zcr = extract_zcr(y, frame_length=2048, hop_length=512)
        assert len(zcr) > 0

    def test_extract_rms_shape(self):
        y = np.random.randn(22050)
        rms = extract_rms(y, frame_length=2048, hop_length=512)
        assert len(rms) > 0
        assert np.all(rms >= 0)

    def test_extract_temporal_features(self):
        y = np.random.randn(22050)
        features = extract_temporal_features(y, sr=22050)
        assert "zcr" in features
        assert "rms" in features
        assert "duration_seconds" in features
        assert features["duration_seconds"] == pytest.approx(1.0)


class TestSpectralFeatures:
    def test_extract_spectral_features(self):
        y = np.random.randn(22050) * 0.1
        features = extract_spectral_features(y, sr=22050, n_fft=2048, hop_length=512)
        assert "centroid" in features
        assert "bandwidth" in features
        assert "rolloff" in features
        assert "flatness" in features
        assert "contrast" in features


class TestCepstralFeatures:
    def test_extract_mfcc_shape(self):
        y = np.random.randn(22050) * 0.1
        mfcc = extract_mfcc(y, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512)
        assert mfcc.shape[0] == 13

    def test_extract_cepstral_features(self):
        y = np.random.randn(22050) * 0.1
        features = extract_cepstral_features(y, sr=22050, n_mfcc=13)
        assert "mfcc" in features
        assert "delta" in features
        assert "delta2" in features
