import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualizations.spectrograms import plot_stft, plot_mel_spectrogram, plot_mfcc
from visualizations.heatmaps import plot_feature_correlation
from core.utils import ensure_dir


class TestVisualizations:
    def setup_method(self):
        self.output_dir = "/tmp/arborisis_test_viz"
        ensure_dir(self.output_dir)
        self.y = np.random.randn(22050) * 0.1
        self.sr = 22050

    def teardown_method(self):
        import shutil
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def test_plot_stft_creates_file(self):
        path = f"{self.output_dir}/stft.png"
        result = plot_stft(self.y, self.sr, output_path=path)
        assert result is not None
        assert os.path.exists(result)

    def test_plot_mel_spectrogram_creates_file(self):
        path = f"{self.output_dir}/mel.png"
        result = plot_mel_spectrogram(self.y, self.sr, output_path=path)
        assert result is not None
        assert os.path.exists(result)

    def test_plot_mfcc_creates_file(self):
        import librosa
        mfccs = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        path = f"{self.output_dir}/mfcc.png"
        result = plot_mfcc(mfccs, self.sr, output_path=path)
        assert result is not None
        assert os.path.exists(result)

    def test_plot_feature_correlation_creates_file(self):
        import pandas as pd
        df = pd.DataFrame({
            "zcr": np.random.rand(100),
            "rms": np.random.rand(100),
            "centroid": np.random.rand(100),
        })
        path = f"{self.output_dir}/corr.png"
        result = plot_feature_correlation(df, output_path=path)
        assert result is not None
        assert os.path.exists(result)
