import librosa
import numpy as np

from app.core.logger import get_logger

logger = get_logger(__name__)


class QualityAnalyzer:
    def analyze(self, y: np.ndarray, sr: int, features: dict) -> dict:
        peak = np.max(np.abs(y))
        rms = np.mean(librosa.feature.rms(y=y)[0])

        clipping_detected = bool(peak >= 0.99)
        silence_threshold = 0.005
        silent_frames = np.sum(librosa.feature.rms(y=y)[0] < silence_threshold)
        total_frames = len(librosa.feature.rms(y=y)[0])
        silence_ratio = round(float(silent_frames / max(total_frames, 1)), 4)

        noise_floor_db = features.get("noise_floor_db", -60)
        rms_db = features.get("rms_db", -30)

        usable = True
        label = "good"

        if features.get("duration_seconds", 0) < 3:
            usable = False
            label = "unusable"
        elif clipping_detected and rms_db > -3:
            usable = False
            label = "unusable"
        elif noise_floor_db > -30:
            label = "poor"
            usable = False
        elif noise_floor_db > -40:
            label = "medium"
        elif noise_floor_db > -50 and not clipping_detected:
            label = "good"
        elif not clipping_detected and silence_ratio < 0.3:
            label = "excellent"

        return {
            "clipping_detected": clipping_detected,
            "noise_floor_db": noise_floor_db,
            "signal_level": rms_db,
            "silence_ratio": silence_ratio,
            "usable_for_analysis": usable,
            "quality_label": label,
        }
