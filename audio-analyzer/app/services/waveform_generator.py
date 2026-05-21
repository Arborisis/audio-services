import json
import os

import librosa
import numpy as np

from app.core.logger import get_logger
from app.services.r2_storage import R2Storage

logger = get_logger(__name__)

DEFAULT_RESOLUTION = 1000


class WaveformGenerator:
    def __init__(self, storage: R2Storage) -> None:
        self.storage = storage

    def generate_and_upload(self, local_path: str, sound_id: int, duration: float) -> str:
        y, sr = librosa.load(local_path, sr=None, mono=True)

        target_resolution = DEFAULT_RESOLUTION
        samples_per_bin = max(1, len(y) // target_resolution)
        bins = len(y) // samples_per_bin

        waveform = []
        for i in range(bins):
            chunk = y[i * samples_per_bin:(i + 1) * samples_per_bin]
            waveform.append(float(np.mean(np.abs(chunk))))

        output = {
            "samples": waveform,
            "resolution": len(waveform),
            "duration_seconds": round(duration, 2),
        }

        temp_json = local_path.replace(os.path.splitext(local_path)[1], "_waveform.json")
        with open(temp_json, "w") as f:
            json.dump(output, f)

        r2_key = f"sounds/analysis/{sound_id}/waveform.json"
        self.storage.upload(temp_json, r2_key, content_type="application/json")
        os.remove(temp_json)

        logger.info("waveform_generated", sound_id=sound_id, r2_key=r2_key, resolution=len(waveform))
        return r2_key
