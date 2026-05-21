import os

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from app.core.logger import get_logger
from app.services.r2_storage import R2Storage

logger = get_logger(__name__)

FIG_WIDTH = 12
FIG_HEIGHT = 6
DPI = 100
QUALITY = 85


class SpectrogramGenerator:
    def __init__(self, storage: R2Storage) -> None:
        self.storage = storage

    def generate_and_upload(self, local_path: str, sound_id: int) -> str:
        y, sr = librosa.load(local_path, sr=22050, mono=True)

        fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
        fig.patch.set_facecolor("#0a0a0a")
        ax.set_facecolor("#0a0a0a")

        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(
            D,
            sr=sr,
            x_axis=None,
            y_axis=None,
            cmap="magma",
            ax=ax,
        )

        ax.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        temp_png = local_path.replace(os.path.splitext(local_path)[1], "_spectrogram.png")
        fig.savefig(temp_png, dpi=DPI, facecolor="#0a0a0a", edgecolor="none")
        plt.close(fig)

        temp_webp = temp_png.replace(".png", ".webp")
        with Image.open(temp_png) as im:
            im.save(temp_webp, "WEBP", quality=QUALITY)

        os.remove(temp_png)

        r2_key = f"sounds/analysis/{sound_id}/spectrogram.webp"
        self.storage.upload(temp_webp, r2_key, content_type="image/webp")
        os.remove(temp_webp)

        logger.info("spectrogram_generated", sound_id=sound_id, r2_key=r2_key)
        return r2_key
