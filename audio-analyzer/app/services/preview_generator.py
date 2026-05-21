import os
import subprocess

from app.core.logger import get_logger
from app.services.r2_storage import R2Storage

logger = get_logger(__name__)


class PreviewGenerator:
    def __init__(self, storage: R2Storage) -> None:
        self.storage = storage

    def generate_and_upload(self, local_path: str, sound_id: int) -> str | None:
        preview_path = local_path.replace(os.path.splitext(local_path)[1], "_preview.mp3")
        cmd = [
            "ffmpeg", "-y",
            "-i", local_path,
            "-codec:a", "libmp3lame",
            "-q:a", "2",
            "-ar", "44100",
            "-ac", "2",
            preview_path,
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True, timeout=120)
        except subprocess.CalledProcessError as e:
            logger.error("preview_generation_failed", path=local_path, error=str(e))
            return None

        r2_key = f"sounds/preview/{sound_id}/preview.mp3"
        self.storage.upload(preview_path, r2_key, content_type="audio/mpeg")
        logger.info("preview_uploaded", sound_id=sound_id, r2_key=r2_key)

        os.remove(preview_path)
        return r2_key
