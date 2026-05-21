import os
import uuid

from app.config import settings
from app.core.exceptions import DownloadError
from app.core.logger import get_logger
from app.services.r2_storage import R2Storage

logger = get_logger(__name__)


class AudioDownloader:
    def __init__(self, storage: R2Storage) -> None:
        self.storage = storage

    def download(self, r2_key: str) -> str:
        temp_dir = f"/tmp/analyzer/{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)

        filename = os.path.basename(r2_key)
        local_path = os.path.join(temp_dir, filename)

        self.storage.download(r2_key, local_path)

        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        if size_mb > settings.max_file_size_mb:
            os.remove(local_path)
            raise DownloadError(f"File too large: {size_mb:.1f}MB > {settings.max_file_size_mb}MB")

        logger.info("audio_downloaded", r2_key=r2_key, local_path=local_path, size_mb=round(size_mb, 2))
        return local_path
