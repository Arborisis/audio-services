import os
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from app.config import settings
from app.core.exceptions import InvalidR2KeyError, DownloadError
from app.core.logger import get_logger

logger = get_logger(__name__)

_AUDIO_EXTENSIONS = {"wav", "mp3", "flac", "ogg", "m4a", "aac", "wma", "webm"}


class R2Storage:
    def __init__(self) -> None:
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name=settings.r2_region,
        )
        self.bucket = settings.r2_bucket

    def _validate_key(self, key: str, allowed_prefixes: tuple[str, ...] = ("sounds/",)) -> None:
        normalized = key.replace("\\", "/")
        if ".." in normalized:
            raise InvalidR2KeyError("Path traversal detected.")
        if not any(normalized.startswith(p) for p in allowed_prefixes):
            raise InvalidR2KeyError(f"Key must start with one of {allowed_prefixes}.")

    def download(self, key: str, local_path: str) -> None:
        self._validate_key(key, allowed_prefixes=("sounds/original/",))
        try:
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            self.client.download_file(self.bucket, key, local_path)
            logger.info("r2_download_complete", r2_key=key, local_path=local_path)
        except ClientError as e:
            logger.error("r2_download_failed", r2_key=key, error=str(e))
            raise DownloadError(f"Failed to download {key}: {e}")

    def upload(self, local_path: str, key: str, content_type: Optional[str] = None) -> None:
        self._validate_key(key, allowed_prefixes=("sounds/",))
        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type
        try:
            self.client.upload_file(local_path, self.bucket, key, ExtraArgs=extra_args)
            logger.info("r2_upload_complete", r2_key=key, local_path=local_path)
        except ClientError as e:
            logger.error("r2_upload_failed", r2_key=key, error=str(e))
            raise DownloadError(f"Failed to upload {key}: {e}")

    def key_exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
