import os
import logging
from typing import Optional

from core.config import settings

logger = logging.getLogger(__name__)


class FileStorageService:
    def __init__(self) -> None:
        self.backend = settings.STORAGE_BACKEND
        if self.backend == "s3":
            try:
                import boto3
                from botocore.client import Config
            except ImportError as exc:
                raise RuntimeError("Missing boto3 for S3 storage backend") from exc

            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.STORAGE_S3_REGION,
                endpoint_url=settings.STORAGE_S3_ENDPOINT_URL or None,
                config=Config(signature_version="s3v4"),
            )
            self.bucket = settings.STORAGE_S3_BUCKET
            if not self.bucket:
                raise ValueError("STORAGE_S3_BUCKET must be configured for S3 backend")
        else:
            self.local_dir = settings.STORAGE_LOCAL_UPLOAD_DIR
            os.makedirs(self.local_dir, exist_ok=True)

    def save_file(self, file_bytes: bytes, key: str) -> str:
        if self.backend == "s3":
            return self._save_file_s3(file_bytes, key)
        return self._save_file_local(file_bytes, key)

    def _save_file_local(self, file_bytes: bytes, key: str) -> str:
        path = os.path.join(self.local_dir, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(file_bytes)
        logger.info("Saved file locally", extra={"path": path})
        return path

    def _save_file_s3(self, file_bytes: bytes, key: str) -> str:
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=file_bytes)
        logger.info("Saved file to S3", extra={"bucket": self.bucket, "key": key})
        return key

    def download_to_temp(self, key: str) -> str:
        if self.backend == "s3":
            return self._download_to_temp_s3(key)
        return self._download_to_temp_local(key)

    def _download_to_temp_local(self, key: str) -> str:
        path = os.path.join(self.local_dir, key)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local file not found: {path}")
        return path

    def _download_to_temp_s3(self, key: str) -> str:
        from tempfile import NamedTemporaryFile

        temp_file = NamedTemporaryFile(delete=False, suffix=os.path.splitext(key)[1] or ".pdf")
        self.s3.download_fileobj(self.bucket, key, temp_file)
        temp_file.close()
        logger.info("Downloaded S3 file to temp", extra={"path": temp_file.name})
        return temp_file.name
