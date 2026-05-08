import os
import mimetypes
import logging

from pathlib import Path

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 5
ALLOWED_EXTENSIONS = {".pdf"}


class FileValidationError(Exception):
    pass


class FileValidationService:

    @staticmethod
    def validate_file(file_path: str):

        logger.info(
            "File validation started",
            extra={
                "event": "file_validation_started",
                "file": file_path,
            },
        )

        path = Path(file_path)

        # check exists
        if not path.exists():

            logger.warning(
                "File does not exist",
                extra={
                    "event": "file_not_found",
                    "file": file_path,
                },
            )

            raise FileValidationError("File does not exist")

        # extension check
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:

            logger.warning(
                "Invalid file extension",
                extra={
                    "event": "invalid_extension",
                    "file": file_path,
                },
            )

            raise FileValidationError("Only PDF files are allowed")

        # mime type check
        mime_type, _ = mimetypes.guess_type(file_path)

        if mime_type != "application/pdf":

            logger.warning(
                "Invalid MIME type",
                extra={
                    "event": "invalid_mime_type",
                    "file": file_path,
                },
            )

            raise FileValidationError("Invalid MIME type")

        # size check
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:

            logger.warning(
                "File too large",
                extra={
                    "event": "file_too_large",
                    "file": file_path,
                    "size_mb": round(file_size_mb, 2),
                },
            )

            raise FileValidationError("File exceeds 5MB limit")

        logger.info(
            "File validation passed",
            extra={
                "event": "file_validation_passed",
                "file": file_path,
                "size_mb": round(file_size_mb, 2),
            },
        )

        return True