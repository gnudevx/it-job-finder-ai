"""
ValidationService — kiểm tra file trước khi đưa vào pipeline.

Tại sao validate ở đây thay vì chỉ ở router?
  - Router validate khi nhận request (HTTP layer)
  - Worker validate lại khi xử lý (vì file có thể bị corrupt sau khi lưu)
  - 2 lớp bảo vệ = an toàn hơn
"""

import os
import logging

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 5 * 1024 * 1024   # 5MB
MIN_FILE_SIZE = 100                # 100 bytes — PDF rỗng thì vô nghĩa


class FileValidationError(Exception):
    """Raise khi file không hợp lệ — Celery sẽ KHÔNG retry lỗi này."""
    pass


class FileValidationService:

    @staticmethod
    def validate_file(file_path: str) -> None:
        """
        Validate file tại đường dẫn cho trước.
        Raise FileValidationError nếu không hợp lệ.
        """
        # 1. File tồn tại không?
        if not os.path.exists(file_path):
            raise FileValidationError(f"File không tồn tại: {file_path}")

        # 2. Kiểm tra size
        size = os.path.getsize(file_path)
        if size < MIN_FILE_SIZE:
            raise FileValidationError(f"File quá nhỏ ({size} bytes) — có thể bị rỗng")
        if size > MAX_FILE_SIZE:
            raise FileValidationError(f"File vượt quá 5MB ({size} bytes)")

        # 3. Kiểm tra magic bytes — đọc 4 byte đầu
        # Không tin vào extension .pdf hay Content-Type header
        with open(file_path, "rb") as f:
            header = f.read(4)

        if header[:4] != b"%PDF":
            raise FileValidationError(
                f"File không phải PDF hợp lệ (magic bytes: {header[:4].hex()})"
            )

        logger.info(
            "File validation passed",
            extra={"event": "file_validated", "file": file_path, "size_bytes": size},
        )