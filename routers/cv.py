from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
import uuid
import logging

from core.dependencies import get_current_user, CurrentUser
from models.schemas import CVUploadResponse, CVStatusResponse

import os

from celery.result import AsyncResult
from workers.cv_worker import process_cv
from services.CV.storage.metadata_service import MetadataService

router = APIRouter()
logger = logging.getLogger(__name__)

# CV upload & processing endpoints. -- sektion này sẽ gọi Celery worker để xử lý background, tránh block request-response cycle. 
# với file upload, cần validate kỹ (MIME type, file size, filename) để tránh bị tấn công bằng file độc hại. 
# Sau khi upload, client sẽ poll endpoint /status/{cv_id} để biết tiến trình xử lý CV.
# Celery worker sẽ thực hiện các bước: extract text → clean → chunk → embed → store vào ChromaDB. 
# Trạng thái và metadata của CV sẽ lưu vào MongoDB để phục vụ cho việc RAG sau này.

# ── Constants ────────────────────────────────────────────────────────────────
MAX_FILE_SIZE = 5 * 1024 * 1024   # 5MB
# ALLOWED_MIME_TYPES = {"application/pdf"}
UPLOAD_DIR = "/tmp/cv_uploads"

@router.post("/upload", response_model=CVUploadResponse)
async def upload_cv(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    """
    Nhận file CV, validate, rồi đẩy vào Celery queue để xử lý background.
    Response ngay lập tức — client poll /status/{cv_id} để biết tiến trình.
    """
    cv_id = str(uuid.uuid4())
    try:
        # ── Bước 1: Validate file ─────────────────────────────────────────────
        # 1a. MIME type thực sự (không tin Content-Type header của client)
        content = await file.read(1024)   # đọc 1KB đầu để check magic bytes
        await file.seek(0)

        if not _is_valid_pdf(content):
            logger.warning(
                "Invalid file type",
                extra={"event": "upload_failed", "user_id": user.user_id, "file": file.filename},
            )
            raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF")

        # 1b. File size (đọc hết để đo, sau đó seek về 0)
        full_content = await file.read()
        if len(full_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File vượt quá 5MB")
        await file.seek(0)

        # 1c. Tên file — chống path traversal attack
        safe_filename = _sanitize_filename(file.filename or "cv.pdf")

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        stored_filename = f"{cv_id}.pdf"

        file_path = f"{UPLOAD_DIR}/{stored_filename}"

        with open(file_path, "wb") as f:
            f.write(full_content)

        logger.info(
            "Temporary file saved",
            extra={
                "event": "temp_file_saved",
                "cv_id": cv_id,
                "file": stored_filename,
            },
        )
        
        logger.info(
            "CV upload received",
            extra={
                "event": "cv_upload_received",
                "user_id": user.user_id,
                "cv_id": cv_id,
                "file": safe_filename,
                "size_bytes": len(full_content),
            },
        )
        
        metadata_service = MetadataService()

        metadata_service.update_status(
            cv_id=cv_id,
            status="uploaded",
        )
        logger.info(
            "Metadata inserted",
            extra={
                "event": "metadata_inserted",
                "cv_id": cv_id,
                "status": "uploaded",
            },
        )
        
        task: AsyncResult = process_cv.delay(  # type: ignore
            cv_id,
            user.user_id,
            stored_filename,
        )
        logger.info(
            "CV task enqueued",
            extra={
                "event": "cv_task_enqueued",
                "cv_id": cv_id,
                "task_id": task.id,
            },
        )
        
        # ── Bước 2: Lưu file tạm + đẩy Celery job ────────────────────────────
        # TODO: lưu file vào /tmp hoặc object storage
        # TODO: celery_app.send_task("workers.cv_worker.process_cv", args=[cv_id, user.user_id, safe_filename])

        return CVUploadResponse(
            cv_id=cv_id,
            filename=safe_filename,
            status="processing",
            message="CV đang được xử lý. Dùng GET /api/cv/status/{cv_id} để kiểm tra.",
        )
    except Exception as exc:
        logger.exception(
            "CV upload failed",
            extra={
                "event": "cv_upload_failed",
                "cv_id": cv_id,
                "user_id": user.user_id,
                "error": str(exc),
            },
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to upload CV",
        )


@router.get("/status/{cv_id}", response_model=CVStatusResponse)
async def cv_status(
    cv_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Client poll endpoint này để biết CV đã được embed xong chưa."""
    # TODO: query MongoDB để lấy trạng thái job
    return CVStatusResponse(cv_id=cv_id, status="processing")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_valid_pdf(header_bytes: bytes) -> bool:
    """PDF magic bytes bắt đầu bằng %PDF"""
    return header_bytes[:4] == b"%PDF"


def _sanitize_filename(filename: str) -> str:
    """Loại bỏ path traversal ký tự nguy hiểm."""
    import re
    # Chỉ giữ lại chữ, số, dấu chấm, gạch ngang, gạch dưới
    safe = re.sub(r"[^\w.\-]", "_", filename)
    # Không cho bắt đầu bằng dấu chấm (hidden file)
    return safe.lstrip(".")