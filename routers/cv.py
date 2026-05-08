"""
routers/cv.py — fixed version.
Fix: except Exception không còn nuốt HTTPException (400, 422...)
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
import uuid
import logging
import os

from celery.result import AsyncResult
from workers.cv_worker import process_cv
from services.CV.storage.metadata_service import MetadataService
from core.dependencies import get_current_user, CurrentUser
from models.schemas import CVUploadResponse, CVStatusResponse

router = APIRouter()
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 5 * 1024 * 1024
UPLOAD_DIR = "/tmp/cv_uploads"


@router.post("/upload", response_model=CVUploadResponse)
async def upload_cv(
    file: UploadFile = File(...),
    user: CurrentUser = Depends(get_current_user),
):
    cv_id = str(uuid.uuid4())

    # ── Validate ──────────────────────────────────────────────────────────────
    content = await file.read(1024)
    await file.seek(0)

    if not _is_valid_pdf(content):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file PDF")

    full_content = await file.read()
    if len(full_content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File vượt quá 5MB")
    await file.seek(0)

    safe_filename = _sanitize_filename(file.filename or "cv.pdf")

    # ── Lưu file + enqueue ────────────────────────────────────────────────────
    # HTTPException đi thẳng ra ngoài, Exception khác → 500
    try:
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        stored_filename = f"{cv_id}.pdf"
        file_path = f"{UPLOAD_DIR}/{stored_filename}"

        with open(file_path, "wb") as f:
            f.write(full_content)
        
        logger.info(f"File saved: {file_path}")

        # ← DEBUG: kiểm tra metadata lưu thành công không
        try:
            MetadataService().update_status(
                cv_id=cv_id,
                user_id=user.user_id.strip(),
                filename=safe_filename,
                status="uploaded",
            )
            logger.info(f"✅ Metadata saved for cv_id={cv_id}, user_id={user.user_id}")
        except Exception as db_err:
            logger.exception(f"❌ Metadata save failed: {db_err}")
            raise

        task: AsyncResult = process_cv.delay( # type: ignore
            cv_id, 
            user.user_id, 
            stored_filename
        )

        logger.info(
            "CV enqueued",
            extra={"event": "cv_enqueued", "cv_id": cv_id, "task_id": task.id},
        )

    except HTTPException:
        raise   # ← fix quan trọng: 400/422 không bị convert thành 500
    except Exception as exc:
        logger.exception("CV upload failed", extra={"cv_id": cv_id})
        raise HTTPException(status_code=500, detail="Failed to upload CV")

    return CVUploadResponse(
        cv_id=cv_id,
        filename=safe_filename,
        status="processing",
        message=f"CV đang xử lý. Poll GET /api/cv/status/{cv_id}",
    )


@router.get("/status/{cv_id}", response_model=CVStatusResponse)
async def cv_status(
    cv_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Query MongoDB để lấy trạng thái thực tế."""
    doc = MetadataService().get_status(cv_id)
    
    logger.info(f"CV status query: cv_id={cv_id}, user_id={user.user_id}, doc={doc}")

    if not doc:
        raise HTTPException(status_code=404, detail="CV không tồn tại")

    # Chỉ cho phép user xem CV của chính họ
    if doc.get("user_id", "").strip() != user.user_id.strip():
        logger.warning(
            f"🚫 Access denied: doc.user_id={doc.get('user_id')}, "
            f"request.user_id={user.user_id}"
        )
        raise HTTPException(status_code=403, detail="Không có quyền truy cập")

    return CVStatusResponse(
        cv_id=cv_id,
        status=doc["status"],
        chunks_count=doc.get("chunks_count"),
        uploaded_at=doc.get("uploaded_at"),
    )


def _is_valid_pdf(header_bytes: bytes) -> bool:
    return header_bytes[:4] == b"%PDF"


def _sanitize_filename(filename: str) -> str:
    import re
    safe = re.sub(r"[^\w.\-]", "_", filename)
    return safe.lstrip(".")