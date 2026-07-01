"""
cv_worker.py — Celery background task xử lý CV sau khi upload.

Luồng: validate → extract → clean → chunk → embed → store vector → update metadata
"""

# from workers.celery_app import celery_app   # Không dùng Celery nữa
import logging

logger = logging.getLogger(__name__)

from services.CV.validation.validation_service import FileValidationService, FileValidationError
from services.CV.extraction.extraction_service import ExtractionService
from services.CV.processing.cleaning_service import CleaningService
from services.CV.processing.chunking_service import ChunkingService
from services.CV.processing.embedding_service import EmbeddingService
from services.CV.storage.vector_service import VectorService
from services.CV.storage.metadata_service import MetadataService


def process_cv(cv_id: str, user_id: str, filename: str):
    """
    Background task: xử lý CV sau khi upload bằng FastAPI BackgroundTasks.
    """
    from services.CV.storage.storage_service import FileStorageService

    storage_service = FileStorageService()
    file_path = storage_service.download_to_temp(filename)
    metadata_service = MetadataService()

    try:
        logger.info(
            "CV processing started",
            extra={"event": "cv_processing_start", "cv_id": cv_id, "user_id": user_id},
        )

        metadata_service.update_status(cv_id, "processing", user_id=user_id)

        # Bước 1: validate file (lần 2 — file có thể corrupt sau khi lưu)
        FileValidationService.validate_file(file_path)

        # Bước 2: extract text từ PDF
        raw_text = ExtractionService.extract_text_from_pdf(file_path)

        # Bước 3: clean + sanitize text
        cleaned_text = CleaningService.clean_text(raw_text)

        # Bước 4: chunk với overlap
        chunk_service = ChunkingService()
        chunks = chunk_service.chunk_document(cleaned_text)
        logger.info(f"Generated {len(chunks)} chunks")

        for chunk in chunks[:3]:
            logger.info(
                f"Chunk #{chunk.chunk_index}: "
                f"{chunk.text[:200]}"
            )
        # Bước 5: embed từng chunk
        embedding_service = EmbeddingService()
        embeddings = embedding_service.embed_chunks(chunks)

        # Bước 6: xóa vectors cũ rồi lưu mới (tránh duplicate khi re-upload)
        vector_service = VectorService()
        vector_service.delete_existing_cv(cv_id)
        vector_service.store_embeddings(
            chunks=chunks,
            embeddings=embeddings,
            user_id=user_id,
            cv_id=cv_id,
            filename=filename,
        )

        # Bước 7: cập nhật metadata → done
        metadata_service.update_status(
            cv_id=cv_id,
            status="done",
            user_id=user_id,
            chunks_count=len(chunks),
        )

        logger.info(
            "CV processing done",
            extra={"event": "cv_processing_done", "cv_id": cv_id, "chunks_count": len(chunks)},
        )

        return {"cv_id": cv_id, "status": "done", "chunks": len(chunks)}

    except FileValidationError as exc:
        logger.warning(
            "CV validation failed — not retrying",
            extra={"event": "cv_validation_failed", "cv_id": cv_id, "error": str(exc)},
        )
        metadata_service.update_status(
            cv_id=cv_id,
            status="failed",
            user_id=user_id,
            error_message=str(exc),
        )

    except Exception as exc:
        logger.exception(
            "CV processing failed",
            extra={"event": "cv_processing_failed", "cv_id": cv_id, "error": str(exc)},
        )
        metadata_service.update_status(
            cv_id=cv_id,
            status="failed",
            user_id=user_id,
            error_message=str(exc),
        )
