from workers.celery_app import celery_app
import logging

logger = logging.getLogger(__name__)

# đây là file dùng để định nghĩa các Celery task chạy background, tránh block request-response cycle khi xử lý CV.

from services.CV.validation.validation_service import FileValidationService
from services.CV.extraction.extraction_service import ExtractionService
from services.CV.processing.cleaning_service import CleaningService
from services.CV.processing.chunking_service import ChunkingService
from services.CV.processing.embedding_service import EmbeddingService
from services.CV.storage.vector_service import VectorService
from services.CV.storage.metadata_service import MetadataService

@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,  # retry sau 30s nếu fail
    name="workers.cv_worker.process_cv",
)
def process_cv(self, cv_id: str, user_id: str, filename: str):
    """
    Background job: xử lý CV sau khi upload.
    Luồng: Extract → Clean → Chunk → Embed → Store

    Gọi từ router:
        process_cv.delay(cv_id, user.user_id, safe_filename)
    """
        # TODO Bước 1: đọc file từ /tmp/cv_uploads/{cv_id}.pdf
        # TODO Bước 2: PyMuPDF extract text
        # TODO Bước 3: clean / normalize text
        # TODO Bước 4: chunking với overlap
        # TODO Bước 5: embedding từng chunk
        # TODO Bước 6: lưu vào ChromaDB
        # TODO Bước 7: lưu metadata vào MongoDB (status="done", chunks_count=N)
    try:
        logger.info(
            "CV processing started",
            extra={
                "event": "cv_processing_start",
                "cv_id": cv_id,
                "user_id": user_id,
            },
        )

        file_path = f"/tmp/cv_uploads/{filename}"

        metadata_service = MetadataService()
        metadata_service.update_status(cv_id, "processing")

        # validate
        FileValidationService.validate_file(file_path)

        # extract
        raw_text = ExtractionService.extract_text_from_pdf(file_path)

        # clean
        cleaned_text = CleaningService.clean_text(raw_text)

        # chunk
        chunk_service = ChunkingService()
        chunks = chunk_service.chunk_document(cleaned_text)

        # embed
        embedding_service = EmbeddingService()
        embeddings = embedding_service.embed_chunks(chunks)

        # store vectors
        vector_service = VectorService()

        vector_service.delete_existing_cv(cv_id)

        vector_service.store_embeddings(
            chunks=chunks,
            embeddings=embeddings,
            user_id=user_id,
            cv_id=cv_id,
            filename=filename,
        )

        # update metadata
        metadata_service.update_status(
            cv_id=cv_id,
            status="done",
            chunks_count=len(chunks),
        )

        logger.info(
            "CV processing done",
            extra={
                "event": "cv_processing_done",
                "cv_id": cv_id,
                "chunks_count": len(chunks),
            },
        )
    except Exception as exc:
        logger.exception(
            "CV processing failed",
            extra={
                "event": "cv_processing_failed",
                "cv_id": cv_id,
                "error": str(exc),
            },
        )
    
        # Celery tự retry theo cấu hình max_retries
        raise self.retry(exc=exc)