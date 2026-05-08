import pymupdf
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class ExtractionService:
    @staticmethod
    def extract_text_from_pdf(file_path: str, user_id: str = "", cv_id: str = "") -> str:
        file_name = Path(file_path).name
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        extra = {
            "event": "extraction_started",
            "file": file_name,
            "size_bytes": file_size,
        }
        if user_id:
            extra["user_id"] = user_id
        if cv_id:
            extra["cv_id"] = cv_id
            
        logger.info(f"Extracting text from PDF: {file_name}", extra=extra)
        
        try:
            document = pymupdf.open(file_path)
            pages = []

            for page in document:
                pages.append(page.get_text())

            document.close()
            
            extracted_text = "\n".join(pages)
            
            extra["event"] = "extraction_completed"
            extra["status"] = "success"
            logger.info(f"Successfully extracted {len(pages)} pages from {file_name}", extra=extra)
            
            return extracted_text
        except Exception as e:
            extra["event"] = "extraction_failed"
            extra["status"] = "error"
            extra["error"] = str(e)
            logger.error(f"Failed to extract PDF {file_name}: {str(e)}", extra=extra, exc_info=True)
            raise