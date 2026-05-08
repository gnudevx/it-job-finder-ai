"""
ExtractionService — extract text từ PDF dùng PyMuPDF (fitz).

Tại sao PyMuPDF thay vì pdfplumber hay pypdf?
  - Nhanh nhất trong các thư viện Python PDF
  - Giữ được thứ tự đọc tốt hơn (quan trọng với CV 2 cột)
  - Xử lý được nhiều encoding hơn
  - Có timeout built-in thông qua signal (tránh treo khi PDF corrupt)
"""

import fitz
import logging

from services.CV.validation.validation_service import FileValidationError

logger = logging.getLogger(__name__)


class ExtractionService:

    @staticmethod
    def extract_text_from_pdf(file_path: str) -> str:

        logger.info(
            "PDF extraction started",
            extra={
                "event": "pdf_extraction_started",
                "file": file_path,
            },
        )

        try:
            with fitz.open(file_path) as doc:

                page_count = doc.page_count

                if page_count == 0:
                    raise FileValidationError("PDF không có trang nào")

                if page_count > 50:

                    logger.warning(
                        "PDF has many pages",
                        extra={
                            "event": "pdf_many_pages",
                            "file": file_path,
                            "pages": page_count,
                        },
                    )

                pages_text = []

                for page_num in range(page_count):

                    page = doc[page_num]

                    text = page.get_text(
                        "text",
                        sort=True,
                    )

                    # fitz.page.get_text can return different types depending on
                    # params/versions; ensure we have a string before calling strip()
                    if isinstance(text, bytes):
                        try:
                            text = text.decode("utf-8")
                        except Exception:
                            text = str(text)

                    if not isinstance(text, str):
                        text = str(text)

                    if text.strip():
                        pages_text.append(text)

                if not pages_text:
                    raise FileValidationError(
                        "Không extract được text nào — PDF có thể là scan/image"
                    )

                full_text = "\n\n".join(pages_text)

                logger.info(
                    "PDF extraction completed",
                    extra={
                        "event": "pdf_extraction_completed",
                        "file": file_path,
                        "pages": page_count,
                        "characters_count": len(full_text),
                    },
                )

                return full_text

        except fitz.FileDataError as e:

            logger.warning(
                "Corrupted PDF file",
                extra={
                    "event": "corrupted_pdf",
                    "file": file_path,
                    "error": str(e),
                },
            )

            raise FileValidationError(
                f"PDF bị hỏng hoặc không đọc được: {e}"
            )

        except FileValidationError as e:

            logger.warning(
                "PDF extraction validation failed",
                extra={
                    "event": "pdf_extraction_validation_failed",
                    "file": file_path,
                    "error": str(e),
                },
            )

            raise

        except Exception as e:

            logger.exception(
                "Unexpected extraction error",
                extra={
                    "event": "pdf_extraction_failed",
                    "file": file_path,
                    "error": str(e),
                },
            )

            raise