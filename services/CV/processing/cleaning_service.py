"""
CleaningService — normalize raw text sau khi extract từ PDF.

Mục đích: text từ PDF thường có rác (header/footer lặp lại,
ký tự đặc biệt, encoding lỗi...). Clean trước khi chunk
giúp embedding chính xác hơn.
"""

import re
import unicodedata
import logging
from services.CV.validation.text_sanitizer_service import TextSanitizerService

logger = logging.getLogger(__name__)


class CleaningService:

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """
        Pipeline làm sạch text:
          raw → unicode normalize → xóa rác → fix whitespace → sanitize
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("Text rỗng sau khi extract")

        text = raw_text

        # 1. Normalize Unicode — chuyển ký tự lạ về dạng chuẩn
        # NFC: ký tự tổ hợp (é = e + ́) → ký tự đơn (é)
        text = unicodedata.normalize("NFC", text)

        # 2. Xóa các dòng chỉ có số trang (vd: "- 2 -", "Page 2 of 5")
        text = re.sub(r"^\s*[-–]\s*\d+\s*[-–]\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[Pp]age\s+\d+\s+of\s+\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

        # 3. Xóa URL (không hữu ích cho embedding CV)
        # Giữ lại domain nếu muốn — hiện tại xóa hết
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)

        # 4. Chuẩn hóa dấu gạch ngang — nhiều loại dash → chuẩn -
        text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2015", "-")

        # 5. Chuẩn hóa dấu nháy
        text = text.replace("\u2018", "'").replace("\u2019", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')

        # 6. Xóa ký tự bullet đặc biệt → giữ text, bỏ bullet
        text = re.sub(r"[•·▪▸►◦‣⁃]", "-", text)

        # 7. Fix khoảng trắng thừa
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 8. Sanitize (prompt injection + giới hạn độ dài)
        text = TextSanitizerService.sanitize(text)

        text = text.strip()

        logger.info(
            "Text cleaning done",
            extra={
                "event": "cleaning_done",
                "original_chars": len(raw_text),
                "cleaned_chars": len(text),
            },
        )

        return text