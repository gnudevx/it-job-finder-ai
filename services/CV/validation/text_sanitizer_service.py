import logging
import re

logger = logging.getLogger(__name__)


class TextSanitizerService:
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        logger.debug(f"Whitespace normalized: {len(text)} -> {len(normalized)} chars")
        return normalized

    @staticmethod
    def remove_weird_unicode(text: str) -> str:
        cleaned = text.encode("utf-8", errors="ignore").decode("utf-8")
        if len(cleaned) != len(text):
            logger.debug(f"Removed non-UTF8 characters: {len(text) - len(cleaned)} chars removed")
        return cleaned

    @classmethod
    def sanitize(cls, text: str) -> str:
        original_length = len(text)
        logger.debug(f"Text sanitization starting (length: {original_length})", extra={"event": "sanitization_started"})
        
        try:
            text = cls.normalize_whitespace(text)
            text = cls.remove_weird_unicode(text)
            
            logger.debug(f"Text sanitization completed (final length: {len(text)})", extra={"event": "sanitization_completed"})
            return text
        except Exception as e:
            logger.error(f"Text sanitization failed: {str(e)}", extra={"event": "sanitization_failed", "error": str(e)}, exc_info=True)
            raise