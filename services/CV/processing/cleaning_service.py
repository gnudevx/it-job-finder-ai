import logging
from validation.text_sanitizer_service import TextSanitizerService
from validation.prompt_injection_service import PromptInjectionService

logger = logging.getLogger(__name__)


class CleaningService:
    @staticmethod
    def clean_text(text: str, user_id: str = "", cv_id: str = "") -> str:
        extra = {
            "event": "cleaning_started",
            "status": "processing"
        }
        if user_id:
            extra["user_id"] = user_id
        if cv_id:
            extra["cv_id"] = cv_id
        
        original_length = len(text)
        logger.info(f"Cleaning text (original length: {original_length})", extra=extra)
        
        try:
            text = TextSanitizerService.sanitize(text)
            logger.debug(f"Text sanitized", extra=extra)
            
            text = PromptInjectionService.sanitize(text)
            logger.debug(f"Prompt injection check completed", extra=extra)
            
            final_length = len(text)
            extra["event"] = "cleaning_completed"
            extra["status"] = "success"
            logger.info(f"Text cleaning completed (final length: {final_length})", extra=extra)
            
            return text
        except Exception as e:
            extra["event"] = "cleaning_failed"
            extra["status"] = "error"
            extra["error"] = str(e)
            logger.error(f"Text cleaning failed: {str(e)}", extra=extra, exc_info=True)
            raise