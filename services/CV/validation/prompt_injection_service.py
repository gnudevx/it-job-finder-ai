import logging
import re

logger = logging.getLogger(__name__)


class PromptInjectionService:
    SUSPICIOUS_PATTERNS = [
        r"ignore previous instructions",
        r"system prompt",
        r"you are now",
        r"developer mode",
    ]

    @classmethod
    def sanitize(cls, text: str) -> str:
        logger.debug(f"Prompt injection check starting", extra={"event": "injection_check_started"})
        
        cleaned = text
        patterns_found = []
        
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                patterns_found.append(pattern)
                logger.warning(f"Suspicious pattern detected: {pattern}", extra={"event": "suspicious_pattern_found", "pattern": pattern})
                cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        if patterns_found:
            logger.info(f"Prompt injection check: {len(patterns_found)} suspicious patterns removed", 
                       extra={"event": "injection_patterns_removed", "patterns_count": len(patterns_found)})
        else:
            logger.debug(f"Prompt injection check: no suspicious patterns found", extra={"event": "injection_check_clean"})
        
        return cleaned