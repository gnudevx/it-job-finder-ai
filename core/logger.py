import logging
import json
from datetime import datetime, timezone

TRACKED_FIELDS = (
    "event",
    "user_id",
    "cv_id",
    "file",
    "status",
    "duration_ms",
    "chunks_count",
    "task_id",
    "retry_count",
    "error",
    "size_bytes",
)

class JSONFormatter(logging.Formatter):
    """
    Thay vì: print("embedding failed")
    Output:  {"time":"...","level":"ERROR","event":"embedding_failed","user_id":"123"}
    Dễ search, filter, debug trên LangFuse / log aggregator.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Nếu logger.error("...", extra={"event": "...", "user_id": "..."})
        for key in TRACKED_FIELDS:
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, ensure_ascii=False)


def setup_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)

    # Bớt noise từ thư viện bên ngoài
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)