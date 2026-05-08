import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class ChunkingService:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "]
        )

    def chunk_document(self, text: str, user_id: str = "", cv_id: str = ""):
        extra: dict[str, object] = {
            "event": "chunking_started",
            "status": "processing"
        }
        if user_id:
            extra["user_id"] = user_id
        if cv_id:
            extra["cv_id"] = cv_id
        
        text_length = len(text)
        logger.info(f"Starting document chunking (text length: {text_length})", extra=extra)
        
        try:
            chunks = self.splitter.split_text(text)
            
            extra["event"] = "chunking_completed"
            extra["status"] = "success"
            extra["chunks_count"] = len(chunks)
            logger.info(f"Document chunked into {len(chunks)} chunks", extra=extra)
            
            return chunks
        except Exception as e:
            extra["event"] = "chunking_failed"
            extra["status"] = "error"
            extra["error"] = str(e)
            logger.error(f"Document chunking failed: {str(e)}", extra=extra, exc_info=True)
            raise