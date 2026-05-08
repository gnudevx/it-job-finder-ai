import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        logger.info("Initializing EmbeddingService with BAAI/bge-small-en-v1.5 model")
        self.model = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        )
        logger.info("Embedding model loaded successfully")

    def embed_chunks(self, chunks: list[str], user_id: str = "", cv_id: str = ""):
        extra = {
            "event": "embedding_started",
            "chunks_count": len(chunks),
            "status": "processing"
        }
        if user_id:
            extra["user_id"] = user_id
        if cv_id:
            extra["cv_id"] = cv_id
        
        logger.info(f"Embedding {len(chunks)} chunks", extra=extra)
        
        try:
            embeddings = self.model.encode(chunks).tolist()
            
            extra["event"] = "embedding_completed"
            extra["status"] = "success"
            logger.info(f"Successfully embedded {len(chunks)} chunks", extra=extra)
            
            return embeddings
        except Exception as e:
            extra["event"] = "embedding_failed"
            extra["status"] = "error"
            extra["error"] = str(e)
            logger.error(f"Embedding failed: {str(e)}", extra=extra, exc_info=True)
            raise