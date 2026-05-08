import logging
import chromadb
from chromadb.api.types import Metadata

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self):
        logger.info("Initializing VectorService with ChromaDB", extra={"event": "vector_service_init"})
        self.client = chromadb.PersistentClient(path="./chroma_db")

        self.collection = self.client.get_or_create_collection(
            name="cv_embeddings"
        )
        logger.info("ChromaDB collection 'cv_embeddings' initialized", extra={"event": "vector_service_ready"})

    def delete_existing_cv(self, cv_id: str):
        extra = {
            "event": "cv_deletion_started",
            "cv_id": cv_id,
            "status": "processing"
        }
        logger.info(f"Deleting existing CV from vector store: {cv_id}", extra=extra)
        
        try:
            self.collection.delete(
                where={"cv_id": cv_id}
            )
            extra["event"] = "cv_deletion_completed"
            extra["status"] = "success"
            logger.info(f"Successfully deleted CV {cv_id} from vector store", extra=extra)
        except Exception as e:
            extra["event"] = "cv_deletion_failed"
            extra["status"] = "error"
            extra["error"] = str(e)
            logger.error(f"Failed to delete CV {cv_id}: {str(e)}", extra=extra, exc_info=True)
            raise

    def store_embeddings(
        self,
        chunks,
        embeddings,
        user_id,
        cv_id,
        filename,
    ):
        extra = {
            "event": "embedding_storage_started",
            "user_id": user_id,
            "cv_id": cv_id,
            "file": filename,
            "chunks_count": len(chunks),
            "status": "processing"
        }
        logger.info(f"Storing {len(chunks)} embeddings for CV {cv_id}", extra=extra)
        
        try:
            ids = [f"{cv_id}_{i}" for i in range(len(chunks))]

            metadatas: list[Metadata] = [
                {
                    "user_id": user_id,
                    "cv_id": cv_id,
                    "filename": filename,
                }
                for _ in chunks
            ]

            self.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            
            extra["event"] = "embedding_storage_completed"
            extra["status"] = "success"
            logger.info(f"Successfully stored {len(chunks)} embeddings in vector store", extra=extra)
        except Exception as e:
            extra["event"] = "embedding_storage_failed"
            extra["status"] = "error"
            extra["error"] = str(e)
            logger.error(f"Failed to store embeddings: {str(e)}", extra=extra, exc_info=True)
            raise