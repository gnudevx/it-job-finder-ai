import logging
from pymongo import MongoClient
import os

logger = logging.getLogger(__name__)


class MetadataService:
    def __init__(self):
        logger.info("Initializing MetadataService with MongoDB", extra={"event": "metadata_service_init"})
        try:
            self.client = MongoClient(os.getenv("MONGO_URI"))
            self.db = self.client["ai_service"]
            self.collection = self.db["cv_metadata"]
            logger.info("MongoDB connection established", extra={"event": "mongodb_connected"})
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}", extra={"event": "mongodb_connection_failed", "error": str(e)}, exc_info=True)
            raise

    def update_status(
        self,
        cv_id: str,
        status: str,
        chunks_count: int = 0,
    ):
        extra = {
            "event": "metadata_update_started",
            "cv_id": cv_id,
            "status": status,
            "chunks_count": chunks_count
        }
        logger.info(f"Updating metadata for CV {cv_id} to status: {status}", extra=extra)
        
        try:
            self.collection.update_one(
                {"cv_id": cv_id},
                {
                    "$set": {
                        "status": status,
                        "chunks_count": chunks_count,
                    }
                },
                upsert=True,
            )
            extra["event"] = "metadata_update_completed"
            logger.info(f"Metadata updated for CV {cv_id}", extra=extra)
        except Exception as e:
            extra["event"] = "metadata_update_failed"
            extra["error"] = str(e)
            logger.error(f"Failed to update metadata for CV {cv_id}: {str(e)}", extra=extra, exc_info=True)
            raise