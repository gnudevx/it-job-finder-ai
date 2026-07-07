"""
MetadataService — lưu metadata và trạng thái xử lý CV vào MongoDB.

Collection: cv_metadata
Schema:
  {
    cv_id: str (unique),
    user_id: str,
    filename: str,
    status: "uploaded" | "processing" | "done" | "failed",
    chunks_count: int | null,
    error_message: str | null,
    created_at: datetime,
    updated_at: datetime,
  }

Tại sao dùng pymongo sync thay vì motor async?
  MetadataService được gọi từ Celery worker (sync context).
  Motor cần event loop — dùng trong FastAPI async route thì đúng,
  nhưng trong Celery task sẽ phức tạp. Pymongo sync đơn giản hơn ở đây.
"""

import logging
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.collection import Collection
from core.config import settings
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

_client = None


def _get_collection() -> Collection:
    """Singleton MongoClient — tái sử dụng connection pool."""
    global _client
    if _client is None:
        try:
            _client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
            # ← Force connection test ngay lúc init
            _client.admin.command('ping')
            logger.info(f"✅ MongoDB connected: {settings.MONGODB_URI}")
        except Exception as e:
            logger.exception(f"❌ MongoDB connection failed: {e}")
            raise
    
    db = _client[settings.MONGODB_DB]
    
    # ← Create collection explicitly if not exists
    col = db["cv_metadata"]
    try:
        db.create_collection("cv_metadata")
        logger.info("✅ Collection 'cv_metadata' created")
    except Exception as e:
        logger.warning(f"Collection creation: {e}")
    
    col = db["cv_metadata"]
    
    # ← Ensure collection has index
    try:
        col.create_index("cv_id", unique=True)
    except Exception as idx_err:
        logger.warning(f"Index creation warning: {idx_err}")
    
    return col


class MetadataService:

    def update_status(
        self,
        cv_id: str,
        status: str,
        user_id: Optional[str] = None,
        filename: Optional[str] = None,
        chunks_count: Optional[int] = None,
        error_message: Optional[str] = None,
        intro_message: Optional[str] = None,
        detected_job_title: Optional[str] = None,
    ) -> None:
        """
        Upsert trạng thái CV.
        Nếu document chưa tồn tại → tạo mới.
        Nếu đã tồn tại → chỉ cập nhật các field được truyền vào.
        """
        col = _get_collection()
        now = datetime.now(timezone.utc)

        # Chỉ set các field có giá trị thật
        update_fields: dict = {
            "status": status,
            "updated_at": now,
        }
        if user_id is not None:
            update_fields["user_id"] = user_id
            col.update_many(
                {"user_id": user_id, "cv_id": {"$ne": cv_id}},
                {"$set": {"is_active": False}},
            )
            update_fields["is_active"] = True

        if filename is not None:
            update_fields["filename"] = filename

        if chunks_count is not None:
            update_fields["chunks_count"] = chunks_count

        if error_message is not None:
            update_fields["error_message"] = error_message

        if intro_message is not None:
            update_fields["intro_message"] = intro_message

        if detected_job_title is not None:
            update_fields["detected_job_title"] = detected_job_title

        try:
            result = col.update_one(
                {"cv_id": cv_id},
                {
                    "$set": update_fields,
                    "$setOnInsert": {"cv_id": cv_id, "created_at": now},
                },
                upsert=True,
            )
            
            logger.info(
                "CV metadata updated",
                extra={
                    "event": "metadata_updated",
                    "cv_id": cv_id,
                    "status": status,
                    "matched_count": result.matched_count,
                    "modified_count": result.modified_count,
                    "upserted_id": str(result.upserted_id) if result.upserted_id else None,
                },
            )
        except Exception as e:
            logger.exception(f"❌ Failed to update CV metadata: {e}")
            raise

    def get_status(self, cv_id: str) -> Optional[Dict]:
        """
        Lấy metadata của CV theo cv_id.
        Returns None nếu không tìm thấy.
        """
        col = _get_collection()
        doc = col.find_one({"cv_id": cv_id}, {"_id": 0})
        return doc

    def get_by_user(self, user_id: str) -> List[Dict]:
        """Lấy tất cả CV của 1 user, sắp xếp mới nhất trước."""
        col = _get_collection()
        return list(
            col.find(
                {"user_id": user_id},
                {"_id": 0},
            ).sort("created_at", -1).limit(20)
        )