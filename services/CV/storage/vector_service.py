"""
VectorService — lưu và query CV embeddings trong MongoDB Atlas Vector Search.

Thay thế ChromaDB bằng MongoDB Atlas Vector Search (free M0 tier).

Collection: cv_vectors (trong DB cv_chatbot)
Schema mỗi document:
  {
    _id: ObjectId,
    chunk_id: str,          # "{cv_id}_{chunk_index}" — unique
    cv_id: str,
    user_id: str,
    filename: str,
    chunk_index: int,
    char_start: int,
    char_end: int,
    text: str,              # text của chunk
    embedding: list[float], # 384-dim vector (all-MiniLM-L6-v2)
  }

Atlas Vector Search Index (tạo thủ công 1 lần trên Atlas UI):
  Index name: cv_embedding_index
  Collection: cv_chatbot.cv_vectors
  Definition:
    {
      "fields": [
        {
          "type": "vector",
          "path": "embedding",
          "numDimensions": 384,
          "similarity": "cosine"
        },
        {
          "type": "filter",
          "path": "cv_id"
        },
        {
          "type": "filter",
          "path": "user_id"
        }
      ]
    }
"""

import logging
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from services.CV.processing.chunking_service import TextChunk
from core.config import settings
from typing import Any

logger = logging.getLogger(__name__)

COLLECTION_NAME = "cv_vectors"
VECTOR_INDEX_NAME = "cv_embedding_index"

_mongo_client = None


def _get_collection() -> Collection:
    """Singleton MongoDB client → collection cv_vectors."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
        logger.info(
            "MongoDB Vector client initialized",
            extra={"event": "mongo_vector_connected"},
        )
    db = _mongo_client[settings.MONGODB_DB]
    col = db[COLLECTION_NAME]

    # Tạo index thường (unique) cho chunk_id — để upsert nhanh
    try:
        col.create_index("chunk_id", unique=True, background=True)
        col.create_index("cv_id", background=True)
        col.create_index("user_id", background=True)
    except Exception:
        pass  # Index đã tồn tại — bỏ qua

    return col

class VectorService:
    def store_embeddings(
        self,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
        user_id: str,
        cv_id: str,
        filename: str,
    ) -> None:
        """
        Lưu chunks + embeddings vào MongoDB Atlas.

        Args:
            chunks: list TextChunk (text + metadata)
            embeddings: list vector float tương ứng 1-1 với chunks
            user_id, cv_id, filename: metadata để filter khi retrieve
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Số chunk ({len(chunks)}) không khớp với số embedding ({len(embeddings)})"
            )

        col = _get_collection()

        # Chuẩn bị batch upsert — dùng bulk_write để tối ưu
        operations = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = f"{cv_id}_{chunk.chunk_index}"
            doc = {
                "chunk_id": chunk_id,
                "cv_id": cv_id,
                "user_id": user_id,
                "filename": filename,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "text": chunk.text,
                "embedding": embedding,  # list[float] 384 dims
            }
            operations.append(
                UpdateOne(
                    {"chunk_id": chunk_id},
                    {"$set": doc},
                    upsert=True,
                )
            )

        if operations:
            result = col.bulk_write(operations, ordered=False)
            logger.info(
                "Vectors stored",
                extra={
                    "event": "vectors_stored",
                    "cv_id": cv_id,
                    "chunk_count": len(chunks),
                    "upserted": result.upserted_count,
                    "modified": result.modified_count,
                },
            )

    def delete_existing_cv(self, cv_id: str) -> None:
        """
        Xóa tất cả chunks của cv_id trước khi lưu lại.
        Tránh duplicate khi user upload lại CV cùng tên.
        """
        col = _get_collection()
        result = col.delete_many({"cv_id": cv_id})
        if result.deleted_count > 0:
            logger.info(
                "Existing vectors deleted",
                extra={
                    "event": "vectors_deleted",
                    "cv_id": cv_id,
                    "count": result.deleted_count,
                },
            )

    def get_first_chunks(self, cv_id: str, limit: int = 8) -> list[dict]:
        """
        Lấy N chunks đầu tiên của CV theo chunk_index (không cần embedding).
        Dùng khi cần đọc nội dung CV để generate intro message hoặc summary.

        Args:
            cv_id: ID của CV
            limit: số lượng chunks cần lấy (mặc định 8 để có đủ nội dung)

        Returns:
            list[dict] với keys: text, chunk_index
        """
        col = _get_collection()
        docs = list(
            col.find(
                {"cv_id": cv_id},
                {"_id": 0, "text": 1, "chunk_index": 1},
            )
            .sort("chunk_index", 1)
            .limit(limit)
        )
        return [{"text": d["text"], "chunk_index": d.get("chunk_index", 0)} for d in docs]

    def query_similar_chunks(
        self,
        query_embedding: list[float],
        user_id: str,
        cv_id: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Tìm top-k chunks gần nhất với query embedding dùng Atlas Vector Search.
        Filter theo cv_id để chỉ tìm trong CV của user đó.

        Returns:
            list[dict] với keys: text, score, chunk_index, cv_id
        """
        col = _get_collection()

        # Atlas Vector Search $vectorSearch aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": top_k * 10,  # candidates = numResults * 10 (best practice)
                    "limit": top_k,
                    "filter": {"cv_id": {"$eq": cv_id}},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "text": 1,
                    "chunk_index": 1,
                    "cv_id": 1,
                    # vectorSearchScore: điểm similarity từ Atlas
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            results = list(col.aggregate(pipeline))
        except Exception as e:
            logger.error(
                "Atlas Vector Search failed",
                extra={"event": "vector_search_failed", "error": str(e), "cv_id": cv_id},
            )
            # Fallback: text search nếu Vector Search index chưa tạo
            return self._fallback_text_search(col, cv_id, top_k)

        logger.info(
            "Vector search done",
            extra={
                "event": "vector_search_done",
                "cv_id": cv_id,
                "results": len(results),
            },
        )

        return [
            {
                "text": r["text"],
                "score": round(r.get("score", 0), 4),
                "chunk_index": r.get("chunk_index"),
                "cv_id": r.get("cv_id"),
            }
            for r in results
        ]

    def _fallback_text_search(self, col: Collection, cv_id: str, top_k: int) -> list[dict]:
        """
        Fallback khi Atlas Vector Search index chưa tạo:
        lấy TẤT CẢ chunks của CV, sort theo chunk_index.
        
        Lý do không giới hạn limit=top_k: nếu chỉ lấy N chunk đầu tiên
        (chunk_index 0,1,2...) thì thường là thông tin cá nhân/mục tiêu chung chung,
        không có nội dung kỹ năng/kinh nghiệm → LLM hỏi câu chung chung.
        Lấy toàn bộ CV để LLM đủ context hỏi câu chuyên sâu.
        """
        logger.warning(
            "Using fallback text search — Atlas Vector Search index may not be ready. "
            "Create 'cv_embedding_index' on Atlas UI for semantic search.",
            extra={"event": "vector_search_fallback", "cv_id": cv_id},
        )
        # Lấy tất cả chunks, không giới hạn theo top_k
        docs = list(
            col.find({"cv_id": cv_id}, {"_id": 0, "text": 1, "chunk_index": 1, "cv_id": 1})
            .sort("chunk_index", 1)
        )
        # Trả về tối đa top_k chunks (nhưng đã có toàn bộ CV để chọn)
        return [
            {
                "text": d["text"],
                "score": 1.0,  # fallback không có score thật
                "chunk_index": d.get("chunk_index"),
                "cv_id": d.get("cv_id"),
            }
            for d in docs[:top_k]
        ]