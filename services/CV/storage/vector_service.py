"""
VectorService — lưu và query CV embeddings trong ChromaDB.

Collection name: "cv_chunks"
Mỗi document trong ChromaDB gồm:
  - id: "{cv_id}_{chunk_index}" (unique)
  - embedding: list[float] 384 dims
  - document: text của chunk
  - metadata: {user_id, cv_id, filename, chunk_index, char_start, char_end}
"""

import logging
import chromadb
from chromadb.config import Settings as ChromaSettings
from services.CV.processing.chunking_service import TextChunk
from core.config import settings
from typing import Any

logger = logging.getLogger(__name__)

COLLECTION_NAME = "cv_chunks"
_chroma_client = None


def _get_client() -> Any:
    """Singleton ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(
            "ChromaDB client initialized",
            extra={"event": "chroma_connected", "host": settings.CHROMA_HOST},
        )
    return _chroma_client


def _get_collection():
    """Lấy hoặc tạo collection cv_chunks."""
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # cosine: phù hợp với normalized embeddings từ EmbeddingService
        metadata={"hnsw:space": "cosine"},
    )


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
        Lưu chunks + embeddings vào ChromaDB.

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

        # ChromaDB nhận list — chuẩn bị batch
        ids = [f"{cv_id}_{chunk.chunk_index}" for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "user_id": user_id,
                "cv_id": cv_id,
                "filename": filename,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }
            for chunk in chunks
        ]

        # upsert: nếu id đã tồn tại → overwrite (an toàn khi re-upload)
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(
            "Vectors stored",
            extra={
                "event": "vectors_stored",
                "cv_id": cv_id,
                "chunk_count": len(chunks),
            },
        )

    def delete_existing_cv(self, cv_id: str) -> None:
        """
        Xóa tất cả chunks của cv_id trước khi lưu lại.
        Tránh duplicate khi user upload lại CV cùng tên.
        """
        col = _get_collection()
        # ChromaDB filter theo metadata
        existing = col.get(where={"cv_id": cv_id})
        if existing and existing["ids"]:
            col.delete(ids=existing["ids"])
            logger.info(
                "Existing vectors deleted",
                extra={"event": "vectors_deleted", "cv_id": cv_id, "count": len(existing["ids"])},
            )

    def query_similar_chunks(
        self,
        query_embedding: list[float],
        user_id: str,
        cv_id: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Tìm top-k chunks gần nhất với query embedding.
        Filter theo user_id + cv_id để chỉ tìm trong CV của user đó.

        Returns:
            list[dict] với keys: text, score, chunk_index, cv_id
        """
        col = _get_collection()

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"cv_id": {"$eq": cv_id}},
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"] or not results["documents"][0]:
            return []

        # ChromaDB trả về nested list [[...]] — lấy phần tử đầu
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {
                "text": docs[i],
                "score": round(1 - distances[i], 4),  # cosine distance → similarity
                "chunk_index": metas[i].get("chunk_index"),
                "cv_id": metas[i].get("cv_id"),
            }
            for i in range(len(docs))
        ]