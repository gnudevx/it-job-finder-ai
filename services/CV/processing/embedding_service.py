"""
EmbeddingService — chuyển text chunks thành vector embeddings.

Model: all-MiniLM-L6-v2
  - Nhỏ (80MB), chạy CPU được, miễn phí
  - Output: 384 dimensions
  - Max input: 256 tokens (~1000 ký tự)
  - Đủ tốt cho CV similarity search

Singleton pattern: load model 1 lần khi khởi động,
tái sử dụng cho tất cả requests — tránh load lại mỗi lần (chậm ~2s).
"""

import logging
import time
from typing import Optional
from services.CV.processing.chunking_service import TextChunk

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
_model_instance = None   # Singleton


def _get_model():
    """Lazy load model — chỉ load lần đầu tiên gọi."""
    global _model_instance
    if _model_instance is None:
        from sentence_transformers import SentenceTransformer
        logger.info(
            "Loading embedding model",
            extra={"event": "model_loading", "model": MODEL_NAME},
        )
        start = time.time()
        _model_instance = SentenceTransformer(MODEL_NAME)
        elapsed = round(time.time() - start, 2)
        logger.info(
            "Embedding model loaded",
            extra={"event": "model_loaded", "model": MODEL_NAME, "duration_ms": elapsed * 1000},
        )
    return _model_instance


class EmbeddingService:

    def embed_chunks(self, chunks: list[TextChunk]) -> list[list[float]]:
        """
        Embed danh sách TextChunk → danh sách vector float.

        Args:
            chunks: list TextChunk từ ChunkingService

        Returns:
            list[list[float]] — mỗi vector có 384 dimensions
            Thứ tự tương ứng 1-1 với chunks đầu vào
        """
        if not chunks:
            raise ValueError("Danh sách chunks rỗng")

        texts = [chunk.text for chunk in chunks]

        logger.info(
            "Starting embedding",
            extra={
                "event": "embedding_start",
                "chunk_count": len(texts),
            },
        )

        start = time.time()
        model = _get_model()

        # batch_size=32: xử lý 32 chunks cùng lúc
        # show_progress_bar=False: không spam stdout trong production
        # normalize_embeddings=True: chuẩn hóa về unit vector
        #   → cosine similarity = dot product (nhanh hơn khi search)
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        elapsed = round(time.time() - start, 2)

        logger.info(
            "Embedding done",
            extra={
                "event": "embedding_done",
                "chunk_count": len(chunks),
                "vector_dims": embeddings.shape[1] if len(embeddings) > 0 else 0,
                "duration_ms": elapsed * 1000,
            },
        )

        # Trả về list[list[float]] — ChromaDB cần format này
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """
        Embed 1 câu query (dùng khi RAG search).
        Tách riêng để dễ cache sau này nếu cần.
        """
        if not query or not query.strip():
            raise ValueError("Query rỗng")

        model = _get_model()
        vector = model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()