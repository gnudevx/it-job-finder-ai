"""
EmbeddingService — chuyển text chunks thành vector embeddings.

Model: gemini-embedding-001 (Google GenAI API)
  - Output: 384 dimensions (configured via output_dimensionality)
  - Chạy hoàn toàn qua Cloud API, không tốn RAM chạy local (tránh OOM 512MB Render)
  - Tốc độ xử lý nhanh, ổn định

Singleton pattern: khởi tạo GenAI client 1 lần.
"""

import logging
import time
from typing import Optional
from services.CV.processing.chunking_service import TextChunk
from google import genai
from google.genai import types
from core.config import settings

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-embedding-001"
_gemini_client = None   # Singleton client


def _get_client():
    """Lazy load client — chỉ khởi tạo khi cần."""
    global _gemini_client
    if _gemini_client is None:
        logger.info(
            "Initializing Gemini client for embeddings",
            extra={"event": "gemini_embeddings_client_loading"},
        )
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
        logger.info(
            "Gemini client for embeddings initialized",
            extra={"event": "gemini_embeddings_client_loaded"},
        )
    return _gemini_client


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
            "Starting embedding via Gemini API",
            extra={
                "event": "embedding_start",
                "chunk_count": len(texts),
            },
        )

        start = time.time()
        client = _get_client()

        # Gọi Gemini embedding API cho list text
        response = client.models.embed_content(
            model=MODEL_NAME,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=384)
        )

        # Trích xuất values từ response
        embeddings = [e.values for e in response.embeddings]

        elapsed = round(time.time() - start, 2)

        logger.info(
            "Embedding done via Gemini API",
            extra={
                "event": "embedding_done",
                "chunk_count": len(chunks),
                "vector_dims": len(embeddings[0]) if len(embeddings) > 0 else 0,
                "duration_ms": elapsed * 1000,
            },
        )

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """
        Embed 1 câu query (dùng khi RAG search).
        """
        if not query or not query.strip():
            raise ValueError("Query rỗng")

        client = _get_client()
        response = client.models.embed_content(
            model=MODEL_NAME,
            contents=query,
            config=types.EmbedContentConfig(output_dimensionality=384)
        )
        return response.embeddings[0].values