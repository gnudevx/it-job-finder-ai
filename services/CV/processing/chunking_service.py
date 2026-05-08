"""
ChunkingService — chia CV text thành chunks nhỏ để embed.

Tại sao cần overlap?
  Nếu không overlap, câu "Tôi có 3 năm kinh nghiệm React"
  nằm ở ranh giới 2 chunk sẽ bị cắt đôi → mất ý nghĩa khi retrieve.
  Overlap 50 token = 2 chunk kề nhau chia sẻ 50 token cuối/đầu.

Tại sao chia theo token thay vì ký tự?
  Embedding model (all-MiniLM-L6-v2) có giới hạn 256 token/chunk.
  1 token ≈ 4 ký tự tiếng Anh, ≈ 2-3 ký tự tiếng Việt.
  Chia theo ký tự có thể tạo chunk vượt giới hạn token → bị truncate.
"""

import logging
import re
from dataclasses import dataclass
from core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Một chunk text với metadata đi kèm."""
    text: str
    chunk_index: int    # thứ tự trong document
    char_start: int     # vị trí bắt đầu trong text gốc
    char_end: int       # vị trí kết thúc trong text gốc


class ChunkingService:
    """
    Chia text thành chunks có overlap.

    Chiến lược:
      1. Ưu tiên cắt tại ranh giới đoạn văn (\\n\\n)
      2. Nếu đoạn quá dài → cắt tại ranh giới câu (. ! ?)
      3. Nếu câu quá dài → cắt theo số ký tự (fallback)
    """

    def __init__(
        self,
        chunk_size: int = -1,
        chunk_overlap: int = -1,
    ):
        # Đọc từ config (settings.CHUNK_SIZE = 500, CHUNK_OVERLAP = 50)
        # × 4 vì 1 token ≈ 4 ký tự
        self.chunk_size = (
            chunk_size if chunk_size is not None else settings.CHUNK_SIZE
        ) * 4
        self.chunk_overlap = (chunk_overlap or settings.CHUNK_OVERLAP) * 4

    def chunk_document(self, text: str) -> list[TextChunk]:
        """
        Chia toàn bộ text thành list TextChunk.

        Returns:
            List[TextChunk] — ít nhất 1 chunk
        """
        if not text or not text.strip():
            raise ValueError("Không thể chunk text rỗng")

        # Tách thành các đoạn văn trước
        paragraphs = self._split_paragraphs(text)

        # Gom các đoạn vào chunks có kích thước phù hợp
        chunks = self._build_chunks(paragraphs, text)

        logger.info(
            "Chunking done",
            extra={
                "event": "chunking_done",
                "total_chars": len(text),
                "chunk_count": len(chunks),
                "chunk_size_chars": self.chunk_size,
                "overlap_chars": self.chunk_overlap,
            },
        )

        return chunks

    # ── Private helpers ───────────────────────────────────────────────────────

    def _split_paragraphs(self, text: str) -> list[str]:
        """Tách text thành đoạn tại \\n\\n, loại đoạn rỗng."""
        paragraphs = re.split(r"\n{2,}", text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _build_chunks(self, paragraphs: list[str], original_text: str) -> list[TextChunk]:
        """
        Gom paragraphs thành chunks.
        Nếu paragraph đơn lẻ vượt chunk_size → tách theo câu.
        """
        chunks: list[TextChunk] = []
        current_parts: list[str] = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)

            # Paragraph quá dài → tách nhỏ trước
            if para_len > self.chunk_size:
                # Flush current buffer trước
                if current_parts:
                    self._flush_chunk(current_parts, chunks, original_text)
                    # Giữ lại overlap từ chunk trước
                    overlap_text = " ".join(current_parts)[-self.chunk_overlap:]
                    current_parts = [overlap_text] if overlap_text.strip() else []
                    current_len = len(overlap_text)

                # Tách paragraph dài thành sub-chunks
                sub_chunks = self._split_long_paragraph(para)
                for sub in sub_chunks:
                    idx = len(chunks)
                    start = original_text.find(sub[:50]) if sub[:50] in original_text else 0
                    chunks.append(TextChunk(
                        text=sub,
                        chunk_index=idx,
                        char_start=start,
                        char_end=start + len(sub),
                    ))
                continue

            # Chunk hiện tại sẽ vượt kích thước → flush
            if current_len + para_len > self.chunk_size and current_parts:
                self._flush_chunk(current_parts, chunks, original_text)

                # Overlap: giữ lại phần cuối của chunk vừa tạo
                combined = " ".join(current_parts)
                overlap_text = combined[-self.chunk_overlap:]
                current_parts = [overlap_text] if overlap_text.strip() else []
                current_len = len(overlap_text)

            current_parts.append(para)
            current_len += para_len + 1  # +1 cho space

        # Flush phần còn lại
        if current_parts:
            self._flush_chunk(current_parts, chunks, original_text)

        return chunks if chunks else [
            TextChunk(text=original_text[:self.chunk_size], chunk_index=0,
                      char_start=0, char_end=min(self.chunk_size, len(original_text)))
        ]

    def _flush_chunk(
        self,
        parts: list[str],
        chunks: list[TextChunk],
        original_text: str,
    ) -> None:
        chunk_text = " ".join(parts).strip()
        if not chunk_text:
            return
        idx = len(chunks)
        # Tìm vị trí trong text gốc
        start = original_text.find(chunk_text[:60]) if len(chunk_text) >= 60 else 0
        chunks.append(TextChunk(
            text=chunk_text,
            chunk_index=idx,
            char_start=max(0, start),
            char_end=max(0, start) + len(chunk_text),
        ))

    def _split_long_paragraph(self, text: str) -> list[str]:
        """Tách đoạn quá dài theo câu."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sub_chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) > self.chunk_size and current:
                sub_chunks.append(current.strip())
                # Overlap: giữ câu cuối
                current = current[-self.chunk_overlap:] + " " + sentence
            else:
                current = current + " " + sentence if current else sentence

        if current.strip():
            sub_chunks.append(current.strip())

        return sub_chunks if sub_chunks else [text[:self.chunk_size]]