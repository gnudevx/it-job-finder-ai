"""
RAGService — lấy context CV từ ChromaDB + lịch sử chat từ MongoDB,
ghép thành messages để gửi cho LLM.

Luồng:
  1. Embed câu hỏi của user
  2. Tìm top-3 chunks CV liên quan trong ChromaDB
  3. Lấy 10 tin nhắn gần nhất từ MongoDB
  4. Ghép: [CV context + history + câu hỏi mới]
  5. Lưu tin nhắn mới vào MongoDB
"""

import logging
from datetime import datetime, timezone
from pymongo import MongoClient, DESCENDING
from services.CV.processing.embedding_service import EmbeddingService
from services.CV.storage.vector_service import VectorService
from core.config import settings

logger = logging.getLogger(__name__)

_mongo_client: MongoClient | None = None

GENERAL_CV_QUESTIONS = [
    "điểm yếu", "điểm mạnh", "nhận xét", "đánh giá", 
    "phân tích", "cải thiện", "review", "cv của tôi"
]

def _get_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
    return _mongo_client[settings.MONGODB_DB]


def _get_session_collection():
    db = _get_db()
    col = db["chat_sessions"]
    try:
        col.create_index([("session_id", 1), ("user_id", 1)], unique=True, background=True)
    except Exception:
        pass
    return col


# ── Conversation history ──────────────────────────────────────────────────────

def resolve_session_id(session_id: str | None, user_id: str) -> str:
    """Resolve một session ID ổn định cho user, tránh mất history khi client không gửi session."""
    raw = (session_id or "").strip()
    if raw:
        return raw
    return f"user:{user_id}:default"


def save_message(session_id: str | None, user_id: str, role: str, content: str) -> None:
    """Lưu 1 tin nhắn vào MongoDB."""
    db = _get_db()
    db["chat_history"].insert_one({
        "session_id": session_id,
        "user_id": user_id,
        "role": role,          # "user" | "assistant"
        "content": content,
        "created_at": datetime.now(timezone.utc),
    })


def get_session_record(session_id: str, user_id: str) -> dict | None:
    col = _get_session_collection()
    return col.find_one({"session_id": session_id, "user_id": user_id}, {"_id": 0})


def set_session_cv_id(session_id: str, user_id: str, cv_id: str) -> None:
    col = _get_session_collection()
    now = datetime.now(timezone.utc)
    col.update_one(
        {"session_id": session_id, "user_id": user_id},
        {
            "$set": {
                "cv_id": cv_id,
                "cleared": False,
                "updated_at": now,
            },
            "$setOnInsert": {"session_id": session_id, "user_id": user_id, "created_at": now},
        },
        upsert=True,
    )


def get_session_cv_id(session_id: str, user_id: str) -> str | None:
    record = get_session_record(session_id, user_id)
    if record and not record.get("cleared", False):
        return record.get("cv_id")
    return None


def mark_session_cleared(session_id: str, user_id: str) -> None:
    col = _get_session_collection()
    now = datetime.now(timezone.utc)
    col.update_one(
        {"session_id": session_id, "user_id": user_id},
        {
            "$set": {
                "cv_id": None,
                "cleared": True,
                "updated_at": now,
            },
            "$setOnInsert": {"session_id": session_id, "user_id": user_id, "created_at": now},
        },
        upsert=True,
    )


def should_use_active_cv(session_id: str, user_id: str) -> bool:
    record = get_session_record(session_id, user_id)
    if record is None:
        return True
    return not record.get("cleared", False)


def get_history(session_id: str | None, limit: int = 10) -> list[dict]:
    """
    Lấy `limit` tin nhắn gần nhất của session.
    Trả về theo thứ tự cũ → mới (đúng format cho LLM).
    """
    db = _get_db()
    docs = list(
        db["chat_history"]
        .find({"session_id": session_id}, {"_id": 0, "role": 1, "content": 1})
        .sort("created_at", DESCENDING)
        .limit(limit)
    )
    # Reverse để thứ tự cũ → mới
    return [{"role": d["role"], "content": d["content"]} for d in reversed(docs)]


def clear_history(session_id: str | None) -> None:
    db = _get_db()
    db["chat_history"].delete_many({"session_id": session_id})


# ── RAG context builder ───────────────────────────────────────────────────────

def _is_general_cv_question(message: str) -> bool:
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in GENERAL_CV_QUESTIONS)
def retrieve_cv_context(user_message: str, user_id: str, cv_id: str | None = None) -> str:
    """Lấy CV context tương tự từ ChromaDB dựa trên câu hỏi của user."""
    if not cv_id:
        return ""
    try:
        embedding_service = EmbeddingService()
        query_vector = embedding_service.embed_query(user_message)

        vector_service = VectorService()
        if _is_general_cv_question(user_message):
            top_k = 8   # lấy gần hết CV
        else:
            top_k = 2 
        chunks = vector_service.query_similar_chunks(
            query_embedding=query_vector,
            user_id=user_id,
            cv_id=cv_id,
            top_k=top_k,
        )
        if chunks:
            context_parts = [f"[Đoạn {i+1}]: {c['text']}" for i, c in enumerate(chunks)]
            cv_context = "\n\n".join(context_parts)
            logger.info(
                "RAG chunks retrieved",
                extra={
                    "event": "rag_retrieved",
                    "cv_id": cv_id,
                    "chunks": len(chunks),
                    "top_score": chunks[0]["score"] if chunks else 0,
                },
            )
            return cv_context
    except Exception as e:
        import traceback
        logger.error(
            "RAG retrieval failed FULL ERROR: " + traceback.format_exc(),
            extra={"event": "rag_failed", "error": str(e)},
        )
    return ""


def build_rag_messages(
    user_message: str,
    session_id: str,
    user_id: str,
    cv_id: str | None = None,
) -> tuple[list[dict], str]:
    """
    Tạo list messages đầy đủ để gửi LLM:
      [history] + [user message hiện tại]
      CV context được trả về riêng để inject vào System Prompt.
    """
    cv_context = retrieve_cv_context(user_message, user_id, cv_id)

    # Lấy lịch sử chat
    history = get_history(session_id, limit=6)
    messages = history + [{"role": "user", "content": user_message}]

    if not cv_context:
        # Nếu không lấy được nội dung CV, thêm prompt rõ ràng để LLM không tái sử dụng CV cũ
        messages.insert(0, {
            "role": "system",
            "content": "Không có nội dung CV cụ thể để tham khảo. Nếu CV mới chưa sẵn sàng, không được sử dụng hoặc suy đoán dựa trên CV cũ."
        })

    return messages, cv_context



# ── Active CV lookup ──────────────────────────────────────────────────────────

def get_active_cv_id(user_id: str) -> str | None:
    """
    Lấy cv_id đang được đánh dấu active cho user.
    Nếu chưa có active flag thì fallback về CV mới nhất theo created_at.
    """
    db = _get_db()
    doc = db["cv_metadata"].find_one(
        {"user_id": user_id, "is_active": True},
        {"cv_id": 1},
        sort=[("updated_at", DESCENDING)],
    )

    if doc:
        return doc["cv_id"]

    doc = db["cv_metadata"].find_one(
        {"user_id": user_id},
        {"cv_id": 1},
        sort=[("created_at", DESCENDING)],
    )
    return doc["cv_id"] if doc else None


def resolve_cv_id(requested_cv_id: str | None, user_id: str) -> str | None:
    """Chọn CV phù hợp cho user, ưu tiên CV active mới nhất."""
    active_cv_id = get_active_cv_id(user_id)
    if not active_cv_id:
        return requested_cv_id

    if requested_cv_id and requested_cv_id != active_cv_id:
        logger.info(
            "Requested CV id does not match active CV; falling back to active CV",
            extra={
                "event": "cv_id_fallback",
                "requested_cv_id": requested_cv_id,
                "active_cv_id": active_cv_id,
                "user_id": user_id,
            },
        )
        return active_cv_id

    return requested_cv_id or active_cv_id