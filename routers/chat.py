"""
routers/chat.py — Chat endpoint kết nối RAG + Gemini.

POST /api/chat/         → gửi tin nhắn, nhận reply
GET  /api/chat/history/{session_id}  → xem lịch sử
DELETE /api/chat/history/{session_id} → xóa để bắt đầu lại
GET  /api/chat/tokens   → xem quota token hôm nay
"""

from fastapi import APIRouter, Depends, HTTPException
import logging

from core.dependencies import get_current_user, CurrentUser
from models.schemas import ChatRequest, ChatResponse
from services.CV.rag.rag_service import (
    build_rag_messages,
    save_message,
    get_history,
    clear_history,
    get_active_cv_id,
)
from services.AI.llm_service import chat_completion, get_token_usage

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """
    Luồng:
    1. Tìm cv_id mới nhất của user (nếu client không gửi lên)
    2. RAG: embed câu hỏi → tìm chunks CV liên quan → lấy history
    3. Gọi Gemini với context đầy đủ
    4. Lưu cả 2 tin nhắn (user + assistant) vào MongoDB
    5. Trả về reply + token info
    """
    logger.info("Chat request", extra={
        "event": "chat_request",
        "user_id": user.user_id,
        "mode": body.mode,
        "session_id": body.session_id,
    })

    # 1. Xác định cv_id — dùng CV mới nhất nếu client không gửi
    cv_id = body.cv_id or get_active_cv_id(user.user_id)

    if not cv_id and body.mode in ("cv_advisor", "mock_interview"):
        # Vẫn cho chat nhưng không có CV context
        logger.info("No CV found for user", extra={"user_id": user.user_id})

    # 2. Build messages với RAG context + history
    try:
        messages, cv_context  = build_rag_messages(
            user_message=body.message,
            session_id=body.session_id,
            user_id=user.user_id,
            cv_id=cv_id,
        )
    except Exception as e:
        logger.exception("RAG failed", extra={"user_id": user.user_id})
        raise HTTPException(status_code=500, detail="Lỗi khi lấy context CV")

    # 3. Gọi Gemini
    try:
        result = chat_completion(
            messages=messages,
            user_id=user.user_id,
            mode=body.mode,
            job_position=body.job_position,
            cv_context=cv_context,
        )
    except Exception as e:
        logger.exception("LLM failed", extra={"user_id": user.user_id})
        raise HTTPException(status_code=502, detail="Lỗi khi gọi AI model")

    # 4. Lưu lịch sử vào MongoDB
    save_message(body.session_id, user.user_id, "user", body.message)
    save_message(body.session_id, user.user_id, "assistant", result["reply"])

    # 5. Trả về
    return ChatResponse(
        reply=result["reply"],
        session_id=body.session_id,
        tokens_used=result["tokens_used"],
        tokens_remaining=result["tokens_remaining"],
        warning=result.get("warning"),
    )


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Lấy lịch sử chat của 1 session."""
    messages = get_history(session_id, limit=50)
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


@router.delete("/history/{session_id}")
async def delete_history(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Xóa lịch sử để bắt đầu phiên mới."""
    clear_history(session_id)
    return {"message": "Đã xóa lịch sử", "session_id": session_id}


@router.get("/tokens")
async def token_usage(
    user: CurrentUser = Depends(get_current_user),
):
    """Xem quota token hôm nay của user."""
    usage = get_token_usage(user.user_id)
    return usage