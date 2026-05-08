from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import logging

from core.dependencies import get_current_user, CurrentUser
from models.schemas import ChatRequest, ChatResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Đây là router chính cho các endpoint liên quan đến chat. 
# Luồng sẽ là: verify auth → check token quota → RAG → build prompt → LLM → lưu history
# Hiện tại mới là placeholder, sẽ implement dần từng bước ở services/ để giữ router gọn nhẹ, chỉ tập trung vào request-response handling.
# AI brain
@router.post("/", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """
    Endpoint chat chính.
    Luồng: verify auth → check token quota → RAG → build prompt → LLM → lưu history
    TODO: implement từng bước ở services/
    """
    logger.info(
        "Chat request",
        extra={
            "event": "chat_request",
            "user_id": user.user_id,
            "mode": body.mode,
            "session_id": body.session_id,
        },
    )

    # ── Placeholder — sẽ replace bằng RAG service ────────────────────────
    # Bước 1: check token quota (service/token_service.py)
    # Bước 2: embed câu hỏi + ChromaDB search
    # Bước 3: lấy conversation history từ MongoDB/Redis
    # Bước 4: build prompt theo mode
    # Bước 5: gọi Gemini / Groq
    # Bước 6: lưu lịch sử, cộng token
    raise HTTPException(status_code=501, detail="RAG pipeline chưa implement — coming next step")


@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Lấy lịch sử chat của 1 session (10 tin gần nhất)."""
    # TODO: query MongoDB
    return {"session_id": session_id, "messages": [], "user_id": user.user_id}


@router.delete("/history/{session_id}")
async def clear_history(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Xóa lịch sử để bắt đầu phiên mới."""
    # TODO: xóa Redis cache + MongoDB
    return {"message": "Cleared", "session_id": session_id}