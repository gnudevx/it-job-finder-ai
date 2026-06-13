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
from services.AI.intent_service import detect_intent, did_context_switch

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """
    Luồng:
    1. Intent detection: phát hiện ý định thực sự của tin nhắn (có thể khác mode)
    2. Route theo intent:
       - faq          → tìm job từ MongoDB, không dùng RAG CV
       - cv_advisor   → RAG từ ChromaDB
       - mock_interview → RAG từ ChromaDB
    3. Gọi LLM với context phù hợp
    4. Lưu cả 2 tin nhắn vào MongoDB
    5. Trả về reply + detected_intent (để FE biết context switch)
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
        logger.info("No CV found for user", extra={"user_id": user.user_id})

    # ── Intent Detection ─────────────────────────────────────────────────────
    # Phát hiện ý định thực sự của tin nhắn (bidirectional — không bị lock theo mode)
    try:
        detected_intent = detect_intent(body.message, body.mode)
        switched = did_context_switch(detected_intent, body.mode)
        if switched:
            logger.info(
                "Context switch detected",
                extra={
                    "event": "context_switch",
                    "from": body.mode,
                    "to": detected_intent,
                    "user_id": user.user_id,
                },
            )
    except Exception as e:
        logger.warning(f"Intent detection failed, using original mode: {e}")
        detected_intent = body.mode

    # ── Route theo detected_intent ────────────────────────────────────────────
    messages: list[dict] = []
    cv_context = ""
    job_context = ""

    if detected_intent == "faq":
        # FAQ mode: tìm job từ DB, có thể kết hợp CV context nếu có
        try:
            from services.jobs.job_search_service import (
                search_jobs_from_message,
                format_jobs_for_llm,
                build_faq_messages,
                get_active_job_suggestions,
                _get_jobs_collection,
            )
            job_results = search_jobs_from_message(body.message)
            is_suggestion = False
            if not job_results:
                col = _get_jobs_collection()
                job_results = get_active_job_suggestions(col.database)
                is_suggestion = True

            job_context = format_jobs_for_llm(job_results, is_suggestion=is_suggestion)
            messages = build_faq_messages(body.session_id, body.message, job_context)
            
            # Hybrid: load thêm CV context để AI trả lời có cá nhân hóa dựa trên CV
            if cv_id:
                from services.CV.rag.rag_service import retrieve_cv_context
                cv_context = retrieve_cv_context(body.message, user.user_id, cv_id)
                
            logger.info(
                "FAQ mode: job search done",
                extra={"event": "faq_search", "jobs_found": len(job_results), "has_cv": bool(cv_context)},
            )
        except Exception as e:
            logger.exception("Job search failed in FAQ mode", extra={"user_id": user.user_id})
            # Graceful fallback: vẫn gọi LLM nhưng không có job data
            messages = [{"role": "user", "content": body.message}]
    else:
        # CV / Interview mode: RAG từ ChromaDB
        try:
            messages, cv_context = build_rag_messages(
                user_message=body.message,
                session_id=body.session_id,
                user_id=user.user_id,
                cv_id=cv_id,
            )
            
            # Hybrid: Nếu trong câu hỏi có đề cập tìm job/vị trí cụ thể, lấy thêm job_context từ MongoDB
            from services.jobs.job_search_service import search_jobs_from_message, format_jobs_for_llm
            job_results = search_jobs_from_message(body.message)
            if job_results:
                job_context = format_jobs_for_llm(job_results)
                logger.info(
                    "Hybrid mode (CV/Interview + Job): loaded job context",
                    extra={"event": "hybrid_job_load", "jobs_found": len(job_results)},
                )
        except Exception as e:
            logger.exception("RAG / Hybrid load failed", extra={"user_id": user.user_id})
            raise HTTPException(status_code=500, detail="Lỗi khi lấy dữ liệu CV/Job")


    # ── Gọi LLM ──────────────────────────────────────────────────────────────
    try:
        result = chat_completion(
            messages=messages,
            user_id=user.user_id,
            mode=detected_intent,           # dùng intent thực tế, không phải body.mode
            job_position=body.job_position,
            cv_context=cv_context,
            job_context=job_context,
        )
    except Exception as e:
        logger.exception("LLM failed", extra={"user_id": user.user_id})
        raise HTTPException(status_code=502, detail="Lỗi khi gọi AI model")

    # ── Lưu lịch sử vào MongoDB ───────────────────────────────────────────────
    save_message(body.session_id, user.user_id, "user", body.message)
    save_message(body.session_id, user.user_id, "assistant", result["reply"])

    # ── Trả về ───────────────────────────────────────────────────────────────
    return ChatResponse(
        reply=result["reply"],
        session_id=body.session_id,
        tokens_used=result["tokens_used"],
        tokens_remaining=result["tokens_remaining"],
        warning=result.get("warning"),
        detected_intent=detected_intent,    # FE dùng để hiện thông báo context switch
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