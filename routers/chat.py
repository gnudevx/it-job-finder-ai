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
    resolve_session_id,
    resolve_cv_id,
    get_session_cv_id,
    mark_session_cleared,
    should_use_active_cv,
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
    try:
        resolved_session_id = resolve_session_id(body.session_id, user.user_id)

        logger.info("Chat request", extra={
            "event": "chat_request",
            "user_id": user.user_id,
            "mode": body.mode,
            "session_id": resolved_session_id,
        })

        # 1. Xác định cv_id — ưu tiên CV active mới nhất của user
        if should_use_active_cv(resolved_session_id, user.user_id):
            cv_id = resolve_cv_id(body.cv_id, user.user_id)
        else:
            cv_id = body.cv_id

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
        job_results: list = []

        if detected_intent == "faq":
            # FAQ mode: tìm job từ DB, có thể kết hợp CV context nếu có
            try:
                from services.jobs.job_search_service import (
                    search_jobs_from_message,
                    get_title_matched_jobs,
                    count_jobs_from_message,
                    format_jobs_for_llm,
                    build_faq_messages,
                    get_active_job_suggestions,
                    _get_jobs_collection,
                )
                # Try title-matched jobs first (exact title match)
                job_results = get_title_matched_jobs(body.message, limit=3)
                is_suggestion = False
                
                # If no title match found, fallback to semantic search
                if not job_results:
                    job_results = search_jobs_from_message(body.message)
                    
                # If still no results, use suggestions
                if not job_results:
                    col = _get_jobs_collection()
                    job_results = get_active_job_suggestions(col.database)
                    is_suggestion = True

                total_active = count_jobs_from_message("")
                matched_active = count_jobs_from_message(body.message) if not is_suggestion else total_active

                job_context = format_jobs_for_llm(
                    job_results,
                    is_suggestion=is_suggestion,
                    total_count=total_active,
                    matched_count=matched_active
                )
                # Note: title-match stats now added later in deterministic summary (avoid duplication)
                messages = build_faq_messages(resolved_session_id, body.message, job_context)
                
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
                    session_id=resolved_session_id,
                    user_id=user.user_id,
                    cv_id=cv_id,
                )
                
                from services.jobs.job_search_service import search_jobs_from_message, count_jobs_from_message, format_jobs_for_llm
                job_results = search_jobs_from_message(body.message)
                if job_results:
                    total_active = count_jobs_from_message("")
                    matched_active = count_jobs_from_message(body.message)
                    job_context = format_jobs_for_llm(
                        job_results,
                        is_suggestion=False,
                        total_count=total_active,
                        matched_count=matched_active
                    )
                    # Note: title-match stats now added later in deterministic summary (avoid duplication)
                    logger.info(
                        "Hybrid mode (CV/Interview + Job): loaded job context",
                        extra={"event": "hybrid_job_load", "jobs_found": len(job_results)},
                    )
            except Exception as e:
                logger.exception("RAG / Hybrid load failed", extra={"user_id": user.user_id})
                raise HTTPException(status_code=500, detail="Lỗi khi lấy dữ liệu CV/Job")


        # ── Gọi LLM ──────────────────────────────────────────────────────────────
        # Build deterministic summary (counts + top-3 jobs) to ensure numerical answers
        deterministic_summary = ""
        try:
            from services.jobs.job_search_service import (
                count_title_matches_by_message,
                count_title_noexp_matches_by_message,
                count_jobs_from_message,
            )

            # compute totals
            total_active = locals().get("total_active")
            if total_active is None:
                total_active = count_jobs_from_message("")

            matched_active = locals().get("matched_active")
            if matched_active is None:
                matched_active = count_jobs_from_message(body.message)

            title_stats = count_title_matches_by_message(body.message)
            noexp_count = count_title_noexp_matches_by_message(body.message)

            parts = []
            if title_stats and title_stats.get("count", 0) > 0:
                term = title_stats.get("term") or body.message
                parts.append(
                    f"Tìm cụm '{term}' trong tiêu đề → {title_stats.get('count')} công việc (số tiêu đề khác nhau: {title_stats.get('distinct_titles_len')})."
                )
                # Use title match count instead of semantic count for accuracy
                display_matched = title_stats.get("count")
            else:
                display_matched = matched_active

            parts.append(f"Số công việc khớp với yêu cầu tìm kiếm: {display_matched} (tổng công việc active: {total_active}).")

            if noexp_count and noexp_count > 0:
                parts.append(f"Trong đó có {noexp_count} công việc không yêu cầu kinh nghiệm.")

            # Append top-3 job summaries (if any)
            if job_results:
                parts.append("\nTop công việc (tối đa 3):")
                for i, job in enumerate(job_results[:3], start=1):
                    parts.append(
                        f"[{i}] {job.get('title')} — {job.get('company')} — Kinh nghiệm: {job.get('experience')} — Lương: {job.get('salary')} — Hạn nộp: {job.get('deadline')} — Link: /jobs/{job.get('id')}"
                    )

            deterministic_summary = "\n".join(parts)
        except Exception as e:
            logger.warning(f"Failed to build deterministic summary: {e}")

        # If we have a deterministic summary from DB, return it directly for count-like questions
        # to ensure authoritative numeric answers and avoid duplicated summaries.

        if deterministic_summary and any(word in body.message.lower() for word in ["bao nhiêu", "số lượng", "how many", "count"]):
            try:
                usage = get_token_usage(user.user_id)
            except Exception as e:
                logger.warning(f"Token usage failed: {e}")

                usage = {
                    "used": 0,
                    "remaining": 0,
                    "warning": None,
                }

            assistant_reply = deterministic_summary

            try:
                save_message(
                    body.session_id,
                    user.user_id,
                    "user",
                    body.message
                )

                save_message(
                    body.session_id,
                    user.user_id,
                    "assistant",
                    assistant_reply
                )

            except Exception as e:
                logger.error(f"Save history failed: {e}")

            return ChatResponse(
                reply=assistant_reply,
                session_id=resolved_session_id,
                tokens_used=usage.get("used", 0),
                tokens_remaining=usage.get("remaining", 0),
                warning=usage.get("warning"),
                detected_intent=detected_intent,
                jobs=(job_results or [])[:3],
            )

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

        # Use LLM reply as final answer (deterministic summary only for FAQ short-circuit above)
        assistant_reply = result.get("reply") or ""

        # ── Lưu lịch sử vào MongoDB ───────────────────────────────────────────────
        try:
            save_message(resolved_session_id, user.user_id, "user", body.message)
            save_message(resolved_session_id, user.user_id, "assistant", assistant_reply)
        except Exception as e:
            logger.error(f"Save history failed: {e}")

        # ── Trả về ───────────────────────────────────────────────────────────────
        return ChatResponse(
            reply=assistant_reply,
            session_id=resolved_session_id,
            tokens_used=result["tokens_used"],
            tokens_remaining=result["tokens_remaining"],
            warning=result.get("warning"),
            detected_intent=detected_intent,    # FE dùng để hiện thông báo context switch
            jobs=(job_results or [])[:3],
        )

    except Exception as e:
        import traceback

        logger.error("===== CHAT ERROR =====")
        logger.error(str(e))
        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@router.get("/history/{session_id}")
async def get_chat_history(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Lấy lịch sử chat của 1 session."""
    resolved_session_id = resolve_session_id(session_id, user.user_id)
    messages = get_history(resolved_session_id, limit=50)
    return {"session_id": resolved_session_id, "messages": messages, "count": len(messages)}


@router.delete("/history/{session_id}")
async def delete_history(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Xóa lịch sử để bắt đầu phiên mới."""
    resolved_session_id = resolve_session_id(session_id, user.user_id)
    clear_history(resolved_session_id)
    mark_session_cleared(resolved_session_id, user.user_id)
    return {"message": "Đã xóa lịch sử", "session_id": resolved_session_id}


@router.get("/tokens")
async def token_usage(
    user: CurrentUser = Depends(get_current_user),
):
    """Xem quota token hôm nay của user."""
    usage = get_token_usage(user.user_id)
    return usage