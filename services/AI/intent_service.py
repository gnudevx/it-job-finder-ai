"""
services/AI/intent_service.py — Bidirectional Intent Detection

Phát hiện ý định (intent) của mỗi tin nhắn, KHÔNG bị lock theo mode hiện tại.
Cho phép context switch tự do giữa 3 luồng:
  - cv_advisor    : hỏi về CV, phân tích hồ sơ
  - mock_interview: luyện phỏng vấn
  - faq           : tìm kiếm job, hỏi về tuyển dụng

Chiến lược:
  1. Rule-based keyword scoring (nhanh, free, không tốn token)
  2. Nếu score cao nhất < CONFIDENCE_THRESHOLD → LLM classify (~50 token)
  3. Nếu vẫn không rõ → giữ nguyên current_mode
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

# ── Threshold ────────────────────────────────────────────────────────────────
# Score cần đạt để switch intent (tránh switch nhầm vì 1 từ ngẫu nhiên)
CONFIDENCE_THRESHOLD = 2

# ── Keyword dictionaries ──────────────────────────────────────────────────────

FAQ_KEYWORDS = [
    # Tìm việc chủ động
    "tìm job", "tìm việc", "có job", "có việc làm", "job nào",
    "còn tuyển", "đang tuyển", "công ty nào tuyển", "vị trí nào còn",
    # Câu hỏi về job cụ thể
    "lương bao nhiêu", "mức lương", "salary", "thu nhập",
    "địa điểm làm việc", "làm ở đâu", "remote", "onsite",
    "deadline nộp", "hạn nộp", "ứng tuyển vào", "apply vào",
    # Từ khoá job search
    "job ", "jobs ", "tuyển dụng", "cơ hội việc làm", "fresher",
    "senior", "junior", "intern", "thực tập",
    "backend", "frontend", "fullstack", "devops", "ai engineer",
    "data engineer", "mobile", "react", "python developer",
    # Hỏi về thị trường
    "thị trường tuyển dụng", "xu hướng tuyển", "nhu cầu tuyển",
]

CV_KEYWORDS = [
    # Đề cập CV/hồ sơ của bản thân
    "cv của tôi", "cv tôi", "hồ sơ của tôi", "hồ sơ tôi",
    "review cv", "phân tích cv", "nhận xét cv", "đánh giá cv",
    "cv có ổn", "cv ổn không", "cv như thế nào",
    "cải thiện cv", "chỉnh sửa cv", "viết lại cv",
    # Nội dung CV
    "kinh nghiệm của tôi", "kỹ năng của tôi",
    "điểm mạnh của tôi", "điểm yếu của tôi",
    "tôi thiếu", "tôi cần bổ sung",
    "ats", "format cv", "template cv",
]

INTERVIEW_KEYWORDS = [
    # Phỏng vấn thử
    "phỏng vấn thử", "mock interview", "luyện phỏng vấn",
    "bắt đầu phỏng vấn", "thử phỏng vấn", "tập phỏng vấn",
    "phỏng vấn",
    # Câu hỏi phỏng vấn
    "câu hỏi phỏng vấn", "hỏi về", "interviewer hỏi",
    "trả lời câu hỏi", "trả lời phỏng vấn", "trả lời thế nào",
    "nếu được hỏi", "khi phỏng vấn",
    # Kỹ thuật
    "technical interview", "behavioral question",
    "star method", "situational question",
    # Tiếp tục session đang chạy
    "câu tiếp theo", "câu hỏi tiếp", "hỏi thêm", "tiếp tục phỏng vấn",
]



# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_message(message: str) -> dict[str, int]:
    """
    Tính score cho mỗi intent dựa trên keyword matching.
    Score = số lượng keyword khớp (mỗi keyword tính 1 điểm).
    """
    msg = message.lower()
    scores = {
        "faq": sum(1 for kw in FAQ_KEYWORDS if kw in msg),
        "cv_advisor": sum(1 for kw in CV_KEYWORDS if kw in msg),
        "mock_interview": sum(1 for kw in INTERVIEW_KEYWORDS if kw in msg),
    }
    return scores


def _llm_classify(message: str, current_mode: str) -> str:
    """
    Dùng Gemini để classify intent khi rule-based không đủ confident.
    Chỉ gọi khi max score < CONFIDENCE_THRESHOLD.
    Tiêu thụ ~50-80 tokens.
    """
    try:
        from core.config import settings
        from google import genai

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        prompt = f"""Phân loại ý định của tin nhắn sau thành MỘT trong 3 loại:
- "faq": hỏi về job/việc làm, tìm kiếm công việc, lương, tuyển dụng
- "cv_advisor": hỏi về CV/hồ sơ cá nhân, phân tích, cải thiện
- "mock_interview": luyện phỏng vấn, câu hỏi phỏng vấn

Tin nhắn: "{message}"
Mode hiện tại: {current_mode}

Chỉ trả về đúng 1 từ: faq | cv_advisor | mock_interview"""

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config={"max_output_tokens": 20, "temperature": 0.0},
        )

        result = (response.text or "").strip().lower()
        if result in ("faq", "cv_advisor", "mock_interview"):
            logger.info(
                "LLM intent classify",
                extra={"event": "intent_llm", "result": result, "message_preview": message[:50]},
            )
            return result

    except Exception as e:
        logger.warning(f"LLM intent classify failed, falling back to current_mode: {e}")

    return current_mode


# ── Public API ────────────────────────────────────────────────────────────────

IntentType = Literal["cv_advisor", "mock_interview", "faq"]


def detect_intent(message: str, current_mode: str) -> IntentType:
    """
    Phát hiện intent thực sự của tin nhắn — bidirectional, không bị lock theo mode.

    Args:
        message: tin nhắn của user
        current_mode: mode đang chạy ("cv_advisor" | "mock_interview" | "faq")

    Returns:
        intent thực tế: "cv_advisor" | "mock_interview" | "faq"

    Logic:
        1. Score mỗi intent bằng keyword matching
        2. Nếu top score >= CONFIDENCE_THRESHOLD → dùng luôn
        3. Nếu top score < CONFIDENCE_THRESHOLD → gọi LLM classify
        4. Cuối cùng fallback về current_mode
    """
    scores = _score_message(message)
    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    logger.info(
        "Intent scores",
        extra={
            "event": "intent_scored",
            "scores": scores,
            "best": best_intent,
            "score": best_score,
            "current_mode": current_mode,
        },
    )

    if best_score >= CONFIDENCE_THRESHOLD:
        # Rule-based confident — switch nếu khác mode hiện tại
        if best_intent != current_mode:
            logger.info(
                "Intent context switch (rule-based)",
                extra={
                    "event": "intent_switch",
                    "from": current_mode,
                    "to": best_intent,
                    "score": best_score,
                },
            )
        return best_intent  # type: ignore[return-value]

    if best_score == 1:
        # Có 1 keyword khớp nhưng chưa confident → LLM classify
        logger.info("Intent ambiguous, calling LLM classify", extra={"event": "intent_ambiguous"})
        return _llm_classify(message, current_mode)  # type: ignore[return-value]

    # Score = 0 (không khớp keyword nào) → giữ nguyên mode hiện tại
    return current_mode  # type: ignore[return-value]


def did_context_switch(detected_intent: str, original_mode: str) -> bool:
    """Helper: kiểm tra xem có xảy ra context switch không."""
    return detected_intent != original_mode
