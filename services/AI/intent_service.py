"""
services/AI/intent_service.py — Bidirectional Intent Detection

Phát hiện ý định (intent) của mỗi tin nhắn, KHÔNG bị lock theo mode hiện tại.
Cho phép context switch tự do giữa 3 luồng:
  - cv_advisor    : hỏi về CV, phân tích hồ sơ
  - mock_interview: luyện phỏng vấn
  - faq           : tìm kiếm job, hỏi về tuyển dụng

Chiến lược:
  1. Rule-based keyword scoring, chia 2 tầng:
       - STRONG: cụm từ thể hiện Ý ĐỊNH rõ ràng, dứt khoát (VD "tìm việc", "review cv")
                 → chỉ cần khớp 1 lần là đủ tin cậy, tính 3 điểm.
       - WEAK  : danh từ/chủ đề chuyên ngành, dễ xuất hiện "vô tình" trong nội dung
                 kỹ thuật không liên quan (VD "mức lương", "tuyển dụng", "data engineer")
                 → chỉ tính 1 điểm, không đủ để tự switch một mình.
     Lý do tách: nếu chỉ đếm số lượng keyword khớp mà không phân biệt độ mạnh, một câu
     trả lời phỏng vấn kỹ thuật dài (nói về dự án phân tích thị trường tuyển dụng, mức
     lương...) có thể vô tình khớp NHIỀU weak keyword và bị hiểu nhầm là ý định rõ ràng,
     trong khi một câu ngắn gọn "tôi muốn tìm việc" chỉ khớp ĐÚNG 1 keyword và bị coi là
     chưa đủ tin cậy. Việc đếm số lượng không phản ánh đúng độ mạnh của ý định.
  2. Nếu score cao nhất < CONFIDENCE_THRESHOLD → LLM classify (~50 token)
  3. Nếu vẫn không rõ → giữ nguyên current_mode
"""

import logging
from typing import Literal

logger = logging.getLogger(__name__)

# ── Threshold ────────────────────────────────────────────────────────────────
# Score cần đạt để switch intent (tránh switch nhầm vì 1 từ ngẫu nhiên)
# Với thang điểm mới: 1 strong keyword = 3 điểm (luôn đủ ngưỡng),
# weak keyword = 1 điểm/keyword (cần ít nhất 2 weak khớp mới đủ ngưỡng).
CONFIDENCE_THRESHOLD = 2

# Điểm tối thiểu 1 keyword STRONG đóng góp — cố tình > CONFIDENCE_THRESHOLD
# để 1 lần khớp strong là đủ, không cần cộng dồn.
STRONG_WEIGHT = 3
WEAK_WEIGHT = 1

# ── Keyword dictionaries ──────────────────────────────────────────────────────
# STRONG = ý định hành động rõ ràng, dứt khoát — chỉ cần khớp 1 lần là đủ tin cậy.
# WEAK   = danh từ/chủ đề chuyên ngành — dễ xuất hiện "vô tình" trong câu trả lời
#          kỹ thuật (mock_interview) hoặc câu hỏi khác, KHÔNG được tự switch một mình.

FAQ_STRONG_KEYWORDS = [
    # Tìm việc chủ động — ý định action rõ ràng
    "tìm job", "tìm việc", "có job", "có việc làm", "job nào",
    "còn tuyển", "đang tuyển", "công ty nào tuyển", "vị trí nào còn",
    "ứng tuyển vào", "apply vào",
    # Câu hỏi cụ thể về 1 job đã xem (deadline/hạn nộp gắn với hành động apply)
    "deadline nộp", "hạn nộp",
]

FAQ_WEAK_KEYWORDS = [
    # Chủ đề lương/địa điểm — có thể chỉ là mô tả trong câu chuyện, không phải hỏi job
    "lương bao nhiêu", "mức lương", "salary", "thu nhập",
    "địa điểm làm việc", "làm ở đâu", "remote", "onsite",
    # Danh từ ngành/job-title — RẤT dễ xuất hiện trong CV/mock_interview
    # (VD ứng viên tự mô tả stack "data engineer", "react", "python developer"...)
    "job ", "jobs ", "tuyển dụng", "cơ hội việc làm", "fresher",
    "senior", "junior", "intern", "thực tập",
    "backend", "frontend", "fullstack", "devops", "ai engineer",
    "data engineer", "mobile", "react", "python developer",
    # Hỏi về thị trường (bỏ "xu hướng tuyển" vì trùng substring với "tuyển dụng"
    # → tránh 1 cụm bị đếm 2 lần làm điểm ảo)
    "thị trường tuyển dụng", "nhu cầu tuyển",
]

CV_STRONG_KEYWORDS = [
    # Yêu cầu hành động trực tiếp lên CV — ý định rõ ràng, không thể nhầm lẫn
    "review cv", "phân tích cv", "nhận xét cv", "đánh giá cv",
    "cv có ổn", "cv ổn không", "cv như thế nào",
    "cải thiện cv", "chỉnh sửa cv", "viết lại cv",
    "dựa trên cv", "từ cv của tôi", "trong cv của tôi", "cv của tôi có",
    "cv của tôi", "cv tôi", "hồ sơ của tôi", "hồ sơ tôi",
]

CV_WEAK_KEYWORDS = [
    # Các câu tự mô tả bản thân ở ngôi thứ nhất — ĐÂY LÀ NGÔN NGỮ TỰ NHIÊN của
    # một câu trả lời phỏng vấn ("tôi đã làm ở...", "dự án của tôi...").
    # Không được coi là đủ để tự thoát mock_interview.
    "kinh nghiệm của tôi", "kỹ năng của tôi",
    "điểm mạnh của tôi", "điểm yếu của tôi",
    "tôi thiếu", "tôi cần bổ sung",
    "ats", "format cv", "template cv",
    "tôi đã làm ở", "tôi đã làm việc", "tôi đã từng làm",
    "kinh nghiệm làm việc của tôi", "số năm kinh nghiệm của tôi",
    "tôi đã làm ở những đâu", "tôi đã làm ở đâu",
    "tôi có bao nhiêu năm", "bao nhiêu năm kinh nghiệm",
    "dự án của tôi", "công ty tôi đã làm",
    "tôi học ở đâu", "trình độ học vấn của tôi",
    "chứng chỉ của tôi", "bằng cấp của tôi",
    "công nghệ tôi biết", "ngôn ngữ tôi dùng",
]

INTERVIEW_KEYWORDS = [
    # Yêu cầu bắt đầu phỏng vấn — cụm dài score cao hơn
    "phỏng vấn thử", "mock interview", "luyện phỏng vấn",
    "bắt đầu phỏng vấn", "thử phỏng vấn", "tập phỏng vấn",
    "muốn phỏng vấn", "phỏng vấn tôi", "hỏi tôi",
    "bắt đầu hỏi", "bạn hỏi tôi", "hãy hỏi tôi",
    "câu hỏi phỏng vấn", "interview",
    # Trả lời / trong khi phỏng vấn
    "trả lời câu hỏi", "trả lời phỏng vấn", "trả lời thế nào",
    "nếu được hỏi", "khi phỏng vấn",
    "interviewer hỏi", "nhà tuyển dụng hỏi",
    # Kỹ thuật
    "technical interview", "behavioral question",
    "star method", "situational question",
    # Tiếp tục session đang chạy
    "câu tiếp theo", "câu hỏi tiếp", "hỏi thêm", "tiếp tục phỏng vấn",
    "câu hỏi tiếp theo", "tiếp tục hỏi", "hỏi câu tiếp",
    # Nhận xét / feedback sau câu trả lời
    "nhận xét câu trả lời", "đánh giá câu trả lời", "phản hồi",
]



# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_message(message: str) -> dict[str, dict[str, int]]:
    """
    Tính score cho mỗi intent, tách riêng số lượng keyword STRONG và WEAK khớp được.

    Trả về:
        {
          "faq":            {"strong": int, "weak": int, "score": int},
          "cv_advisor":      {...},
          "mock_interview":  {...},   # interview giữ 1 tầng duy nhất (ít rủi ro false-positive hơn)
        }
    score = strong * STRONG_WEIGHT + weak * WEAK_WEIGHT — dùng để so sánh/threshold chung.
    """
    msg = message.lower()

    def _count(keywords: list[str]) -> int:
        return sum(1 for kw in keywords if kw in msg)

    faq_strong = _count(FAQ_STRONG_KEYWORDS)
    faq_weak = _count(FAQ_WEAK_KEYWORDS)

    cv_strong = _count(CV_STRONG_KEYWORDS)
    cv_weak = _count(CV_WEAK_KEYWORDS)

    interview_hits = _count(INTERVIEW_KEYWORDS)

    return {
        "faq": {
            "strong": faq_strong,
            "weak": faq_weak,
            "score": faq_strong * STRONG_WEIGHT + faq_weak * WEAK_WEIGHT,
        },
        "cv_advisor": {
            "strong": cv_strong,
            "weak": cv_weak,
            "score": cv_strong * STRONG_WEIGHT + cv_weak * WEAK_WEIGHT,
        },
        "mock_interview": {
            "strong": interview_hits,
            "weak": 0,
            "score": interview_hits * STRONG_WEIGHT,
        },
    }


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
    detail = _score_message(message)
    scores = {k: v["score"] for k, v in detail.items()}
    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]
    best_strong = detail[best_intent]["strong"]

    logger.info(
        "Intent scores",
        extra={
            "event": "intent_scored",
            "detail": detail,
            "best": best_intent,
            "score": best_score,
            "current_mode": current_mode,
        },
    )

    # ── Bảo vệ chế độ mock_interview ──────────────────────────────────────────
    # Khi đang trong mock_interview, câu trả lời kỹ thuật của ứng viên rất dễ chứa
    # các danh từ chuyên ngành trùng với FAQ/CV (VD "data engineer", "tuyển dụng",
    # "dự án của tôi"...) mà KHÔNG hề mang ý định rời khỏi phỏng vấn.
    # Do đó điều kiện thoát không dựa vào tổng điểm (dễ bị cộng dồn ảo từ nhiều
    # weak keyword), mà bắt buộc phải có ÍT NHẤT 1 STRONG keyword — tức người dùng
    # phát ngôn một cụm ý định rõ ràng, không thể là trùng hợp ngẫu nhiên.
    if current_mode == "mock_interview":
        if best_intent in ("cv_advisor", "faq") and best_strong >= 1:
            logger.info(
                "Intent switch from mock_interview (strong keyword matched)",
                extra={
                    "event": "intent_switch",
                    "from": current_mode,
                    "to": best_intent,
                    "score": best_score,
                    "strong_hits": best_strong,
                },
            )
            return best_intent  # type: ignore[return-value]
        return "mock_interview"

    if best_score >= CONFIDENCE_THRESHOLD:
        # Rule-based confident — switch nếu khác mode hiện tại
        # (1 strong keyword luôn đạt ngưỡng này; weak keyword cần >= 2 khớp)
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
        # Đúng 1 weak keyword khớp, chưa đủ tin cậy → LLM classify
        logger.info("Intent ambiguous, calling LLM classify", extra={"event": "intent_ambiguous"})
        return _llm_classify(message, current_mode)  # type: ignore[return-value]

    # Score = 0 (không khớp keyword nào) → giữ nguyên mode hiện tại
    return current_mode  # type: ignore[return-value]



def did_context_switch(detected_intent: str, original_mode: str) -> bool:
    """Helper: kiểm tra xem có xảy ra context switch không."""
    return detected_intent != original_mode