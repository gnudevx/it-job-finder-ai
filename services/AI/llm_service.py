"""
LLMService — Smart routing + fallback.

Chiến lược:
  cv_advisor   → Gemini 1.5 Flash (giỏi đọc hiểu dài, phân tích)
                 Nếu Gemini rate limit (429) → fallback Groq

  mock_interview → Groq Llama 3 (nhanh hơn, real-time feel)
                   Nếu Groq lỗi → fallback Gemini

Token tracking: Redis, reset mỗi ngày, cảnh báo 90%.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Optional
from typing import cast
from google import genai
from groq import Groq, RateLimitError as GroqRateLimitError
from google.genai.errors import ClientError as GeminiClientError
from groq.types.chat import ChatCompletionMessageParam
from core.config import settings
from redis.client import Redis
from google.genai import types
logger = logging.getLogger(__name__)

# ── Singleton clients ─────────────────────────────────────────────────────────
_groq_client: Optional[Groq] = None
_redis_client = None

DAILY_TOKEN_LIMIT = settings.DAILY_TOKEN_LIMIT
WARNING_THRESHOLD = settings.TOKEN_WARNING_THRESHOLD

_gemini_client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)

def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client


def _get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis_client


# ── Token tracking ────────────────────────────────────────────────────────────

def _token_key(user_id: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"tokens:{user_id}:{today}"


def get_token_usage(user_id: str) -> dict:
    r = _get_redis()

    value = cast(str | None, r.get(_token_key(user_id)))

    used = int(value) if value is not None else 0

    remaining = max(0, DAILY_TOKEN_LIMIT - used)

    return {
        "used": used,
        "limit": DAILY_TOKEN_LIMIT,
        "remaining": remaining,
        "warning": used / DAILY_TOKEN_LIMIT >= WARNING_THRESHOLD,
    }


def _add_tokens(user_id: str, count: int) -> dict:
    r = _get_redis()

    key = _token_key(user_id)

    new_total = cast(int, r.incrby(key, count))

    r.expire(key, 90000)

    remaining = max(0, DAILY_TOKEN_LIMIT - new_total)

    warning = (
        new_total / DAILY_TOKEN_LIMIT >= WARNING_THRESHOLD
    )

    return {
        "used": new_total,
        "remaining": remaining,
        "warning": warning,
    }


# ── System prompts ────────────────────────────────────────────────────────────

def build_system_prompt(
    mode: str,
    job_position: str | None = None,
    cv_context: str = "",
) -> str:
    base = "Bạn là trợ lý AI chuyên về IT career. Trả lời ngắn gọn, thực tế bằng tiếng Việt."

    if mode == "cv_advisor":
        prompt = base + """

Vai trò: Chuyên gia HR phân tích CV IT.
Khi nhận context CV hãy:
1. Chỉ ra điểm mạnh cụ thể (kỹ năng, kinh nghiệm nổi bật)
2. Chỉ ra điểm yếu, thiếu sót (từ khóa ATS, format, số liệu cụ thể)  
3. Gợi ý cải thiện từng phần rõ ràng
Quan trọng: không bịa thông tin. Chỉ nhận xét dựa trên CV thực tế.
Nếu không có CV → yêu cầu upload trước."""

    elif mode == "mock_interview":
        pos = job_position or "Software Engineer"
        prompt = base + f"""


Vai trò: Interviewer tại công ty công nghệ lớn, phỏng vấn vị trí {pos}.
Quy tắc:
- Hỏi từng câu một, đợi trả lời rồi mới hỏi tiếp
- Bắt đầu: "Hãy giới thiệu về bản thân bạn"
- Xen kẽ technical + behavioral questions
- Sau mỗi câu trả lời: nhận xét ngắn (tốt/cần cải thiện) rồi hỏi tiếp
- Sau 5-7 câu: tổng kết điểm mạnh/yếu của ứng viên
Dựa vào CV để hỏi câu hỏi phù hợp với kinh nghiệm thực tế."""

    else:
        prompt = base
    # CV đưa vào system prompt → LLM đọc trước tiên
    
    # CV đưa vào system prompt → LLM đọc trước tiên
    if cv_context:
        prompt += f"""

    === CV ỨNG VIÊN ===
    {cv_context}
    === HẾT CV ===
    Phân tích dựa trên CV thực tế ở trên, không được nói "không có thông tin"."""

    return prompt

# ── Provider implementations ──────────────────────────────────────────────────

def _call_gemini(
    messages: list[dict],
    system_prompt: str
) -> tuple[str, int]:

    contents = []

    contents.append({
        "role": "user",
        "parts": [{"text": system_prompt}]
    })

    for m in messages:
        role = "model" if m["role"] == "assistant" else "user"

        contents.append({
            "role": role,
            "parts": [{"text": m["content"]}]
        })

    response = _gemini_client.models.generate_content(
        model="gemini-2.0-flash",   # ✅ ĐÚNG — giữ nguyên cái này
        contents=contents,
        config={
            "system_instruction": system_prompt,
            "max_output_tokens": 600,
            "temperature": 0.7,
        },
    )

    reply = response.text or ""

    token_count = (
        response.usage_metadata.total_token_count
        if response.usage_metadata
        else None
    )

    tokens = token_count if token_count is not None else 500

    return reply, tokens


def _call_groq(
    messages: list[dict],
    system_prompt: str
) -> tuple[str, int]:

    full_messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    for m in messages:
        full_messages.append({
            "role": m["role"],
            "content": m["content"],
        })

    client = _get_groq()

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=full_messages,
        max_tokens=1000,
        temperature=0.7,
    )

    reply = response.choices[0].message.content or ""

    tokens = (
        response.usage.total_tokens
        if response.usage
        else 500
    )

    return reply, tokens


# ── Smart router ──────────────────────────────────────────────────────────────

def chat_completion(
    messages: list[dict],
    user_id: str,
    mode: str = "cv_advisor",
    job_position: str | None = None,
    cv_context: str = "",
) -> dict:
    """
    Smart routing + fallback:
      cv_advisor    → Gemini primary,  Groq fallback
      mock_interview → Groq primary,   Gemini fallback

    Returns:
      {"reply", "tokens_used", "tokens_remaining", "warning", "model_used"}
    """
    # 1. Check quota trước
    quota = get_token_usage(user_id)
    if quota["remaining"] <= 0:
        return {
            "reply": "⚠️ Bạn đã dùng hết quota token hôm nay. Thử lại vào ngày mai.",
            "tokens_used": quota["used"],
            "tokens_remaining": 0,
            "warning": "Hết quota",
            "model_used": None,
        }

    system_prompt = build_system_prompt(mode, job_position, cv_context)

    # 2. Chọn thứ tự provider theo mode
    # Use simple (name, call_fn) tuples and a rate-limit detector.
    if mode == "cv_advisor":
        primary_fn = ("gemini", _call_gemini)
        fallback_fn = ("groq", _call_groq)
    else:
        primary_fn = ("groq", _call_groq)
        fallback_fn = ("gemini", _call_gemini)

    def _is_rate_limit_exc(exc: Exception) -> bool:
        """Return True if exception looks like a rate-limit (429).

        We avoid depending on SDK-specific exception classes to keep
        behavior robust across versions. Check common indicators.
        """
        # Groq explicit exception class
        try:
            if isinstance(exc, GroqRateLimitError):
                return True
        except Exception:
            pass

        # Gemini / Google SDK may include a status_code or http_status
        code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
        if isinstance(code, int) and code == 429:
            return True

        # Fallback textual check
        msg = str(exc).lower()
        if "rate" in msg and ("limit" in msg or "429" in msg):
            return True

        return False

    # 3. Gọi primary, nếu rate limit → fallback
    reply = None
    tokens = 0
    model_used = None
    start = time.time()

    try:
        model_name, call_fn = primary_fn
        reply, tokens = call_fn(messages, system_prompt)
        model_used = model_name
        logger.info("LLM primary success", extra={
            "event": "llm_primary_ok", "model": model_name,
            "user_id": user_id, "mode": mode,
        })

    except Exception as e:
        # If primary failed due to rate-limiting, attempt fallback.
        if _is_rate_limit_exc(e):
            fallback_name, fallback_call = fallback_fn
            logger.warning("Primary rate limited, switching to fallback", extra={
                "event": "llm_fallback", "primary": primary_fn[0],
                "fallback": fallback_name, "user_id": user_id,
            })
            try:
                reply, tokens = fallback_call(messages, system_prompt)
                model_used = fallback_name
            except Exception as e2:
                logger.exception("Fallback also failed", extra={"event": "llm_both_failed"})
                raise RuntimeError(f"Cả 2 model đều lỗi: {e2}")
        else:
            logger.exception("LLM call failed", extra={
                "event": "llm_call_failed", "user_id": user_id, "error": str(e),
            })
            raise

    elapsed = round((time.time() - start) * 1000)
    logger.info("LLM call done", extra={
        "event": "llm_call_done", "model": model_used,
        "tokens": tokens, "duration_ms": elapsed, "mode": mode,
    })

    # 4. Cộng token vào Redis
    token_info = _add_tokens(user_id, tokens)

    warning_msg = None
    if token_info["warning"]:
        pct = round(token_info["used"] / DAILY_TOKEN_LIMIT * 100)
        warning_msg = (
            f"⚠️ Đã dùng {pct}% quota hôm nay "
            f"({token_info['used']}/{DAILY_TOKEN_LIMIT} tokens)"
        )

    return {
        "reply": reply,
        "tokens_used": token_info["used"],
        "tokens_remaining": token_info["remaining"],
        "warning": warning_msg,
        "model_used": model_used,   # để debug biết dùng model nào
    }