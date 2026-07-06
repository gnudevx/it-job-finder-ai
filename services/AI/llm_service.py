"""
LLMService — Smart routing + fallback.

Chiến lược:
  cv_advisor   → Gemini 1.5 Flash (giỏi đọc hiểu dài, phân tích)
                 Nếu Gemini rate limit (429) → fallback Groq

  mock_interview → Groq Llama 3 (nhanh hơn, real-time feel)
                   Nếu Groq lỗi → fallback Gemini

Token tracking: Redis, reset mỗi ngày, cảnh báo 90%.
"""

import re
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
from google.genai import types
from pymongo import MongoClient, ReturnDocument
from pymongo.collection import Collection

logger = logging.getLogger(__name__)

# ── Singleton clients ─────────────────────────────────────────────────────────
_groq_client: Optional[Groq] = None
_mongo_client = None

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


def _get_mongo_tokens_collection() -> Collection:
    global _mongo_client
    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
            _mongo_client.admin.command('ping')
            logger.info("✅ MongoDB token client connected successfully")
        except Exception as e:
            logger.exception(f"❌ MongoDB token connection failed: {e}")
            raise
    db = _mongo_client[settings.MONGODB_DB]
    return db["user_tokens"]


# ── Token tracking ────────────────────────────────────────────────────────────

def _token_key(user_id: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"tokens:{user_id}:{today}"


def get_token_usage(user_id: str) -> dict:
    try:
        col = _get_mongo_tokens_collection()
        doc = col.find_one({"_id": _token_key(user_id)})
        used = doc.get("used", 0) if doc else 0
    except Exception as e:
        logger.error(f"Failed to get token usage from MongoDB: {e}")
        used = 0

    remaining = max(0, DAILY_TOKEN_LIMIT - used)

    return {
        "used": used,
        "limit": DAILY_TOKEN_LIMIT,
        "remaining": remaining,
        "warning": (
            f"⚠️ Đã dùng {round(used/DAILY_TOKEN_LIMIT*100)}% quota"
            if used / DAILY_TOKEN_LIMIT >= WARNING_THRESHOLD
            else None
        ),
    }


def _add_tokens(user_id: str, count: int) -> dict:
    try:
        col = _get_mongo_tokens_collection()
        doc = col.find_one_and_update(
            {"_id": _token_key(user_id)},
            {"$inc": {"used": count}, "$set": {"updated_at": datetime.now(timezone.utc)}},
            upsert=True,
            return_document=ReturnDocument.AFTER
        )
        new_total = doc.get("used", 0) if doc else count
    except Exception as e:
        logger.error(f"Failed to add tokens in MongoDB: {e}")
        new_total = count

    remaining = max(0, DAILY_TOKEN_LIMIT - new_total)
    warning = new_total / DAILY_TOKEN_LIMIT >= WARNING_THRESHOLD

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
    job_context: str = "",
) -> str:
    base = "Bạn là trợ lý AI chuyên về IT career. Trả lời ngắn gọn, thực tế bằng tiếng Việt."

    if mode == "cv_advisor":
        prompt = base + """

Vai trò: Chuyên gia HR & Tech Lead kỳ cựu chuyên phân tích và tối ưu CV IT.
Khi nhận được nội dung CV, hãy phản hồi theo đúng cấu trúc sau đây bằng tiếng Việt:

---

### 📊 ĐÁNH GIÁ CHUNG & ĐIỂM SỐ CV
- **Điểm sức khỏe CV ước lượng:** [Chấm điểm từ 0-100 dựa trên độ hoàn thiện của CV]
- **Tóm tắt chuyên môn:** [1-2 câu tóm tắt vị trí, công nghệ nổi bật nhất của ứng viên]

### 🔍 PHÂN TÍCH THEO 4 TRỤ CỘT IT CỐT LÕI
1. **Kết quả & Số liệu định lượng (Impact & Metrics):** 
   - Đánh giá xem CV có các con số định lượng (%, throughput, thời gian, số user...) chứng minh hiệu quả công việc chưa. Chỉ rõ những câu mô tả còn mơ hồ.
2. **Độ sâu Tech Stack & Phân nhóm kỹ năng (Tech Stack Depth):** 
   - Đánh giá cách sắp xếp tech stack trong CV. Các nhóm ngôn ngữ, framework, tools đã phân loại khoa học và thể hiện rõ kỹ năng cốt lõi chưa.
3. **Mức độ tối ưu từ khóa ATS (ATS Keywords):**
   - Chỉ ra chính xác 3-5 từ khóa kỹ thuật quan trọng của vị trí ứng viên còn thiếu trong CV (ví dụ: RESTful API, Database Indexing, System Design, Unit Test...).
4. **Cấu trúc Mô tả Dự án (Project Structure):**
   - Xem các dự án đã viết theo chuẩn (Bối cảnh - Công nghệ - Đóng góp cá nhân - Kết quả) chưa.

### ✍️ MẪU VIẾT LẠI THỰC TẾ (BEFORE ➜ AFTER)
[Chọn ra đúng 1 hoặc 2 câu chưa tốt trong CV thực tế của họ và viết lại mẫu để họ thay thế ngay]
- **Trước:** "[Câu mơ hồ/thiếu số liệu trong CV của họ]"
- **Sau:** "[Câu viết lại cực kỳ chuyên nghiệp, có số liệu giả định/gợi ý thêm số liệu]"

### 💬 HÃY BẮT ĐẦU CẢI THIỆN
[Đưa ra 1 câu hỏi tương tác, gợi mở để ứng viên trả lời, từ đó bạn sẽ viết lại giúp họ. Ví dụ: hỏi về kết quả/tốc độ xử lý của một dự án cụ thể xuất hiện trong CV của họ]

---

LƯU Ý QUAN TRỌNG:
- KHÔNG đưa ra nhận xét chung chung mang tính lý thuyết suông. Mọi nhận xét phải trích dẫn trực tiếp tên dự án, công nghệ hoặc nội dung có trong CV của họ.
- Không tự bịa ra các công nghệ mới ứng viên chưa từng làm trừ khi đó là đề xuất từ khóa ATS cần bổ sung.
- Nếu không có CV → yêu cầu upload trước."""

    elif mode == "mock_interview":
        pos = job_position or "Software Engineer"
        prompt = base + f"""

Vai trò: Senior Technical Interviewer tại công ty công nghệ, phỏng vấn vị trí {pos}.

NGUYÊN TẮC BẮT BUỘC — đọc kỹ trước khi đặt câu hỏi:
1. LUÔN đọc kỹ CV context được cung cấp trước khi đặt câu hỏi
2. Câu hỏi PHẢI trích dẫn trực tiếp kỹ năng/dự án/công nghệ có trong CV
   - Ví dụ đúng: "CV của bạn đề cập đến React và Node.js — bạn đã xử lý state management như thế nào trong dự án đó?"
   - Ví dụ SAI: "Bạn có kinh nghiệm với framework JavaScript nào không?" (quá chung chung)
3. KHÔNG hỏi câu "Bạn biết gì về...", "Bạn có kinh nghiệm gì về..." nếu CV đã có thông tin đó
4. KHÔNG hỏi về trường học, gia đình, sở thích, giới thiệu bản thân
5. Hỏi từng câu một, đợi trả lời rồi mới hỏi tiếp

QUY TRÌNH PHỎNG VẤN:
- Câu 1: Dựa trên dự án/kinh nghiệm GẦN NHẤT trong CV → hỏi về technical decision quan trọng nhất
- Câu 2-3: Đào sâu vào kỹ năng chuyên môn cụ thể đã nêu (công nghệ, framework, pattern...)
- Câu 4-5: Behavioral + system design liên quan đến kinh nghiệm thực trong CV
- Câu 6-7: Tình huống khó (debug, performance, conflict...) dựa trên tech stack trong CV
- Cuối: Tổng kết điểm mạnh/yếu cụ thể từ các câu trả lời

SAU MỖI CÂU TRẢ LỜI:
- Nhận xét ngắn: điểm tốt cụ thể + điểm cần cải thiện
- Đặt câu hỏi follow-up nếu câu trả lời chưa đủ depth

Nếu CV context trống hoặc không có: thông báo cần upload CV trước để phỏng vấn hiệu quả."""

    elif mode == "faq":
        prompt = base + """

Vai trò: Trợ lý tuyển dụng của IT Job Finder.
Bạn có thể tra cứu và tư vấn về các công việc đang tuyển dụng.
Quy tắc:
- Chỉ trả lời dựa trên dữ liệu jobs được cung cấp — KHÔNG bịa thêm.
- Số lượng công việc là dữ liệu tuyệt đối từ hệ thống
- Danh sách hiển thị có thể chỉ là một phần kết quả
- Không được suy luận rằng danh sách hiện tại là toàn bộ dữ liệu
- Không được tự suy luận có thêm công việc ngoài số đã cung cấp.
- Không được dùng kiến thức bên ngoài.
- Không được thêm nhận xét CV khi người dùng hỏi về việc làm.
- Chỉ mô tả các công việc thực sự có trong dữ liệu.
- Nếu không tìm thấy job phù hợp → nói thật "hiện tại chưa có vị trí này".
- Đề xuất từ khóa tìm kiếm khác nếu không có kết quả.
- Trình bày thông tin job rõ ràng: tên vị trí, địa điểm, lương, hạn nộp.
- Không tiết lộ thông tin nội bộ hệ thống."""

    else:
        prompt = base

    # CV context → đưa vào system prompt (cv_advisor / mock_interview)
    if cv_context:
        prompt += f"""

    === CV ỨNG VIÊN ===
    {cv_context}
    === HẾT CV ===
    Phân tích dựa trên CV thực tế ở trên, không được nói "không có thông tin"."""

    # Job context → đưa vào system prompt (faq mode)
    if job_context:
        prompt += f"""

=== DANH SÁCH VIỆC LÀM ĐANG TUYỂN (dữ liệu thực tế từ hệ thống) ===
{job_context}
=== HẾT DANH SÁCH ===
Hãy dựa vào danh sách trên để trả lời. Nếu không có job phù hợp hãy nói thật."""

    return prompt

# ── Anti-Hallucination Utilities ──────────────────────────────────────────────

def clean_hallucinated_job_links(reply: str, job_context: str) -> str:
    """
    Phát hiện các link /jobs/{id} trong reply mà không có trong job_context.
    Thay thế các link ảo giác bằng link chung '/jobs' để tránh lỗi 404 cho người dùng.
    """
    if not reply:
        return reply

    # Tìm các ID job hợp lệ trong job_context (ví dụ: /jobs/123456)
    # Nếu job_context trống, valid_ids sẽ trống
    valid_ids = set(re.findall(r"/jobs/([a-zA-Z0-9_-]+)", job_context)) if job_context else set()
    
    # Tìm các link /jobs/{id} trong câu trả lời của LLM
    llm_links = re.findall(r"/jobs/([a-zA-Z0-9_-]+)", reply)
    
    cleaned_reply = reply
    for job_id in llm_links:
        if job_id not in valid_ids:
            logger.warning(f"Hallucination detected: Job ID {job_id} is not in job_context.")
            # Thay thế link ảo giác bằng link tìm kiếm chung /jobs
            cleaned_reply = re.sub(
                rf"/jobs/{re.escape(job_id)}",
                "/jobs",
                cleaned_reply
            )
            
    return cleaned_reply


# ── Provider implementations ──────────────────────────────────────────────────

def _call_gemini(
    messages: list[dict],
    system_prompt: str,
    mode: str,
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
        model="gemini-2.0-flash",
        contents=contents,
        config={
            "system_instruction": system_prompt,
            "max_output_tokens": 600,
            "temperature": 0.1 if mode == "faq" else 0.7,  # Giảm nhiệt độ cho FAQ
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
    system_prompt: str,
    mode: str,
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
        temperature=0.1 if mode == "faq" else 0.7,  # Giảm nhiệt độ cho FAQ
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
    job_context: str = "",
) -> dict:
    """
    Smart routing + fallback:
      cv_advisor     → Gemini primary,  Groq fallback
      mock_interview → Groq primary,    Gemini fallback
      faq            → Gemini primary,  Groq fallback (Gemini giỏi đọc context dài)

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

    system_prompt = build_system_prompt(mode, job_position, cv_context, job_context)

    # 2. Chọn thứ tự provider theo mode
    # cv_advisor + faq → Gemini primary (tốt với context dài)
    # mock_interview  → Groq primary (nhanh hơn, real-time feel)
    if mode in ("cv_advisor", "faq"):
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
        reply, tokens = call_fn(messages, system_prompt, mode)
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
                reply, tokens = fallback_call(messages, system_prompt, mode)
                model_used = fallback_name
            except Exception as e2:
                logger.exception("Fallback also failed", extra={"event": "llm_both_failed"})
                raise RuntimeError(f"Cả 2 model đều lỗi: {e2}")
        else:
            logger.exception("LLM call failed", extra={
                "event": "llm_call_failed", "user_id": user_id, "error": str(e),
            })
            raise

    # Lọc ảo giác link job trước khi lưu và trả về
    if reply:
        reply = clean_hallucinated_job_links(reply, job_context)

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


def generate_cv_intro_message(cv_text: str) -> str:
    """
    Phân tích nội dung CV bằng Gemini và tạo ra tin nhắn chào mừng cá nhân hóa:
    "Tôi đã thấy CV của bạn -> tôi thấy bạn đang ứng tuyển vị trí *xxx*..."
    """
    if not cv_text or not cv_text.strip():
        return "Chào bạn, tôi đã nhận được CV của bạn. Bạn muốn tôi phỏng vấn thử (mock interview) hay nâng cấp CV (cv advisor)?"

    prompt = f"""
    Bạn là một trợ lý AI chuyên về tư vấn sự nghiệp IT. Đây là một đoạn nội dung từ CV của ứng viên:
    ---
    {cv_text[:3000]}
    ---
    
    Hãy viết một lời chào bằng tiếng Việt ngắn gọn (khoảng 2-3 câu), thân thiện để bắt đầu hội thoại:
    - Xác nhận đã nhận được CV của ứng viên.
    - Tìm và chỉ ra vị trí chuyên môn/công nghệ chính mà ứng viên đang ứng tuyển/hướng tới (Ví dụ: Frontend Developer, Backend Developer, Java Engineer, Data Analyst, Node.js Developer, v.v. - hãy dùng vị trí thực tế trong CV).
    - Mẫu câu cần có dạng: "Tôi đã nhận được CV của bạn và thấy bạn đang hướng tới vị trí *[Tên vị trí]*..."
    - Hỏi xem ứng viên muốn bắt đầu phỏng vấn thử (mock interview) hay tư vấn/nâng cấp CV (cv advisor).
    
    Lưu ý: Không dùng các từ generic như '[Tên vị trí]', hãy điền thông tin thực tế từ CV. Trả lời thật tự nhiên.
    """

    try:
        response = _gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "max_output_tokens": 150,
                "temperature": 0.5,
            },
        )
        reply = response.text.strip() if response.text else ""
        if reply:
            return reply
    except Exception as e:
        logger.warning(f"Failed to generate custom CV intro message: {e}")

    # Fallback message
    return "Chào bạn, tôi đã nhận được CV của bạn. Bạn muốn tôi phỏng vấn thử (mock interview) hay nâng cấp CV (cv advisor)?"