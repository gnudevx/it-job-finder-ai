"""
LLMService — Chỉ dùng Gemini cho mọi mode (cv_advisor, mock_interview, faq).

Trước đây có cơ chế fallback qua lại Groq khi Gemini rate-limit, nhưng vì
Groq free tier (llama-3.1-8b-instant) chỉ có 6000 TPM nên khi fallback với
context dài (CV, lịch sử chat, job_context...) rất dễ dính lỗi 413 "Request
too large" ngay lập tức. Ngoài ra, Groq (model nhỏ hơn nhiều) không tuân thủ
tốt các rule định dạng chặt của mock_interview (dễ lộ "Câu hỏi:", "Dưới đây
là...", các câu meta hướng dẫn...). Do đó bỏ hẳn việc chuyển qua lại giữa 2
model: luôn gọi Gemini, nếu lỗi thì trả về thông báo thân thiện thay vì
raise 502.

Token tracking: MongoDB, reset mỗi ngày, cảnh báo ở 90%.
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
# _groq_client / _call_groq được giữ lại (không xoá) phòng khi cần dùng lại,
# nhưng KHÔNG còn được gọi ở đâu trong chat_completion nữa.
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
   - Trích xuất TẤT CẢ keywords kỹ thuật hiện có trong CV (ngôn ngữ, framework, tools, methodology).
   - So sánh với từ khóa tiêu chuẩn ngành cho vị trí ứng viên đang ứng tuyển.
   - Chỉ ra chính xác 3-5 từ khóa kỹ thuật quan trọng ĐANG THIẾU trong CV.
   - Ví dụ từ khóa thiếu: RESTful API, Database Indexing, System Design, Unit Testing, CI/CD...
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

QUY TẮC PHÁT NGÔN BẮT BUỘC:
- Tuyệt đối KHÔNG in ra các tiêu đề hệ thống, ghi chú hướng dẫn của prompt như "Bước 1 — Câu hỏi kỹ thuật đầu tiên:", "Bước 2 — Nhận xét + Câu tiếp theo:", "⚠️ CRITICAL RULE", "QUY TRÌNH PHỎNG VẤN", hoặc "Nhận xét:" ở ngoài phần định dạng được yêu cầu. Mọi phản hồi chỉ chứa văn bản hội thoại tự nhiên của người phỏng vấn.
- Mỗi lần phản hồi chỉ được hỏi ĐÚNG 1 câu duy nhất. Không hỏi lồng ghép nhiều câu hỏi hoặc nhiều chủ đề trong cùng một lượt.
- TUYỆT ĐỐI KHÔNG mang các nhận xét, đánh giá hoặc lỗi sai của các câu trả lời cũ từ các lượt chat trước vào phần nhận xét của lượt này. Phần nhận xét ở mỗi lượt chỉ tập trung duy nhất và trực tiếp vào câu trả lời mới nhất vừa nhận được của ứng viên. Tránh việc lặp đi lặp lại lỗi cũ (việc tổng hợp toàn bộ điểm mạnh/điểm yếu chỉ được làm duy nhất một lần ở phần TỔNG KẾT cuối cùng).


QUY TRÌNH PHỎNG VẤN VÀ CẤU TRÚC PHẢN HỒI:

Hãy đếm số lượt câu hỏi của Người phỏng vấn trong lịch sử chat để xác định trạng thái hiện tại:

1. NẾU LỊCH SỬ CHAT CHƯA CÓ CÂU HỎI PHỎNG VẤN NÀO:
   - Hãy chọn dự án hoặc công nghệ GẦN NHẤT / NỔI BẬT NHẤT trong CV của ứng viên.
   - Đặt câu hỏi phỏng vấn kỹ thuật đầu tiên liên quan đến quyết định lựa chọn công nghệ (Technical Decision) hoặc kiến trúc trong dự án đó.
   - Ví dụ: "Trong dự án [Tên dự án] bạn có sử dụng [Công nghệ A], tại sao bạn lại lựa chọn [Công nghệ A] thay vì [Công nghệ B]?"
   - Yêu cầu: Đi thẳng vào câu hỏi, không chào hỏi rườm rà, không yêu cầu giới thiệu bản thân hay hỏi câu lý thuyết suông.

2. NẾU ĐANG TRONG QUÁ TRÌNH PHỎNG VẤN (Đã hỏi ít hơn 5 câu):
   - Đọc câu trả lời mới nhất của ứng viên.
   - Phản hồi theo cấu trúc bắt buộc sau đây:
     ```
     **Nhận xét:** [Phân tích chi tiết và sâu sắc về câu trả lời kỹ thuật của ứng viên. Chỉ rõ điểm tốt, điểm chưa chính xác hoặc thiếu sót về mặt kiến trúc/tối ưu/hiệu năng, và gợi ý cách trả lời chuyên nghiệp hơn.]

     [Đặt đúng 1 câu hỏi phỏng vấn tiếp theo]
     ```
   - YÊU CẦU ĐA DẠNG HÓA KIẾN THỨC: Không chỉ xoáy sâu vào một công nghệ (như Spark/Hadoop). Hãy luân phiên hỏi sang các mảng công nghệ khác có trong CV của ứng viên (Ví dụ: Frontend React, Backend Spring Boot/Java, Database Indexing/SQL/NoSQL, Docker/CI-CD, hoặc system design). Câu hỏi phải mang tính thực tế, giải quyết bài toán cụ thể chứ không hỏi định nghĩa lý thuyết suông.

3. NẾU ĐÃ HỎI ĐỦ TỪ 5 CÂU HỎI TRỞ LÊN:
   - Đọc câu trả lời cuối cùng của ứng viên, đưa ra nhận xét ngắn cho câu đó.
   - Sau đó tiến hành TỔNG KẾT & KẾT THÚC cuộc phỏng vấn.
   - Cấu trúc tổng kết bắt buộc:
     ```
     **Nhận xét câu trả lời vừa rồi:** [Nhận xét câu trả lời cuối]

     🏆 **TỔNG KẾT PHỎNG VẤN THỬ**
     - **Điểm số đánh giá:** [Chấm điểm từ 0-100 dựa trên toàn bộ quá trình trả lời kỹ thuật]
     - **Điểm mạnh cốt lõi:** [Ghi rõ những điểm ứng viên trả lời tốt, thể hiện sự hiểu biết sâu sắc]
     - **Điểm yếu & Lỗ hổng kiến thức cần bù đắp:** [Liệt kê các điểm trả lời chưa tốt, thiếu sót kỹ thuật hoặc câu trả lời né tránh/không biết]
     - **Lời khuyên từ Tech Lead:** [Gợi ý lộ trình ôn tập, tài liệu học thêm hoặc cách cải thiện kỹ năng giao tiếp/phỏng vấn]
     ```
   - Tuyệt đối KHÔNG đặt thêm bất kỳ câu hỏi phỏng vấn nào nữa sau khi đã tổng kết.

NÚT ESCAPE AN TOÀN:
- Nếu CV context trống: YÊU CẦU ứng viên tải lên CV trước rồi mới bắt đầu phỏng vấn.
- Nếu ứng viên trả lời lạc đề, hãy lịch sự nhắc nhở họ tập trung vào câu hỏi kỹ thuật và đặt lại câu hỏi."""


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

    # Job context → đưa vào system prompt
    # QUAN TRỌNG: chỉ ra lệnh "dựa vào danh sách để trả lời" khi mode=faq —
    # ở cv_advisor/mock_interview, job_context chỉ mang tính THAM KHẢO phụ,
    # tránh làm model lạc đề, chạy nhầm sang liệt kê job không liên quan.
    if job_context and mode == "faq":
        prompt += f"""

=== DANH SÁCH VIỆC LÀM ĐANG TUYỂN (dữ liệu thực tế từ hệ thống) ===
{job_context}
=== HẾT DANH SÁCH ===
Hãy dựa vào danh sách trên để trả lời. Nếu không có job phù hợp hãy nói thật."""
    elif job_context:
        # cv_advisor / mock_interview: gợi ý job chỉ là thông tin phụ, không được
        # đưa vào trừ khi người dùng hỏi trực tiếp về cơ hội việc làm, và không được
        # làm thay đổi cấu trúc output đã quy định ở trên.
        prompt += f"""

(Thông tin thêm — KHÔNG PHẢI yêu cầu chính, chỉ dùng khi thật sự liên quan trực tiếp đến câu hỏi của người dùng, tuyệt đối KHÔNG tự ý chèn danh sách job vào cuối câu trả lời phân tích CV hoặc câu hỏi phỏng vấn):
{job_context}"""

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

    # mock_interview dùng temperature thấp hơn để tuân thủ rules tốt hơn
    temperature = 0.1 if mode == "faq" else (0.5 if mode == "mock_interview" else 0.7)
    # Tăng max_tokens cho mock_interview để LLM đọc đủ CV và trả lời sâu
    max_tokens = 400 if mode == "faq" else (1200 if mode == "mock_interview" else 800)

    response = _gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config={
            "system_instruction": system_prompt,
            "max_output_tokens": max_tokens,
            "temperature": temperature,
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
    """NOTE: không còn được gọi trong chat_completion() — giữ lại phòng khi cần."""

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


# ── Router (chỉ Gemini) ────────────────────────────────────────────────────────

def chat_completion(
    messages: list[dict],
    user_id: str,
    mode: str = "cv_advisor",
    job_position: str | None = None,
    cv_context: str = "",
    job_context: str = "",
) -> dict:
    """
    Chỉ dùng Gemini cho mọi mode (cv_advisor, mock_interview, faq).
    Không còn fallback qua Groq — xem giải thích ở docstring đầu file.

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

    # 2. Gọi Gemini. Nếu lỗi (rate limit, timeout, v.v...) → trả message
    #    thân thiện cho user thay vì raise 502, vì không còn model nào khác
    #    để fallback sang nữa.
    reply = None
    tokens = 0
    model_used = "gemini"
    start = time.time()

    try:
        reply, tokens = _call_gemini(messages, system_prompt, mode)
        logger.info("LLM call success", extra={
            "event": "llm_primary_ok", "model": "gemini",
            "user_id": user_id, "mode": mode,
        })
    except Exception as e:
        logger.exception("Gemini call failed", extra={
            "event": "llm_call_failed", "user_id": user_id, "mode": mode, "error": str(e),
        })
        reply = "⚠️ Hệ thống AI đang tạm thời quá tải hoặc gặp sự cố. Bạn vui lòng thử lại sau ít phút giúp mình nhé."
        tokens = 0
        model_used = None

    # Lọc ảo giác link job trước khi lưu và trả về
    if reply:
        reply = clean_hallucinated_job_links(reply, job_context)

    elapsed = round((time.time() - start) * 1000)
    logger.info("LLM call done", extra={
        "event": "llm_call_done", "model": model_used,
        "tokens": tokens, "duration_ms": elapsed, "mode": mode,
    })

    # 3. Cộng token vào MongoDB
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