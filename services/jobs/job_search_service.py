"""
services/jobs/job_search_service.py — Tìm kiếm job từ MongoDB trực tiếp.

Kết nối cùng MongoDB cluster với Node.js server (cùng MONGODB_URI trong .env).
Query read-only, chỉ lấy job có:
  - publishStatus = "approved"
  - visibility = "visible"

Không cần HTTP endpoint nội bộ — đơn giản hơn, an toàn hơn (Python code kiểm soát query).
"""

import re
import logging
from datetime import datetime, timezone
from typing import Optional
from pymongo import MongoClient, DESCENDING
from core.config import settings

logger = logging.getLogger(__name__)

# ── Singleton client ──────────────────────────────────────────────────────────
_mongo_client: MongoClient | None = None

# Tên DB của Node.js server (xem .env Node.js hoặc config/db.js)
JOBS_DB_NAME = "it-job-finder"   # Sẽ fallback sang MONGODB_DB nếu không tìm thấy
MAX_JOBS_PER_QUERY = 5


def _get_jobs_collection():
    """Lấy collection 'jobs' từ MongoDB của Node.js server."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)

    # Thử DB của Node.js server trước
    for db_name in ["ITJOBS", JOBS_DB_NAME, settings.MONGODB_DB, "test"]:
        try:
            db = _mongo_client[db_name]
            # Kiểm tra collection 'jobs' có tồn tại không
            if "jobs" in db.list_collection_names():
                logger.info(f"Found jobs collection in DB: {db_name}")
                return db["jobs"]
        except Exception:
            continue

    # Fallback: dùng DB trong config
    return _mongo_client[settings.MONGODB_DB]["jobs"]



# ── Keyword extraction ────────────────────────────────────────────────────────

# Các từ stopword tiếng Việt để loại bỏ khỏi keyword search
STOPWORDS = {
    "có", "không", "tôi", "bạn", "này", "đây", "và", "hoặc",
    "hay", "thì", "là", "của", "trong", "nào", "gì", "với",
    "hiện", "tại", "đang", "còn", "muốn", "tìm", "hỏi",
    "về", "cho", "được", "một", "những", "các", "web", "site",
}

def extract_keywords(message: str) -> list[str]:
    """
    Trích xuất keywords tìm kiếm từ câu hỏi người dùng.
    Ví dụ: "Có job React Developer tại Hà Nội không?" → ["React", "Developer", "Hà Nội"]
    """
    # Loại bỏ dấu câu, giữ chữ cái + khoảng trắng
    cleaned = re.sub(r"[^\w\s]", " ", message)
    words = cleaned.split()

    keywords = []
    for w in words:
        w_lower = w.lower()
        # Bỏ stopword, bỏ từ ngắn hơn 2 ký tự
        if w_lower not in STOPWORDS and len(w) >= 2:
            keywords.append(w)

    logger.info("Extracted keywords", extra={"event": "keywords_extracted", "keywords": keywords})
    return keywords[:10]  # Giới hạn 10 keywords


# ── Query builder ─────────────────────────────────────────────────────────────

def _get_location_name(db, location_id) -> str:
    if not location_id:
        return "Không rõ địa điểm"
    try:
        loc = db["locations"].find_one({"_id": location_id})
        if loc:
            return loc.get("name", "Không rõ địa điểm")
    except Exception:
        pass
    return "Không rõ địa điểm"


def _get_company_name(db, employer_id) -> str:
    if not employer_id:
        return "IT Job Finder Partner"
    try:
        employer = db["employer"].find_one({"_id": employer_id})
        if employer and "companyId" in employer:
            company = db["COMPANY"].find_one({"_id": employer["companyId"]})
            if company:
                return company.get("name", "IT Job Finder Partner")
    except Exception:
        pass
    return "IT Job Finder Partner"


def _build_search_query(keywords: list[str], location_ids: list = []) -> dict:
    """
    Xây dựng MongoDB query an toàn (không nhận raw query từ user).
    Chỉ tìm trong các fields: title, mustHaveSkills, optionalSkills, specialization, experience, level.
    Luôn filter: publishStatus="approved" AND visibility="visible".
    """
    if not keywords:
        # Không có keyword → lấy jobs mới nhất
        return {
            "publishStatus": "approved",
            "visibility": "visible",
        }

    # Tạo regex pattern từ keywords (case-insensitive)
    keyword_patterns = [{"$regex": kw, "$options": "i"} for kw in keywords]

    # Mỗi keyword OR với nhau trên nhiều fields
    or_conditions = []
    for pattern in keyword_patterns:
        or_conditions.extend([
            {"title": pattern},
            {"mustHaveSkills": pattern},
            {"optionalSkills": pattern},
            {"specialization": pattern},
            {"experience": pattern},
            {"level": pattern},
        ])

    if location_ids:
        or_conditions.append({"location": {"$in": location_ids}})

    return {
        "publishStatus": "approved",
        "visibility": "visible",
        "$or": or_conditions,
    }


# ── Formatter ─────────────────────────────────────────────────────────────────

def _format_salary(job: dict) -> str:
    """Format salary để LLM đọc được."""
    salary_raw = job.get("salary_raw", "")
    if salary_raw:
        return salary_raw
    salary_from = job.get("salaryFrom", "")
    salary_to = job.get("salaryTo", "")
    currency = job.get("currency_unit", "VND")
    if salary_from and salary_to:
        return f"{salary_from} - {salary_to} {currency}"
    return "Thỏa thuận"


def _format_deadline(job: dict) -> str:
    """Format deadline."""
    deadline = job.get("applicationDeadline")
    if not deadline:
        return "Không rõ"
    if isinstance(deadline, datetime):
        return deadline.strftime("%d/%m/%Y")
    return str(deadline)


def _safe_job_fields(job: dict, db) -> dict:
    """
    Chỉ lấy fields công khai — không expose internal fields nhạy cảm.
    """
    return {
        "id": str(job.get("_id", "")),
        "title": job.get("title", "Không rõ"),
        "company": _get_company_name(db, job.get("employer_id")),
        "province": _get_location_name(db, job.get("location")),
        "salary": _format_salary(job),
        "experience": job.get("experience", "Không yêu cầu cụ thể"),
        "level": job.get("level", ""),
        "specialization": job.get("specialization", ""),
        "must_have_skills": job.get("mustHaveSkills", [])[:5],  # Max 5 skills
        "deadline": _format_deadline(job),
        "job_type": job.get("jobType", ""),
    }


def format_jobs_for_llm(jobs: list[dict], is_suggestion: bool = False) -> str:
    """
    Format list jobs thành context text để đưa vào system prompt cho LLM.
    Mỗi job được trình bày rõ ràng, dễ đọc.
    """
    if not jobs:
        return ""

    if is_suggestion:
        lines = [f"Không tìm thấy job khớp chính xác với yêu cầu của bạn. Dưới đây là {len(jobs)} vị trí nổi bật hiện có trên hệ thống để bạn tham khảo:\n"]
    else:
        lines = [f"Tìm thấy {len(jobs)} vị trí phù hợp:\n"]
    for i, job in enumerate(jobs, 1):
        skills_str = ", ".join(job["must_have_skills"]) if job["must_have_skills"] else "Không ghi cụ thể"
        lines.append(
            f"[Job {i}]\n"
            f"  Vị trí: {job['title']}\n"
            f"  Địa điểm: {job['province']}\n"
            f"  Lương: {job['salary']}\n"
            f"  Kinh nghiệm: {job['experience']}\n"
            f"  Cấp bậc: {job['level'] or 'Không ghi'}\n"
            f"  Kỹ năng cần: {skills_str}\n"
            f"  Loại công việc: {job['job_type'] or 'Không ghi'}\n"
            f"  Hạn nộp: {job['deadline']}\n"
            f"  Link: /jobs/{job['id']}\n"
        )

    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def search_jobs_from_message(message: str) -> list[dict]:
    """
    Entry point: nhận câu hỏi của user → trả về list job đã filter.

    Returns:
        list[dict] — danh sách job với fields public (đã qua _safe_job_fields)
    """
    try:
        keywords = extract_keywords(message)
        collection = _get_jobs_collection()
        db = collection.database

        # Tìm location matches dựa trên keywords
        location_ids = []
        if keywords:
            loc_queries = [{"name": {"$regex": kw, "$options": "i"}} for kw in keywords]
            try:
                matched_locs = list(db["locations"].find({"$or": loc_queries}, {"_id": 1}))
                location_ids = [loc["_id"] for loc in matched_locs]
            except Exception as e:
                logger.warning(f"Error fetching location matching keywords: {e}")

        query = _build_search_query(keywords, location_ids)

        raw_jobs = list(
            collection.find(query)
            .sort("createdAt", DESCENDING)
            .limit(MAX_JOBS_PER_QUERY)
        )

        safe_jobs = [_safe_job_fields(j, db) for j in raw_jobs]

        logger.info(
            "Job search completed",
            extra={
                "event": "job_search_done",
                "keywords": keywords,
                "found": len(safe_jobs),
            },
        )
        return safe_jobs

    except Exception as e:
        logger.exception(f"Job search failed: {e}")
        return []



def get_active_job_suggestions(db) -> list[dict]:
    """
    Lấy danh sách các công việc hoạt động (approved + visible) mới nhất để gợi ý.
    """
    try:
        collection = db["jobs"]
        query = {
            "publishStatus": "approved",
            "visibility": "visible",
        }
        raw_jobs = list(
            collection.find(query)
            .sort("createdAt", DESCENDING)
            .limit(MAX_JOBS_PER_QUERY)
        )
        safe_jobs = [_safe_job_fields(j, db) for j in raw_jobs]
        return safe_jobs
    except Exception as e:
        logger.exception(f"Failed to get active job suggestions: {e}")
        return []



def build_faq_messages(session_id: str, user_message: str, job_context: str) -> list[dict]:
    """
    Tạo messages list cho FAQ mode — không dùng RAG CV, dùng job context.
    Lịch sử chat vẫn được lấy từ MongoDB để maintain conversation.
    """
    from services.CV.rag.rag_service import get_history

    history = get_history(session_id, limit=6)
    messages = history.copy()

    if job_context:
        # Inject job data như một context message
        messages.append({
            "role": "user",
            "content": f"[DỮ LIỆU JOB TỪ HỆ THỐNG]\n{job_context}\n[HẾT DỮ LIỆU JOB]\n\nCâu hỏi của tôi: {user_message}",
        })
    else:
        messages.append({"role": "user", "content": user_message})

    return messages
