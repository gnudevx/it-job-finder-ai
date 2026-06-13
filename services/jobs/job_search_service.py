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
JOBS_DB_NAME = "it-job-finder"   # Sẽ fallback sang JOBS_MONGO_DB nếu không tìm thấy
MAX_JOBS_PER_QUERY = 5


def _get_jobs_collection():
    """Lấy collection 'jobs' từ MongoDB của Node.js server."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(settings.JOBS_MONGO_URI, serverSelectionTimeoutMS=5000)

    # Thử DB của Node.js server trước
    for db_name in [settings.JOBS_MONGO_DB, "ITJOBS", JOBS_DB_NAME, settings.MONGODB_DB, "test"]:
        try:
            db = _mongo_client[db_name]
            # Kiểm tra collection 'jobs' có tồn tại không
            if "jobs" in db.list_collection_names():
                logger.info(f"Found jobs collection in DB: {db_name}")
                return db["jobs"]
        except Exception:
            continue

    # Fallback: dùng DB trong config
    return _mongo_client[settings.JOBS_MONGO_DB]["jobs"]



# ── Keyword extraction ────────────────────────────────────────────────────────

# Các từ stopword tiếng Việt để loại bỏ khỏi keyword search
STOPWORDS = {
    "có", "không", "tôi", "bạn", "này", "đây", "và", "hoặc",
    "hay", "thì", "là", "của", "trong", "nào", "gì", "với",
    "hiện", "tại", "đang", "còn", "muốn", "tìm", "hỏi",
    "về", "cho", "được", "một", "những", "các", "web", "site",
    "công", "ty", "tnhh", "cp", "cổ", "phần", "trên", "hệ",
    "thống", "website", "ở", "bao", "nhiêu", "ai", "đâu",
    "job", "jobs", "việc", "làm", "tuyển", "dụng", "nào", "gì",
    "vị", "trí"
}

# Các từ khóa thường là tiêu đề công việc, kỹ năng, không dùng để map sang tên công ty
JOB_TITLE_KEYWORDS = {
    "giám", "đốc", "kinh", "doanh", "nhân", "viên", "trưởng", "phòng",
    "kỹ", "sư", "lập", "trình", "thiết", "kế", "quản", "trị", "phân", "tích",
    "developer", "engineer", "designer", "manager", "tester", "intern", "fresher",
    "junior", "senior", "lead", "architect", "sales", "marketing", "hr", "admin"
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
        from bson import ObjectId
        query_id = ObjectId(location_id) if isinstance(location_id, str) and ObjectId.is_valid(location_id) else location_id
        loc = db["locations"].find_one({"_id": query_id})
        if loc:
            return loc.get("name", "Không rõ địa điểm")
    except Exception:
        pass
    return "Không rõ địa điểm"


def _get_company_name(db, employer_id) -> str:
    if not employer_id:
        return "IT Job Finder Partner"
    try:
        from bson import ObjectId
        query_emp_id = ObjectId(employer_id) if isinstance(employer_id, str) and ObjectId.is_valid(employer_id) else employer_id
        employer = db["employer"].find_one({"_id": query_emp_id})
        if employer and "companyId" in employer:
            company_id = employer["companyId"]
            query_comp_id = ObjectId(company_id) if isinstance(company_id, str) and ObjectId.is_valid(company_id) else company_id
            company = db["COMPANY"].find_one({"_id": query_comp_id})
            if company:
                return company.get("name", "IT Job Finder Partner")
    except Exception:
        pass
    return "IT Job Finder Partner"


def _build_search_query(keywords: list[str], location_ids: list = [], employer_ids: list = [], matched_company_keywords: list = [], matched_location_keywords: list = []) -> dict:
    """
    Xây dựng MongoDB query an toàn (không nhận raw query từ user).
    Chỉ tìm trong các fields: title, mustHaveSkills, optionalSkills, specialization, experience, level.
    Luôn filter: publishStatus="approved" AND visibility="visible".
    """
    query = {
        "publishStatus": "approved",
        "visibility": "visible",
    }

    # Lọc các keywords không phải tên công ty hoặc địa điểm để tìm kiếm theo title/skills
    job_keywords = [kw for kw in keywords if kw not in matched_company_keywords and kw not in matched_location_keywords]

    # 1. Điều kiện về công ty (nếu có)
    if employer_ids:
        query["employer_id"] = {"$in": employer_ids}

    # 2. Điều kiện về địa điểm (nếu có)
    if location_ids:
        query["location"] = {"$in": location_ids}

    # 3. Điều kiện về kỹ năng / title (nếu có job_keywords)
    if job_keywords:
        and_conditions = []
        for kw in job_keywords:
            pattern = {"$regex": kw, "$options": "i"}
            and_conditions.append({
                "$or": [
                    {"title": pattern},
                    {"mustHaveSkills": pattern},
                    {"optionalSkills": pattern},
                    {"specialization": pattern},
                    {"experience": pattern},
                    {"level": pattern},
                ]
            })
        query["$and"] = and_conditions

    return query


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


def format_jobs_for_llm(jobs: list[dict], is_suggestion: bool = False, total_count: int = 0, matched_count: int = 0) -> str:
    """
    Format list jobs thành context text để đưa vào system prompt cho LLM.
    Đồng thời hiển thị thống kê tổng số jobs và số jobs khớp.
    """
    lines = []
    lines.append("Thống kê hệ thống:")
    lines.append(f"  - Tổng số công việc đang hoạt động trên hệ thống: {total_count}")
    
    if not is_suggestion:
        lines.append(f"  - Số công việc khớp với yêu cầu tìm kiếm: {matched_count}")
        
    if not jobs:
        if is_suggestion:
            lines.append("\nHiện tại không có công việc nào trên hệ thống.")
        else:
            lines.append("\nKhông tìm thấy công việc nào khớp với từ khóa tìm kiếm của bạn.")
        return "\n".join(lines)

    if is_suggestion:
        lines.append(f"\nDưới đây là {len(jobs)} công việc mới nhất hiện có để bạn tham khảo:\n")
    else:
        lines.append(f"\nDưới đây là {len(jobs)} công việc phù hợp nhất trong số các công việc tìm thấy:\n")
        
    for i, job in enumerate(jobs, 1):
        skills_str = ", ".join(job["must_have_skills"]) if job["must_have_skills"] else "Không ghi cụ thể"
        lines.append(
            f"[Job {i}]\n"
            f"  Vị trí: {job['title']}\n"
            f"  Công ty: {job['company']}\n"
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

        # 1. Xác định địa điểm phù hợp từ message (sử dụng $expr index query để tìm tên địa điểm là substring của message)
        location_ids = []
        matched_location_keywords = []
        try:
            locs = list(db["locations"].find({
                "$expr": {
                    "$gte": [
                        { "$indexOfCP": [ message.lower(), { "$toLower": "$name" } ] },
                        0
                    ]
                }
            }, {"_id": 1, "name": 1}))
            if locs:
                location_ids = [loc["_id"] for loc in locs]
                # Xác định keyword nào thuộc địa điểm đã khớp
                for l in locs:
                    for kw in keywords:
                        if kw.lower() in l["name"].lower() and kw not in matched_location_keywords:
                            matched_location_keywords.append(kw)
        except Exception as e:
            logger.warning(f"Error fetching locations: {e}")

        # 2. Xác định công ty phù hợp từ keywords (bỏ qua các từ khóa là job title)
        employer_ids = []
        matched_company_keywords = []
        if keywords:
            for kw in keywords:
                if kw.lower() in JOB_TITLE_KEYWORDS:
                    continue
                companies = list(db["COMPANY"].find({"name": {"$regex": kw, "$options": "i"}}, {"_id": 1}))
                if companies:
                    matched_company_keywords.append(kw)
                    company_ids = [c["_id"] for c in companies]
                    company_ids_all = company_ids + [str(cid) for cid in company_ids]
                    employers = list(db["employer"].find({"companyId": {"$in": company_ids_all}}, {"_id": 1}))
                    if employers:
                        for emp in employers:
                            employer_ids.append(emp["_id"])
                            employer_ids.append(str(emp["_id"]))

        # 3. Tạo query
        query = _build_search_query(
            keywords=keywords,
            location_ids=location_ids,
            employer_ids=employer_ids,
            matched_company_keywords=matched_company_keywords,
            matched_location_keywords=matched_location_keywords
        )

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


def count_jobs_from_message(message: str) -> int:
    """
    Đếm số lượng job phù hợp với tin nhắn của user.
    """
    try:
        collection = _get_jobs_collection()
        db = collection.database

        # Nếu tin nhắn rỗng, đếm tất cả jobs đang hoạt động
        if not message.strip():
            return collection.count_documents({
                "publishStatus": "approved",
                "visibility": "visible"
            })

        keywords = extract_keywords(message)

        # 1. Xác định địa điểm phù hợp từ message
        location_ids = []
        matched_location_keywords = []
        try:
            locs = list(db["locations"].find({
                "$expr": {
                    "$gte": [
                        { "$indexOfCP": [ message.lower(), { "$toLower": "$name" } ] },
                        0
                    ]
                }
            }, {"_id": 1, "name": 1}))
            if locs:
                location_ids = [loc["_id"] for loc in locs]
                for l in locs:
                    for kw in keywords:
                        if kw.lower() in l["name"].lower() and kw not in matched_location_keywords:
                            matched_location_keywords.append(kw)
        except Exception as e:
            logger.warning(f"Error counting locations: {e}")

        # 2. Xác định công ty phù hợp từ keywords
        employer_ids = []
        matched_company_keywords = []
        if keywords:
            for kw in keywords:
                if kw.lower() in JOB_TITLE_KEYWORDS:
                    continue
                companies = list(db["COMPANY"].find({"name": {"$regex": kw, "$options": "i"}}, {"_id": 1}))
                if companies:
                    matched_company_keywords.append(kw)
                    company_ids = [c["_id"] for c in companies]
                    company_ids_all = company_ids + [str(cid) for cid in company_ids]
                    employers = list(db["employer"].find({"companyId": {"$in": company_ids_all}}, {"_id": 1}))
                    if employers:
                        for emp in employers:
                            employer_ids.append(emp["_id"])
                            employer_ids.append(str(emp["_id"]))

        query = _build_search_query(
            keywords=keywords,
            location_ids=location_ids,
            employer_ids=employer_ids,
            matched_company_keywords=matched_company_keywords,
            matched_location_keywords=matched_location_keywords
        )

        return collection.count_documents(query)

    except Exception as e:
        logger.exception(f"Count jobs failed: {e}")
        return 0



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
