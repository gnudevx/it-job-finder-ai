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
from typing import Any
from difflib import SequenceMatcher
import time

logger = logging.getLogger(__name__)

# ── Singleton client ──────────────────────────────────────────────────────────
_mongo_client: MongoClient | None = None

# Cache for distinct job titles (refresh every 30 minutes)
_job_titles_cache = {
    "titles": [],
    "timestamp": 0,
    "ttl": 1800  # 30 minutes
}

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


def _get_all_job_titles_cached() -> list[str]:
    """
    Get all distinct job titles from DB with caching.
    Refreshes cache every 30 minutes.
    """
    global _job_titles_cache
    
    now = time.time()
    # Check if cache is still valid
    if (_job_titles_cache["titles"] and 
        (now - _job_titles_cache["timestamp"]) < _job_titles_cache["ttl"]):
        return _job_titles_cache["titles"]
    
    try:
        collection = _get_jobs_collection()
        titles = collection.distinct("title", {
            "publishStatus": "approved",
            "visibility": "visible"
        })
        
        # Update cache
        _job_titles_cache["titles"] = titles or []
        _job_titles_cache["timestamp"] = now
        
        logger.info(f"Updated job titles cache: {len(titles)} titles")
        return _job_titles_cache["titles"]
    except Exception as e:
        logger.exception(f"Failed to get job titles from DB: {e}")
        # Return cached titles even if update fails
        return _job_titles_cache.get("titles", [])


def detect_best_matching_job_title(message: str, min_similarity: float = 0.4) -> Optional[str]:
    """
    Detect best matching job title from DB by comparing message keywords with all titles.
    Uses fuzzy string matching to handle typos, variations.
    
    Strategy:
    1. If message contains keyword that appears in multiple titles, prefer those titles
    2. Score each title by number of keyword matches + similarity
    3. Return title with highest score
    
    Args:
        message: User message
        min_similarity: Minimum similarity ratio (0-1) to consider a match
        
    Returns:
        Best matching job title or None if no good match found
    """
    if not message or not message.strip():
        return None
    
    all_titles = _get_all_job_titles_cached()
    if not all_titles:
        return None
    
    msg_lower = message.lower().strip()
    keywords = extract_keywords(message)
    
    if not keywords:
        return None
    
    best_match = None
    best_score = min_similarity
    
    logger.info(f"Detecting job title for message: '{message}', keywords: {keywords}")
    
    # Score each title based on keyword matches + similarity
    for title in all_titles:
        title_lower = title.lower()
        keyword_match_count = 0
        max_similarity = 0
        
        # Count how many keywords match this title
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Check if keyword is substring of title (exact match = high score)
            if keyword_lower in title_lower:
                keyword_match_count += 1
                # Substring match is very good
                similarity = 1.0
            else:
                # Fuzzy match
                similarity = SequenceMatcher(None, keyword_lower, title_lower).ratio()
            
            max_similarity = max(max_similarity, similarity)
        
        # Composite score: prefer titles with more keyword matches, then by similarity
        composite_score = (
            (keyword_match_count / len(keywords))
            * 0.7
            + max_similarity * 0.3
        )
        
        if composite_score > best_score:
            best_score = composite_score
            best_match = title
            logger.debug(f"Better match: '{title}' (keywords_matched: {keyword_match_count}, similarity: {max_similarity:.2f}, score: {composite_score:.2f})")
    
    if best_match:
        logger.info(f"Best matching job title: '{best_match}' (score: {best_score:.2f})")
    else:
        logger.info(f"No job title matched for message: '{message}'")
    
    return best_match



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

JOB_ALIASES = {
    "software engineering":[
        "software engineer",
        "software developer",
        "software engineering",
        "swe"
    ],

    "frontend":[
        "frontend",
        "front end",
        "frontend developer"
    ],

    "backend":[
        "backend",
        "back end",
        "backend developer"
    ],

    "fullstack":[
        "fullstack",
        "full stack",
        "fullstack developer"
    ],

    "it support":[
        "it support",
        "it helpdesk",
        "helpdesk",
        "support"
    ]
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
    query: dict[str, Any] = {
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
        query["$or"]=[]

        for kw in job_keywords:

            pattern = {
                "$regex": kw,
                "$options":"i"
            }

            query["$or"].extend([
                {"title":pattern},
                {"mustHaveSkills":pattern},
                {"optionalSkills":pattern},
                {"jobDescription":pattern}
            ])

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
        normalized_terms = normalize_job_terms(
            message
        )

        keywords.extend(normalized_terms)

        keywords = list(set(keywords))

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
        logger.info(
            "Built job search query",
            extra={
                "event": "built_query",
                "keywords": keywords,
                "matched_company_keywords": matched_company_keywords,
                "matched_location_keywords": matched_location_keywords,
                "location_ids": [str(x) for x in location_ids],
                "employer_ids": [str(x) for x in employer_ids],
                "query": query,
            },
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

        logger.info(
            "Count jobs query",
            extra={
                "event": "count_query",
                "keywords": keywords,
                "matched_company_keywords": matched_company_keywords,
                "matched_location_keywords": matched_location_keywords,
                "location_ids": [str(x) for x in location_ids],
                "employer_ids": [str(x) for x in employer_ids],
                "query": query,
            },
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

def normalize_job_terms(message: str) -> list[str]:
    """
    Normalize job terms from message using dynamic title detection.
    Returns list of related job titles from DB.
    
    This replaces the old static JOB_ALIASES with dynamic detection.
    """
    if not message or not message.strip():
        return []
    
    # Get the best matching title
    matched_title = detect_best_matching_job_title(message, min_similarity=0.4)
    
    if not matched_title:
        return []
    
    # Extract keywords to find all related titles
    keywords = extract_keywords(message)
    significant_keywords = [kw for kw in keywords if len(kw) >= 3]
    
    if not significant_keywords:
        return [matched_title]
    
    # Find all titles matching the keywords
    try:
        collection = _get_jobs_collection()
        keyword_patterns = []
        for kw in significant_keywords:
            keyword_patterns.append({"title": {"$regex": re.escape(kw), "$options": "i"}})
        
        query = {
            "publishStatus": "approved",
            "visibility": "visible",
            "$or": keyword_patterns
        }
        
        related_titles = collection.distinct("title", query)
        # Limit to top 10 related titles
        return related_titles[:10] if related_titles else [matched_title]
    except Exception as e:
        logger.exception(f"Error in normalize_job_terms: {e}")
        return [matched_title]


# Helper: count title matches for a detected job term in the message

def count_title_matches_by_message(message: str) -> dict:
    try:
        collection = _get_jobs_collection()

        if not message.strip():
            return {
                "term": None,
                "count": 0,
                "distinct_titles_len": 0
            }

        # Extract significant keywords (length >= 3)
        keywords = extract_keywords(message)
        significant_keywords = [kw for kw in keywords if len(kw) >= 3]

        if not significant_keywords:
            return {
                "term": None,
                "count": 0,
                "distinct_titles_len": 0
            }

        # Build  query for title containing each keyword (case-insensitive)
        keyword_patterns = []
        for kw in significant_keywords:
            keyword_patterns.append({"title": {"$regex": re.escape(kw), "$options": "i"}})

        title_query = {
            "publishStatus": "approved",
            "visibility": "visible",
            "$and": keyword_patterns
        }

        count = collection.count_documents(title_query)

        distinct_titles = collection.distinct("title", title_query)

        logger.info(
            "Title count (AND keywords)",
            extra={
                "keywords": significant_keywords,
                "count": count,
                "distinct_titles": len(distinct_titles)
            }
        )

        # For term, we can join significant_keywords with space for display
        term = " ".join(significant_keywords)

        return {
            "term": term,
            "count": count,
            "distinct_titles_len": len(distinct_titles)
        }

    except Exception as e:
        logger.exception(e)
        return {
            "term": None,
            "count": 0,
            "distinct_titles_len": 0
        }


def count_title_noexp_matches_by_message(message: str) -> int:
    """
    Count jobs whose titles match significant keywords (length >= 3) AND require no experience.
    Uses dynamic title detection to find all related titles.
    Returns integer count.
    """
    try:
        collection = _get_jobs_collection()
        if not message or not message.strip():
            return 0

        # Extract keywords (only significant ones)
        keywords = extract_keywords(message)
        significant_keywords = [kw for kw in keywords if len(kw) >= 3]
        
        if not significant_keywords:
            return 0

        # Build OR query for titles containing significant keywords
        keyword_patterns = []
        for kw in significant_keywords:
            keyword_patterns.append({"title": {"$regex": re.escape(kw), "$options": "i"}})

        # Experience patterns indicating no experience required
        exp_pattern = "không|không yêu cầu|no experience|no|none|0|fresher|intern|mới"
        
        # Use $and with $or for title matching
        query = {
            "$and": [
                {"$or": keyword_patterns},
                {
                    "$or": [
                        {"experience": {"$regex": exp_pattern, "$options": "i"}},
                        {"experience": ""},
                        {"experience": {"$exists": False}}
                    ]
                }
            ],
            "publishStatus": "approved",
            "visibility": "visible"
        }
        
        count = int(collection.count_documents(query))
        logger.info(f"No-experience title matches: {count}")
        return count
    except Exception as e:
        logger.exception(f"Count title no-exp matches failed: {e}")
        return 0


def get_title_matched_jobs(message: str, limit: int = 5) -> list[dict]:
    """
    Get jobs whose title matches the best-matching title detected from message.
    Uses dynamic title detection and exact regex match (not semantic search).
    Returns list of safe job fields, sorted by recency.
    """
    try:
        collection = _get_jobs_collection()
        db = collection.database

        if not message or not message.strip():
            return []

        # Use dynamic title detection
        matched_title = detect_best_matching_job_title(message, min_similarity=0.5)
        
        if not matched_title:
            return []

        query = {
            "title": {"$regex": re.escape(matched_title), "$options": "i"},
            "publishStatus": "approved",
            "visibility": "visible"
        }

        raw_jobs = list(
            collection.find(query)
            .sort("createdAt", -1)  # DESCENDING
            .limit(limit)
        )

        safe_jobs = [_safe_job_fields(j, db) for j in raw_jobs]
        logger.info(
            "Title-matched jobs found",
            extra={
                "event": "title_matched_search",
                "matched_title": matched_title,
                "count": len(safe_jobs),
            },
        )
        return safe_jobs
    except Exception as e:
        logger.exception(f"Title-matched job search failed: {e}")
        return []

