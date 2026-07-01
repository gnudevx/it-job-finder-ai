
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
from datetime import datetime



"""_summary_ 
    file này định nghĩa các schema (dữ liệu đầu vào/ra, ChromaDB : 
    vector memory, MongoDB: conversation history) dùng trong API, 
    giúp validate dữ liệu đầu vào/ra và tự động tạo docs
"""
# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: Optional[str] = Field(
        default=None,
        description="ID phiên chat. Nếu trống, hệ thống sẽ dùng session mặc định cho user",
    )
    mode: Literal["cv_advisor", "mock_interview", "faq"] = "cv_advisor"
    cv_id: Optional[str] = Field(
        default=None,
        description="ID của CV đã upload. Nếu None → tự dùng CV mới nhất",
    )
    job_position: Optional[str] = Field(
        default=None,
        description="Vị trí phỏng vấn, chỉ cần khi mode=mock_interview. VD: 'Backend Developer'",
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tokens_used: int
    tokens_remaining: int
    warning: Optional[str] = None
    detected_intent: Optional[str] = None  # intent thực tế sau khi detect (để FE biết có context switch không)
    jobs: Optional[List[dict]] = None  # Optional list of matched jobs (safe fields)


# ── CV Upload ─────────────────────────────────────────────────────────────────

class CVUploadResponse(BaseModel):
    cv_id: str
    filename: str
    status: Literal["processing", "done", "failed"]
    message: str


class CVStatusResponse(BaseModel):
    cv_id: str
    status: Literal["uploaded", "processing", "done", "failed"]
    chunks_count: Optional[int] = None
    uploaded_at: Optional[datetime] = None


# ── Token quota ───────────────────────────────────────────────────────────────

class TokenQuota(BaseModel):
    user_id: str
    used: int
    limit: int
    remaining: int
    warning: bool