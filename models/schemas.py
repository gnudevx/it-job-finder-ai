from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


# ── Chat ─────────────────────────────────────────────────────────────────────
# file này định nghĩa các schema (dữ liệu đầu vào/ra, ChromaDB : vector memory, MongoDB: conversation history) dùng trong API, 
# giúp validate dữ liệu đầu vào/ra và tự động tạo docs

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str = Field(..., description="UUID của phiên chat")
    mode: Literal["cv_advisor", "mock_interview"] = "cv_advisor"
    job_position: Optional[str] = Field(
        default=None,
        description="Chỉ cần khi mode=mock_interview, vd: 'Backend Developer'",
    )


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tokens_used: int
    tokens_remaining: int
    warning: Optional[str] = None   # Cảnh báo khi gần hết quota


# ── CV Upload ─────────────────────────────────────────────────────────────────

class CVUploadResponse(BaseModel):
    cv_id: str
    filename: str
    status: Literal["processing", "done", "failed"]
    message: str


class CVStatusResponse(BaseModel):
    cv_id: str
    status: Literal["processing", "done", "failed"]
    chunks_count: Optional[int] = None
    uploaded_at: Optional[datetime] = None


# ── Token quota ───────────────────────────────────────────────────────────────

class TokenQuota(BaseModel):
    user_id: str
    used: int
    limit: int
    remaining: int
    warning: bool