from fastapi import APIRouter
from datetime import datetime, timezone

router = APIRouter()
## ── Health Check Endpoint ───────────────────────────────────────────────
# Dùng để Docker healthcheck, monitoring, và test nhanh service có chạy không.

@router.get("/")
async def health_check():
    """Endpoint để Docker healthcheck và monitoring ping."""
    return {
        "status": "ok",
        "service": "cv-chatbot-ai",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }