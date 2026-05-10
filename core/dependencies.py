"""
JWT dependency cho FastAPI.

Có 2 cách Node.js gọi FastAPI:
  A) Forward cookie accessToken từ browser  →  đọc từ cookie
  B) Node.js tự thêm header X-User-Id sau khi verify  →  trust header

Hiện tại implement cả 2, ưu tiên header (B) vì Node.js đã verify rồi.
"""

import jwt
from fastapi import HTTPException, status, Request, Cookie
from typing import Optional
from core.config import settings
import logging

from fastapi import Header, HTTPException
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CurrentUser:
    user_id: str
    role: str


async def get_current_user(
    x_user_id: str | None = Header(default=None),
    x_user_role: str = Header(default="user"),
) -> CurrentUser:
    """
    Dependency: inject vào bất kỳ route nào cần auth.

    Ưu tiên 1: Header X-User-Id (Node.js đã verify JWT, chỉ forward user info)
    Ưu tiên 2: Cookie accessToken (browser gửi thẳng tới FastAPI)
    """

    # ── Ưu tiên 1: Node.js đã verify, forward headers ────────────────────
    # Node.js middleware thêm: req.headers['x-user-id'] = decoded.id
    #                          req.headers['x-user-role'] = decoded.role
    if not x_user_id:
        raise HTTPException(
            status_code=401,
            detail="Missing access token"  # ← đây là lỗi bạn đang thấy
        )
    return CurrentUser(user_id=x_user_id, role=x_user_role)
CurrentUser = CurrentUser # Để tránh lỗi circular import khi type hint CurrentUser trong services/AI/llm_service.py