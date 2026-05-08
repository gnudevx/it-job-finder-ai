"""
JWT dependency cho FastAPI.

Có 2 cách Node.js gọi FastAPI:
  A) Forward cookie accessToken từ browser  →  đọc từ cookie
  B) Node.js tự thêm header X-User-Id sau khi verify  →  trust header

Hiện tại implement cả 2, ưu tiên header (B) vì Node.js đã verify rồi.
"""

import jwt
from fastapi import Depends, HTTPException, status, Request, Cookie
from typing import Optional
from core.config import settings
import logging

logger = logging.getLogger(__name__)


class CurrentUser:
    def __init__(self, user_id: str, role: str):
        self.user_id = user_id
        self.role = role

    def __repr__(self):
        return f"<User {self.user_id} role={self.role}>"


def get_current_user(
    request: Request,
    # Nếu Node.js forward cookie thẳng
    accessToken: Optional[str] = Cookie(default=None),
) -> CurrentUser:
    """
    Dependency: inject vào bất kỳ route nào cần auth.

    Ưu tiên 1: Header X-User-Id (Node.js đã verify JWT, chỉ forward user info)
    Ưu tiên 2: Cookie accessToken (browser gửi thẳng tới FastAPI)
    """

    # ── Ưu tiên 1: Node.js đã verify, forward headers ────────────────────
    # Node.js middleware thêm: req.headers['x-user-id'] = decoded.id
    #                          req.headers['x-user-role'] = decoded.role
    x_user_id = request.headers.get("X-User-Id")
    x_user_role = request.headers.get("X-User-Role", "user")

    if x_user_id:
        return CurrentUser(user_id=x_user_id, role=x_user_role)

    # ── Ưu tiên 2: Tự verify cookie (dùng khi test trực tiếp) ────────────
    token = accessToken or request.headers.get("Authorization", "").removeprefix("Bearer ")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is missing",
        )

    try:
        payload = jwt.decode(
            token,
            settings.ACCESS_TOKEN_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return CurrentUser(
            user_id=str(payload.get("id") or payload.get("sub")),
            role=payload.get("role", "user"),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")