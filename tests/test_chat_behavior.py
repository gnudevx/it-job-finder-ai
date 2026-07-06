import pytest

from services.AI.llm_service import build_system_prompt
from services.CV.rag import rag_service
from services.CV.rag.rag_service import resolve_session_id, resolve_cv_id


def test_resolve_session_id_falls_back_to_user_default():
    assert resolve_session_id(None, "user-123") == "user:user-123:default"
    assert resolve_session_id("   ", "user-123") == "user:user-123:default"
    assert resolve_session_id("null", "user-123") == "user:user-123:default"
    assert resolve_session_id("undefined", "user-123") == "user:user-123:default"
    assert resolve_session_id("default", "user-123") == "user:user-123:default"
    assert resolve_session_id("session-abc", "user-123") == "user:user-123:session-abc"


def test_mock_interview_prompt_uses_cv_and_does_not_start_with_self_intro():
    prompt = build_system_prompt(
        "mock_interview",
        job_position="Backend Engineer",
        cv_context="Kinh nghiệm 3 năm Python",
    )

    lowered = prompt.lower()
    assert "giới thiệu về bản thân" not in lowered
    assert "câu 1" in lowered or "câu hỏi đầu tiên" in lowered
    assert "dựa vào cv" in lowered or "dựa vào cv thực tế" in lowered or "dựa trên cv" in lowered


def test_resolve_cv_id_prefers_active_cv(monkeypatch):
    monkeypatch.setattr(rag_service, "get_active_cv_id", lambda user_id: "new-cv")

    assert resolve_cv_id("old-cv", "user-1") == "new-cv"
    assert resolve_cv_id(None, "user-1") == "new-cv"
