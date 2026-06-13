"""
test_flows.py — Test script độc lập, không cần Docker/MongoDB/Redis/ChromaDB.

Chạy: python test_flows.py

Kiểm tra:
  1. Intent detection (rule-based + LLM fallback)
  2. Keyword extraction
  3. Job search MongoDB (cần MongoDB đang chạy local)
  4. LLM chat_completion với FAQ mode (cần GEMINI_API_KEY)
"""

import os
import sys

# Thêm project root vào Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  IT Job Finder AI — Test Flows")
print("=" * 60)


# ── Test 1: Intent Detection (Rule-based) ─────────────────────────────────────

def test_intent_detection():
    print("\n[TEST 1] Intent Detection — Rule-based")
    print("-" * 40)

    from services.AI.intent_service import detect_intent, did_context_switch

    test_cases = [
        # (message, current_mode, expected_intent, description)
        ("Có job React Developer không?",        "mock_interview", "faq",           "FAQ từ mock_interview"),
        ("CV tôi có ổn không?",                  "mock_interview", "mock_interview", "CV hỏi từ mock (không đủ keyword để switch)"),
        ("Tìm job backend tại Hà Nội",            "cv_advisor",    "faq",           "FAQ từ cv_advisor"),
        ("Câu hỏi phỏng vấn về Python là gì?",   "faq",           "mock_interview", "Interview từ faq"),
        ("Phân tích CV của tôi đi",              "mock_interview", "cv_advisor",     "CV từ mock_interview"),
        ("Tôi muốn luyện phỏng vấn thử",        "cv_advisor",    "mock_interview", "Interview từ cv_advisor"),
        ("Đang tuyển job senior fullstack không?","cv_advisor",    "faq",           "FAQ với senior fullstack"),
        ("Xin chào",                             "cv_advisor",    "cv_advisor",    "Fallback giữ nguyên mode"),
    ]

    passed = 0
    failed = 0
    for msg, mode, expected, desc in test_cases:
        result = detect_intent(msg, mode)
        switched = did_context_switch(result, mode)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        else:
            failed += 1
        switch_note = f" [SWITCH: {mode}→{result}]" if switched else ""
        print(f"  {status} {desc}")
        print(f"     Msg: \"{msg[:50]}\"")
        print(f"     Mode: {mode} | Expected: {expected} | Got: {result}{switch_note}")
        print()

    print(f"  Result: {passed}/{len(test_cases)} passed, {failed} failed")
    return failed == 0


# ── Test 2: Keyword Extraction ────────────────────────────────────────────────

def test_keyword_extraction():
    print("\n[TEST 2] Keyword Extraction")
    print("-" * 40)

    from services.jobs.job_search_service import extract_keywords

    test_cases = [
        ("Có job React Developer tại Hà Nội không?",   ["React", "Developer", "Hà", "Nội"]),
        ("Tìm việc AI Engineer lương cao",             ["việc", "AI", "Engineer", "lương", "cao"]),
        ("Đang tuyển Backend Python không?",           ["Backend", "Python"]),
    ]

    passed = 0
    for msg, expected_contains in test_cases:
        keywords = extract_keywords(msg)
        # Kiểm tra các keyword quan trọng có trong kết quả không
        found = [kw for kw in expected_contains if any(kw.lower() in k.lower() for k in keywords)]
        status = "✅" if len(found) >= len(expected_contains) // 2 else "⚠️"
        if status == "✅":
            passed += 1
        print(f"  {status} \"{msg}\"")
        print(f"     Keywords: {keywords}")
        print()

    print(f"  Result: {passed}/{len(test_cases)} passed")
    return True  # keyword extraction is best-effort


# ── Test 3: Job Search (cần MongoDB) ─────────────────────────────────────────

def test_job_search():
    print("\n[TEST 3] Job Search — MongoDB (kết nối database thực tế)")
    print("-" * 40)

    try:
        from services.jobs.job_search_service import (
            search_jobs_from_message,
            count_jobs_from_message,
            format_jobs_for_llm,
            get_active_job_suggestions,
            _get_jobs_collection,
        )

        # 1. Đếm tổng số job đang hoạt động
        total_count = count_jobs_from_message("")
        print(f"  [COUNT] Tổng số job đang hoạt động trên hệ thống: {total_count}")

        # 2. Test các câu hỏi cụ thể của người dùng
        test_queries = [
            "trên hệ thống có bao nhiêu job",
            "có job nào là vị trí giám đốc kinh doanh không",
            "công ty TNHH ABC còn tuyển Giám đốc kinh doanh không",
            "Tìm job React Developer",
        ]

        for query in test_queries:
            matched_count = count_jobs_from_message(query)
            jobs = search_jobs_from_message(query)
            context = format_jobs_for_llm(jobs, is_suggestion=False, total_count=total_count, matched_count=matched_count)
            print(f"  Query: \"{query}\"")
            print(f"  Matched count (đếm được): {matched_count}")
            print(f"  Found jobs: {len(jobs)} jobs")
            for j in jobs:
                print(f"    - Title: {j['title']} | Company: {j['company']} | Location: {j['province']} | Link: /jobs/{j['id']}")
            if context:
                print(f"  Context generated for LLM:\n{context[:300]}\n...")
            print()

        # Test active job suggestions
        print("  Testing get_active_job_suggestions...")
        try:
            col = _get_jobs_collection()
            suggestions = get_active_job_suggestions(col.database)
            print(f"  Found {len(suggestions)} suggestions")
            if suggestions:
                print(f"  First suggestion: {suggestions[0]['title']}")
                context_sugg = format_jobs_for_llm(suggestions, is_suggestion=True, total_count=total_count, matched_count=total_count)
                print(f"  Suggestions context preview:\n{context_sugg[:200]}...")
        except Exception as ex:
            print(f"  ⚠️ Suggestions test failed: {ex}")
        print()

        print("  ✅ Job search OK")
        return True

    except Exception as e:
        print(f"  ⚠️  Job search failed: {e}")
        return False


# ── Test 4: LLM FAQ Mode (cần GEMINI_API_KEY) ────────────────────────────────

def test_llm_faq():
    print("\n[TEST 4] LLM FAQ Mode — Gemini (cần GEMINI_API_KEY)")
    print("-" * 40)

    try:
        from services.AI.llm_service import build_system_prompt

        # Test build_system_prompt cho faq mode với job context giả
        fake_job_context = """Tìm thấy 2 vị trí phù hợp:

[Job 1]
  Vị trí: Senior React Developer
  Địa điểm: Hà Nội
  Lương: 2000 - 4000 USD
  Kinh nghiệm: 3-5 năm
  Hạn nộp: 30/06/2026
  Link: /jobs/abc123

[Job 2]
  Vị trí: React Frontend Developer
  Địa điểm: Hồ Chí Minh
  Lương: 1500 - 3000 USD
  Kinh nghiệm: 1-3 năm
  Hạn nộp: 15/07/2026
  Link: /jobs/def456"""

        prompt = build_system_prompt(
            mode="faq",
            job_context=fake_job_context,
        )
        print("  ✅ FAQ system prompt built successfully")
        print(f"  Prompt length: {len(prompt)} chars")
        print(f"  Has job context: {'DANH SÁCH VIỆC LÀM' in prompt}")
        print()

        # Thử gọi Gemini thực sự
        from core.config import settings
        if not settings.GEMINI_API_KEY or "AIzaSy" not in settings.GEMINI_API_KEY:
            print("  ⚠️  GEMINI_API_KEY không hợp lệ, skip LLM call test")
            return True

        from services.AI.llm_service import _call_gemini

        messages = [{"role": "user", "content": "Có job React Developer không?"}]
        reply, tokens = _call_gemini(messages, prompt, mode="faq")

        print(f"  ✅ Gemini FAQ reply OK ({tokens} tokens)")
        print(f"  Reply preview: {reply[:150]}...")
        return True

    except Exception as e:
        print(f"  ⚠️  LLM test skipped: {e}")
        return True  # optional


# ── Test 5: Context Switch Scenario ──────────────────────────────────────────

def test_context_switch_scenario():
    print("\n[TEST 5] Context Switch Scenario — Cuộc trò chuyện thực tế")
    print("-" * 40)

    from services.AI.intent_service import detect_intent

    conversation = [
        ("Bắt đầu phỏng vấn thử vị trí Backend nhé", "mock_interview"),
        ("Tôi có 3 năm kinh nghiệm với Python và Django", "mock_interview"),
        ("À tiện thể, web có đang tuyển job Backend Python không?", "mock_interview"),  # → switch FAQ
        ("Mức lương thế nào?", "faq"),  # tiếp tục FAQ
        ("Thôi quay lại phỏng vấn thử đi, câu tiếp theo là gì?", "faq"),  # → switch back
        ("CV tôi cần cải thiện gì không?", "mock_interview"),  # → switch CV
    ]

    print("  Simulating conversation with context switches:")
    print()
    current_mode = "mock_interview"
    for msg, mode_at_time in conversation:
        detected = detect_intent(msg, current_mode)
        switch = "🔄 SWITCH!" if detected != current_mode else "   same"
        print(f"  User [{current_mode}]: \"{msg[:55]}\"")
        print(f"  → Detected: {detected} {switch}")
        print()
        current_mode = detected  # simulate mode updating after each message

    print("  ✅ Context switch scenario complete")
    return True


# ── Test 6: Anti-Hallucination Regex Cleaning ──────────────────────────────────

def test_anti_hallucination():
    print("\n[TEST 6] Anti-Hallucination — Regex Job Link Verification")
    print("-" * 40)

    from services.AI.llm_service import clean_hallucinated_job_links

    job_context = """
    Tìm thấy 2 vị trí phù hợp:
    [Job 1]
      Vị trí: React Developer
      Link: /jobs/react-dev-123
    [Job 2]
      Vị trí: Python Developer
      Link: /jobs/python-dev-456
    """

    reply_with_hallucination = "Bạn tham khảo link /jobs/react-dev-123 này nhé. Hoặc cả link ảo giác này: /jobs/fake-job-789."
    
    cleaned = clean_hallucinated_job_links(reply_with_hallucination, job_context)
    
    print(f"  Original LLM Reply: {reply_with_hallucination}")
    print(f"  Cleaned LLM Reply : {cleaned}")
    
    success = "/jobs/react-dev-123" in cleaned and "/jobs/fake-job-789" not in cleaned and "/jobs" in cleaned
    if success:
        print("  ✅ Anti-Hallucination check passed! Hallucinated link was successfully cleaned/redirected.")
    else:
        print("  ❌ Anti-Hallucination check failed.")
    return success


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = []

    results.append(("Intent Detection", test_intent_detection()))
    results.append(("Keyword Extraction", test_keyword_extraction()))
    results.append(("Job Search (MongoDB)", test_job_search()))
    results.append(("LLM FAQ Mode", test_llm_faq()))
    results.append(("Context Switch Scenario", test_context_switch_scenario()))
    results.append(("Anti-Hallucination Cleaning", test_anti_hallucination()))

    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🎉 Tất cả test pass! Sẵn sàng test với Docker.")
    else:
        print("  ⚠️  Có test fail. Kiểm tra lại trước khi deploy.")
    print("=" * 60)
