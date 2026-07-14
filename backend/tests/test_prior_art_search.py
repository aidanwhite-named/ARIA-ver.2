import asyncio

from backend.models.schemas import Settings
from backend.services.prior_art_search import (
    build_parallel_search_prompts,
    build_synthesis_prompt,
    needs_expansion,
    run_adaptive_prior_art_search,
    search_result_stats,
)


def test_parallel_search_has_three_distinct_axes():
    prompts = build_parallel_search_prompts("claim", [{"label": "A", "text": "feature"}], [])
    assert [item["axis"] for item in prompts] == ["functional", "patent", "scholarly"]
    joined = "\n".join(item["prompt"] for item in prompts)
    assert "gap role" in joined
    assert "구현어와 기능어" in joined
    assert "고정 키워드" in joined
    assert "gap role 커버리지" in joined


def test_expansion_gate_uses_urls_and_identifiers():
    rich = [
        "US20250123456A1 https://patents.google.com/patent/US20250123456A1",
        "arXiv:2411.12345 https://arxiv.org/abs/2411.12345",
        "WO2024123456A1 https://patentscope.wipo.int/example",
    ]
    assert search_result_stats(rich)["url_count"] == 3
    assert not needs_expansion(rich)
    assert needs_expansion(["one result https://example.com"])


def test_synthesis_prompt_requires_card_friendly_candidates_and_visible_urls():
    prompt = build_synthesis_prompt("claim", [{"label": "A", "text": "feature"}], [])

    assert "### 후보 1: 문헌명" in prompt
    assert "**문헌번호(이름)**" in prompt
    assert "전체 URL 문자열을 그대로 표시" in prompt
    assert "논문은 논문의 정식 제목" in prompt
    assert "DOI, 저널 권·호·쪽수 또는 데이터베이스 식별자만 적지 말고" in prompt
    assert "후보 분류용 `###` 또는 `####` 중간 제목은 만들지 마십시오" in prompt


def test_adaptive_search_skips_expansion_when_initial_results_are_sufficient():
    calls = []

    async def fake_runner(prompt, system, settings, agent="compare", web_search=False):
        calls.append(web_search)
        if web_search:
            idx = sum(1 for value in calls if value)
            return (
                f"US2025012345{idx}A1 "
                f"https://patents.google.com/patent/US2025012345{idx}A1"
            )
        return "## 후보 문헌\n정리 결과"

    result = asyncio.run(run_adaptive_prior_art_search(
        "claim",
        [{"label": "A", "text": "feature"}],
        [],
        Settings(),
        runner=fake_runner,
    ))
    assert result["expanded"] is False
    assert calls == [True, True, True, False]
    assert result["result_md"].startswith("## 후보 문헌")


def test_adaptive_search_expands_weak_initial_results_once():
    calls = []

    async def fake_runner(prompt, system, settings, agent="compare", web_search=False):
        calls.append(web_search)
        if not web_search:
            return "## 후보 문헌\n확장 결과 정리"
        if len(calls) <= 3:
            return "관련 후보 없음"
        return "US20250124622A1 https://patents.google.com/patent/US20250124622A1"

    result = asyncio.run(run_adaptive_prior_art_search(
        "claim",
        [{"label": "A", "text": "feature"}],
        [],
        Settings(),
        runner=fake_runner,
    ))
    assert result["expanded"] is True
    assert calls == [True, True, True, True, False]
