import asyncio

from backend.models.schemas import Settings
from backend.services.prior_art_search import (
    build_parallel_search_prompts,
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
