"""주/보조 인용발명 판정(LLM-A / LLM-B)과 자격 게이트 회귀 테스트.

이 파일의 테스트는 LLM을 호출하지 않는다. call_ai는 모두 모킹되며, 골든셋
회귀는 순수 함수만 실행한다.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from backend.models.schemas import (
    ClaimElement,
    ExtractedDocument,
    ParsedClaim,
    Settings,
)
from backend.services import reference_adjudicator as ra
from backend.services.citation_chain import (
    _compute_family_context,
    _eligible_primary_indices,
    _select_family_reference_pair,
    build_citation_chain_from_comparisons,
    shortlist_primary_candidates,
    shortlist_secondary_candidates,
)

GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"


# ---------------------------------------------------------------------------
# 픽스처 헬퍼
# ---------------------------------------------------------------------------

def _claim(labels: tuple[str, ...] = ("A", "B", "C")) -> ParsedClaim:
    texts = {
        "A": "제1 회전축과 제2 회전축이 서로 직교하도록 배치되고 동기 회전하는 구동부",
        "B": "상기 구동부의 회전 위상차를 검출하여 보정하는 제어부",
        "C": "데이터를 저장하는 메모리",
    }
    importance = {"A": "5", "B": "5", "C": "2"}
    return ParsedClaim(
        claim_number=1,
        claim_type="independent",
        text="구동부와 제어부를 포함하는 장치",
        elements=[
            ClaimElement(label=label, text=texts[label], importance=importance[label])
            for label in labels
        ],
    )


def _item(
    label: str,
    judgment: str,
    quote: str,
    *,
    directness: str = "direct",
    missing: list[str] | None = None,
    motivation: str = "",
    risk: str = "none_explicit",
    chunk_id: str = "",
) -> dict:
    chunk = chunk_id or ("[0001]" if quote else "")
    return {
        "label": label,
        "found": bool(quote),
        "quote": quote,
        "chunk_id": chunk,
        "judgment": judgment,
        "판단_이유": "직접 대응 여부를 검토함",
        "directness": directness,
        "missing_limitations": missing or [],
        "evidence": ([{"limitation": label, "quote": quote, "chunk_id": chunk}] if quote else []),
        "motivation_quote": motivation,
        "combination_risk": risk,
        "combination_risk_reason": "",
    }


def _absent(label: str) -> dict:
    return _item(label, "대응 없음", "", directness="absent", missing=[f"{label} 전체"])


def _docs(count: int) -> list[ExtractedDocument]:
    return [ExtractedDocument(filename=f"doc{index}.pdf") for index in range(count)]


def _settings() -> Settings:
    return Settings(engine="claude", model_compare="test-model")


# ---------------------------------------------------------------------------
# 1층 골든셋 회귀 — LLM 호출 없음
# ---------------------------------------------------------------------------

def _golden_cases() -> list[Path]:
    if not GOLDENS_DIR.exists():
        return []
    return sorted(
        path for path in GOLDENS_DIR.iterdir()
        if path.is_dir() and (path / "expected.json").exists()
    )


@pytest.mark.parametrize("case_dir", _golden_cases(), ids=lambda path: path.name)
def test_golden_reference_selection_is_stable(case_dir: Path) -> None:
    """상수 조정·리팩터링이 실제 사건의 주/보조 선정을 바꾸지 않는지 확인한다."""
    expected = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
    claims = [
        ParsedClaim(**item)
        for item in json.loads((case_dir / "claims.json").read_text(encoding="utf-8"))
    ]
    prior_docs = [ExtractedDocument(filename=name) for name in expected["documents"]]

    # 체인 빌드는 job_dir에 citation_chain.json을 쓰고 다음 실행에서 그 파일의
    # selection_locks를 승계한다. 픽스처 폴더에서 그대로 돌리면 테스트가 자신의
    # 이전 출력에 잠겨 선정을 다시 계산하지 않으므로 임시 복사본에서 실행한다.
    with tempfile.TemporaryDirectory() as temp_dir:
        work_dir = Path(temp_dir)
        for path in case_dir.glob("comparisons_*.json"):
            (work_dir / path.name).write_bytes(path.read_bytes())
        chain = build_citation_chain_from_comparisons(str(work_dir), claims, prior_docs)

    actual = {
        key: {
            "primary_idx": value.get("primary_idx"),
            "secondary_idx": value.get("secondary_idx"),
        }
        for key, value in (chain.get("families") or {}).items()
    }
    assert actual == expected["families"], (
        f"{case_dir.name}의 인용발명 선정이 바뀌었습니다. "
        f"의도한 변경이면 expected.json을 갱신하십시오 "
        f"(verified_by={expected.get('verified_by')})."
    )


def test_goldens_directory_is_seeded() -> None:
    """골든셋이 통째로 비면 회귀 테스트가 조용히 사라지므로 방어한다."""
    assert _golden_cases(), "backend/tests/goldens/에 케이스가 없습니다. seed_golden.py로 추가하십시오."


# ---------------------------------------------------------------------------
# 주인용 자격 게이트 (Gate 1)
# ---------------------------------------------------------------------------

def test_eligibility_gate_drops_clearly_inferior_primary_candidates() -> None:
    details = {
        0: {"distinctive_direct_coverage": 0.90},
        1: {"distinctive_direct_coverage": 0.85},
        2: {"distinctive_direct_coverage": 0.20},  # 핵심 직접개시가 크게 미달
    }
    assert _eligible_primary_indices(details, 3) == [0, 1]


def test_eligibility_gate_is_disabled_when_no_document_discloses_core() -> None:
    """핵심 직접개시가 전부 0이면 자격 판단 근거가 없으므로 전수 탐색으로 되돌린다."""
    details = {index: {"distinctive_direct_coverage": 0.0} for index in range(3)}
    assert _eligible_primary_indices(details, 3) == [0, 1, 2]


def test_shortlist_force_includes_core_disclosing_document() -> None:
    """전체 점수가 낮아도 핵심 구성을 원문으로 직접 개시한 문헌은 후보에 남는다."""
    claim = _claim()
    caches = {
        # 범용 구성만 넓게 개시한 문헌
        0: {"1": [_item("A", "일부 유사", "구동 수단이 있다"),
                  _item("B", "일부 유사", "제어 수단이 있다"),
                  _item("C", "동일", "메모리를 포함한다")]},
        1: {"1": [_item("A", "일부 유사", "구동 수단"),
                  _item("B", "일부 유사", "제어 수단"),
                  _item("C", "동일", "메모리")]},
        2: {"1": [_item("A", "일부 유사", "구동 수단"),
                  _item("B", "일부 유사", "제어 수단"),
                  _item("C", "동일", "메모리")]},
        3: {"1": [_item("A", "일부 유사", "구동 수단"),
                  _item("B", "일부 유사", "제어 수단"),
                  _item("C", "동일", "메모리")]},
        # 핵심 A를 원문으로 직접 개시했지만 나머지는 없는 문헌
        4: {"1": [_item("A", "동일", "제1 회전축과 제2 회전축이 직교 배치되어 동기 회전한다"),
                  _absent("B"), _absent("C")]},
    }
    result = shortlist_primary_candidates([claim], caches, 5)
    included = {candidate["doc_idx"] for candidate in result["candidates"]}
    assert 4 in included
    forced = {c["doc_idx"] for c in result["candidates"] if c["forced_include"]}
    assert 4 in forced or result["candidates"][0]["doc_idx"] == 4


def test_shortlist_skips_adjudication_when_single_eligible_candidate() -> None:
    """후보가 하나뿐이면 LLM-A가 고를 것이 없으므로 판정을 건너뛴다."""
    claim = _claim()
    caches = {
        0: {"1": [_item("A", "동일", "제1 회전축과 제2 회전축이 직교 배치되어 동기 회전한다"),
                  _item("B", "동일", "위상차를 검출해 보정한다"),
                  _item("C", "동일", "메모리")]},
        1: {"1": [_absent("A"), _absent("B"), _item("C", "동일", "메모리")]},
    }
    result = shortlist_primary_candidates([claim], caches, 2)
    assert result["algorithm_top1"] == 0
    assert result["needs_adjudication"] is False


# ---------------------------------------------------------------------------
# 보조 후보 게이트 (Gate 2)
# ---------------------------------------------------------------------------

def test_secondary_shortlist_excludes_documents_with_explicit_contrary_teaching() -> None:
    claim = _claim()
    caches = {
        0: {"1": [_item("A", "동일", "직교 동기 회전 구조"), _absent("B"), _item("C", "동일", "메모리")]},
        1: {"1": [_absent("A"), _item("B", "동일", "위상차 보정부", risk="contrary_teaching"), _absent("C")]},
        2: {"1": [_absent("A"), _item("B", "실질적 동일", "위상 오차 보정 회로"), _absent("C")]},
    }
    result = shortlist_secondary_candidates(_claim(), caches, 3, primary_idx=0)
    included = {candidate["doc_idx"] for candidate in result["candidates"]}
    assert 1 not in included
    assert "1" in result["rejected_for_explicit_risk"]
    assert 2 in included


# ---------------------------------------------------------------------------
# 결합 근거 조립 (결정론)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("review,refs,expected", [
    ({"explicit_suggestion_quote_ref": "c1", "technical_conflict": False}, {"c1"}, "explicit"),
    # 실재하지 않는 chunk_id는 명시적 근거로 인정하지 않는다.
    ({"explicit_suggestion_quote_ref": "c9", "technical_conflict": False,
      "field_adjacency": "distant", "io_compatibility": "incompatible",
      "substitution_feasibility": "neither"}, {"c1"}, "unproven"),
    ({"field_adjacency": "adjacent", "io_compatibility": "compatible",
      "substitution_feasibility": "addable", "technical_conflict": False}, set(), "implicit"),
    # 기술적 모순이 있으면 다른 항목이 아무리 좋아도 근거로 승격하지 않는다.
    ({"field_adjacency": "same", "io_compatibility": "compatible",
      "substitution_feasibility": "substitutable", "technical_conflict": True,
      "explicit_suggestion_quote_ref": "c1"}, {"c1"}, "unproven"),
    ({"field_adjacency": "distant", "io_compatibility": "compatible",
      "substitution_feasibility": "addable", "technical_conflict": False}, set(), "unproven"),
])
def test_combination_basis_is_derived_from_facts_not_llm_conclusion(review, refs, expected) -> None:
    assert ra._derive_combination_basis(review, refs) == expected


# ---------------------------------------------------------------------------
# 응답 파싱
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('설명 문장\n{"selected_primary_idx": 2}', {"selected_primary_idx": 2}),
    ('{"x": "닫는 괄호 } 포함", "y": 1}', {"x": "닫는 괄호 } 포함", "y": 1}),
    ("json이 없음", None),
    ('{"broken": ', None),
])
def test_json_extraction_tolerates_cli_wrapping(raw, expected) -> None:
    assert ra._extract_json_object(raw) == expected


# ---------------------------------------------------------------------------
# LLM-A 검증과 폴백
# ---------------------------------------------------------------------------

def _two_way_caches() -> dict:
    return {
        0: {"1": [_item("A", "실질적 동일", "직교 배치된 두 축이 함께 회전한다"),
                  _absent("B"), _item("C", "동일", "메모리")]},
        1: {"1": [_item("A", "동일", "제1 회전축과 제2 회전축이 직교하여 동기 회전한다"),
                  _absent("B"), _item("C", "동일", "메모리")]},
        2: {"1": [_absent("A"), _item("B", "실질적 동일", "위상차 검출 보정 회로", chunk_id="[0007]"),
                  _absent("C")]},
    }


def _run_adjudication(response: str):
    claim = _claim()
    caches = _two_way_caches()
    with patch.object(ra, "call_ai", new=AsyncMock(return_value=response)), \
         patch.object(ra, "_cache_read", return_value=None), \
         patch.object(ra, "_cache_write", return_value=None):
        return asyncio.run(
            ra.adjudicate_family([claim], caches, _docs(3), _settings())
        )


def test_llm_choice_inside_shortlist_is_applied() -> None:
    result = _run_adjudication(json.dumps({
        "selected_primary_idx": 0,
        "critical_feature_basis": ["A"],
        "reason": "직교 동기 회전을 직접 개시함",
        "confidence": "high",
        # LLM-B 응답과 같은 함수를 공유하므로 보조 필드도 함께 넣는다.
        "selected_secondary_idx": 2,
        "field_adjacency": "same",
        "io_compatibility": "compatible",
        "substitution_feasibility": "addable",
        "technical_conflict": False,
        "explicit_suggestion_quote_ref": None,
    }))
    assert result["primary_idx"] == 0
    assert result["primary_decision"]["source"] == "llm"
    assert result["audit"]["changed_primary"] is True
    assert "llm_overrode_primary" in result["audit"]["critical_reason_codes"]


def test_llm_choice_outside_shortlist_falls_back_to_algorithm() -> None:
    result = _run_adjudication(json.dumps({"selected_primary_idx": 99}))
    assert result["primary_decision"]["source"] == "algorithm"
    assert "out_of_shortlist" in result["primary_decision"]["fallback_reason"]
    assert result["primary_idx"] == result["audit"]["algorithm_top1"]


def test_unparsable_response_falls_back_to_algorithm() -> None:
    result = _run_adjudication("모델이 설명만 하고 JSON을 내지 않았습니다.")
    assert result["primary_decision"]["source"] == "algorithm"
    assert result["primary_decision"]["fallback_reason"] == "unparsable_response"


def test_call_failure_falls_back_to_algorithm() -> None:
    claim = _claim()
    caches = _two_way_caches()
    with patch.object(ra, "call_ai", new=AsyncMock(side_effect=RuntimeError("CLI 없음"))), \
         patch.object(ra, "_cache_read", return_value=None), \
         patch.object(ra, "_cache_write", return_value=None):
        result = asyncio.run(ra.adjudicate_family([claim], caches, _docs(3), _settings()))
    assert result["primary_decision"]["source"] == "algorithm"
    assert result["primary_decision"]["fallback_reason"].startswith("call_failed")


def test_fabricated_quote_ref_is_rejected_and_downgrades_basis() -> None:
    """원문에 없는 chunk_id를 결합 근거로 쓰면 명시적 근거로 인정하지 않는다."""
    result = _run_adjudication(json.dumps({
        "selected_primary_idx": 1,
        "selected_secondary_idx": 2,
        "field_adjacency": "distant",
        "io_compatibility": "incompatible",
        "substitution_feasibility": "neither",
        "technical_conflict": False,
        "explicit_suggestion_quote_ref": "[9999]",
    }))
    review = result["secondary_decision"]["technical_review"]
    assert review["explicit_suggestion_quote_ref"] is None
    assert review["quote_ref_rejected"] == "[9999]"
    assert result["combination_basis"] == "unproven"


# ---------------------------------------------------------------------------
# 판정 결과의 체인 반영
# ---------------------------------------------------------------------------

def test_selection_honours_valid_adjudication() -> None:
    claim = _claim()
    caches = _two_way_caches()
    context = _compute_family_context([claim], caches, 3)
    eligible = context and shortlist_primary_candidates([claim], caches, 3, context)["eligible_indices"]
    assert 0 in eligible and 1 in eligible

    selection = _select_family_reference_pair(
        [claim], caches, 3, adjudication={"primary_idx": 0, "secondary_idx": 2}
    )
    assert selection["primary_idx"] == 0
    assert selection["secondary_idx"] == 2
    assert selection["selection_method"].endswith("llm_adjudicated")


def test_explicit_no_secondary_is_not_overridden_by_algorithm() -> None:
    """기술 검토가 결합 불가로 판정하면 알고리즘이 보조문헌을 도로 넣지 않는다."""
    claim = _claim()
    caches = _two_way_caches()
    selection = _select_family_reference_pair(
        [claim], caches, 3,
        adjudication={"primary_idx": 1, "secondary_idx": None, "secondary_explicitly_none": True},
    )
    assert selection["primary_idx"] == 1
    assert selection["secondary_idx"] is None

    # 반대로 판정이 없거나 폴백한 경우에는 기존 알고리즘 선정을 그대로 쓴다.
    fallback = _select_family_reference_pair(
        [claim], caches, 3,
        adjudication={"primary_idx": 1, "secondary_idx": None},
    )
    assert fallback["secondary_idx"] is not None


def test_shortlisted_document_is_always_selectable_as_primary() -> None:
    """후보로 제시한 문헌은 반드시 자격 게이트도 통과해야 한다.

    LLM에 고르라고 준 후보를 나중에 "자격 미달"로 되돌리면, 판정이 조용히
    폐기되고 LLM 호출이 낭비된다. 두 경로가 같은 풀을 쓰는지 확인한다.
    """
    claim = _claim()
    caches = dict(_two_way_caches())
    # 핵심 A만 정확히 짚고 나머지를 놓쳐 평균 점수가 낮은 문헌
    caches[3] = {"1": [
        _item("A", "동일", "제1 회전축과 제2 회전축이 직교하도록 배치되어 동기 회전한다"),
        _absent("B"), _absent("C"),
    ]}
    shortlist = shortlist_primary_candidates([claim], caches, 4)
    context = _compute_family_context([claim], caches, 4)
    eligible = set(shortlist["eligible_indices"])

    for candidate in shortlist["candidates"]:
        assert candidate["doc_idx"] in eligible, (
            f"doc[{candidate['doc_idx']}]이 후보에는 있으나 자격 게이트를 통과하지 못합니다."
        )
        selection = _select_family_reference_pair(
            [claim], caches, 4,
            adjudication={"primary_idx": candidate["doc_idx"], "secondary_idx": None},
        )
        assert selection["primary_idx"] == candidate["doc_idx"]


def test_selection_ignores_adjudication_outside_eligibility_gate() -> None:
    """자격 게이트를 통과하지 못한 문헌은 판정으로도 주인용이 될 수 없다.

    doc 3은 차별적 핵심(A·B)을 전혀 개시하지 않고 범용 구성(C)만 가진 문헌이라
    보조문헌으로는 몰라도 주인용 자리에는 앉을 수 없어야 한다.
    """
    claim = _claim()
    caches = dict(_two_way_caches())
    caches[3] = {"1": [_absent("A"), _absent("B"), _item("C", "동일", "데이터를 저장하는 메모리")]}

    context = _compute_family_context([claim], caches, 4)
    assert 3 not in _eligible_primary_indices(context["primary_details"], 4)

    selection = _select_family_reference_pair(
        [claim], caches, 4, adjudication={"primary_idx": 3, "secondary_idx": 0}
    )
    assert selection["primary_idx"] != 3
    assert not selection["selection_method"].endswith("llm_adjudicated")


# ---------------------------------------------------------------------------
# 캐시 키
# ---------------------------------------------------------------------------

def test_cache_key_changes_with_prompt_version_and_model() -> None:
    claim = _claim()
    caches = _two_way_caches()
    shortlist = shortlist_primary_candidates([claim], caches, 3)

    base = ra._cache_key(claim, caches, shortlist, "primary", "model-a")
    other_model = ra._cache_key(claim, caches, shortlist, "primary", "model-b")
    assert base != other_model

    original = ra.ADJUDICATION_PROMPT_VERSION
    try:
        ra.ADJUDICATION_PROMPT_VERSION = original + 1
        bumped = ra._cache_key(claim, caches, shortlist, "primary", "model-a")
    finally:
        ra.ADJUDICATION_PROMPT_VERSION = original
    assert base != bumped


def test_cache_key_changes_when_comparison_judgment_changes() -> None:
    claim = _claim()
    caches = _two_way_caches()
    shortlist = shortlist_primary_candidates([claim], caches, 3)
    base = ra._cache_key(claim, caches, shortlist, "primary", "m")

    mutated = {key: json.loads(json.dumps(value)) for key, value in caches.items()}
    mutated[0]["1"][0]["judgment"] = "일부 차이"
    assert base != ra._cache_key(claim, mutated, shortlist, "primary", "m")


# ---------------------------------------------------------------------------
# 문헌 수에 따른 우회
# ---------------------------------------------------------------------------

def test_adjudication_is_skipped_for_single_document_case() -> None:
    """문헌이 2개 미만이면 고를 대상이 없으므로 LLM을 부르지 않는다."""
    call = AsyncMock()
    with tempfile.TemporaryDirectory() as temp_dir, patch.object(ra, "call_ai", new=call):
        result = asyncio.run(
            ra.adjudicate_all(temp_dir, [_claim()], _docs(1), _settings())
        )
    assert result == {}
    call.assert_not_awaited()


def test_novelty_complete_document_skips_adjudication() -> None:
    """단일 문헌 신규성 게이트를 통과하면 결합 판정 자체를 하지 않는다."""
    claim = _claim()
    caches = {
        0: {"1": [_item("A", "동일", "제1 회전축과 제2 회전축이 직교하여 동기 회전한다"),
                  _item("B", "동일", "위상차를 검출하여 보정한다"),
                  _item("C", "동일", "데이터를 저장하는 메모리")]},
        1: {"1": [_absent("A"), _absent("B"), _absent("C")]},
    }
    call = AsyncMock()
    with patch.object(ra, "call_ai", new=call):
        result = asyncio.run(ra.adjudicate_family([claim], caches, _docs(2), _settings()))
    assert result["skipped"] == "novelty_single_reference"
    assert result["primary_idx"] == 0
    call.assert_not_awaited()
