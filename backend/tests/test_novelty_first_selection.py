from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile
from unittest.mock import AsyncMock, patch

from backend.models.schemas import ClaimElement, ElementMatch, ExtractedDocument, ParsedClaim, Settings
from backend.routers.analyze import _used_inventions_for
from backend.services.citation_chain import (
    _claim_similarity,
    _single_document_disclosure,
    build_citation_chain_from_comparisons,
    get_claim_chain_info,
)
from backend.services.report_generator import (
    _build_system,
    _generate_rejection_impossible_report,
    _make_phase1_prompt,
    enforce_phase1_judgment_headers,
    find_unselected_reference_mentions,
)


def _claim() -> ParsedClaim:
    return ParsedClaim(
        claim_number=1,
        claim_type="independent",
        text="A와 B를 포함하는 시스템",
        elements=[
            ClaimElement(label="A", text="핵심 배열 구조", importance="5"),
            ClaimElement(label="B", text="동기 제어 구조", importance="5"),
        ],
    )


def _item(
    label: str,
    judgment: str,
    quote: str,
    *,
    directness: str = "direct",
    missing: list[str] | None = None,
) -> dict:
    return {
        "label": label,
        "found": bool(quote),
        "quote": quote,
        "chunk_id": "[0001]" if quote else "",
        "judgment": judgment,
        "판단_이유": "직접 대응 여부를 검토함",
        "directness": directness,
        "missing_limitations": missing or [],
        "evidence": ([{"limitation": label, "quote": quote, "chunk_id": "[0001]"}] if quote else []),
        "motivation_quote": "",
        "combination_risk": "none_explicit",
        "combination_risk_reason": "명시적 저해 없음",
    }


def test_single_document_gate_requires_every_limitation_to_be_direct() -> None:
    claim = _claim()
    complete = {
        "1": [
            _item("A", "실질적 동일", "A 직접 근거"),
            _item("B", "동일", "B 직접 근거"),
        ]
    }
    assert _single_document_disclosure(complete, claim)["is_complete"] is True

    inferred = {
        "1": [
            _item("A", "실질적 동일", "A 직접 근거"),
            _item("B", "일부 차이", "B 일부 근거", directness="inferred", missing=["동기 관계"]),
        ]
    }
    result = _single_document_disclosure(inferred, claim)
    assert result["is_complete"] is False
    assert result["missing_or_indirect_labels"] == ["B"]


def test_complete_single_document_preempts_cross_document_combination() -> None:
    claim = _claim()
    caches = [
        {"1": [_item("A", "동일", "A만 직접 개시"), _item("B", "대응 없음", "", directness="absent", missing=["B 전체"])]},
        {"1": [_item("A", "대응 없음", "", directness="absent", missing=["A 전체"]), _item("B", "동일", "B만 직접 개시")]},
        {"1": [_item("A", "실질적 동일", "A 직접 개시"), _item("B", "동일", "B 직접 개시")]},
    ]
    docs = [ExtractedDocument(filename=f"doc{idx}.pdf") for idx in range(3)]

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, cache in enumerate(caches):
            (Path(temp_dir) / f"comparisons_{idx}.json").write_text(
                json.dumps(cache, ensure_ascii=False),
                encoding="utf-8",
            )
        result = build_citation_chain_from_comparisons(temp_dir, [claim], docs)

    family = result["families"]["1"]
    assert family["analysis_track"] == "novelty_single_reference"
    assert family["primary_idx"] == 2
    assert family["secondary_idx"] is None
    assert result["chains"]["1"]["total"] == [2]


def test_no_complete_document_uses_difference_filling_pair() -> None:
    claim = _claim()
    caches = [
        {"1": [_item("A", "동일", "A 직접 개시"), _item("B", "일부 유사", "B 관련 기능", directness="inferred", missing=["동기 관계"])]},
        {"1": [_item("A", "일부 유사", "A 관련 기능", directness="inferred", missing=["배열 구조"]), _item("B", "동일", "B 직접 개시")]},
    ]
    docs = [ExtractedDocument(filename=f"doc{idx}.pdf") for idx in range(2)]

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, cache in enumerate(caches):
            (Path(temp_dir) / f"comparisons_{idx}.json").write_text(
                json.dumps(cache, ensure_ascii=False),
                encoding="utf-8",
            )
        result = build_citation_chain_from_comparisons(temp_dir, [claim], docs)

    family = result["families"]["1"]
    assert family["analysis_track"] == "inventive_step_combination"
    assert family["novelty_screen"]["result"] == "no_single_document_complete"
    assert len(result["chains"]["1"]["total"]) == 2


def test_partial_similarity_boundary_is_not_complete_combination_coverage() -> None:
    claim = ParsedClaim(
        claim_number=1,
        claim_type="independent",
        text="텍스트 연속성에 기초한 씬 그룹핑",
        elements=[ClaimElement(label="B", text="텍스트 연속성에 기초한 씬 그룹핑", importance="5")],
    )
    primary = {
        "1": [_item(
            "B",
            "일부 유사",
            "visual scene boundaries",
            directness="absent",
            missing=["텍스트 연속성에 기초한 그룹핑"],
        )]
    }
    secondary = {
        "1": [_item(
            "B",
            "일부 유사",
            "multimodal consecutive shot boundaries",
            directness="inferred",
            missing=["텍스트 세그먼트 분할 순서"],
        )]
    }

    result = _claim_similarity(primary, secondary, claim)

    assert result["uncovered_labels"] == ["B"]
    assert result["combined_similarity"] <= 35.0


def test_supporting_evidence_pair_cannot_conflict_with_complete_coverage() -> None:
    claim = _claim()
    caches = [
        {"1": [
            _item("A", "동일", "A 직접 개시"),
            _item("B", "일부 유사", "B 관련 기능", directness="absent", missing=["동기 제어 구조"]),
        ]},
        {"1": [
            _item("A", "일부 유사", "A 관련 기능", directness="inferred", missing=["핵심 배열 구조"]),
            _item("B", "일부 유사", "B 보강 근거", directness="inferred", missing=["동기 제어 구조"]),
        ]},
    ]
    docs = [ExtractedDocument(filename=f"doc{idx}.pdf") for idx in range(2)]

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx, cache in enumerate(caches):
            (Path(temp_dir) / f"comparisons_{idx}.json").write_text(
                json.dumps(cache, ensure_ascii=False),
                encoding="utf-8",
            )
        result = build_citation_chain_from_comparisons(temp_dir, [claim], docs)

    family = result["families"]["1"]
    assert family["secondary_detail"]["hard_fill_count"] == 0
    assert family["combination_validity"]["coverage_complete"] is False
    assert family["combination_validity"]["remaining_uncovered_labels"] == ["B"]


def test_report_disclosure_status_is_derived_from_structured_evidence() -> None:
    report = """### [구성요소]

(A) 동일 99%

- 개시 상태: 부분 개시

- 청구항 구성: 배열 구조
"""
    direct = ElementMatch(
        label="A",
        found=True,
        quote="직접 근거",
        judgment="실질적 동일",
        directness="direct",
        missing_limitations=[],
    )
    normalized = enforce_phase1_judgment_headers(report, [direct])
    assert "(A) 실질적동일 92%" in normalized
    assert "- 개시 상태: 직접 개시" in normalized

    partial = direct.model_copy(update={
        "judgment": "일부 차이",
        "directness": "inferred",
        "missing_limitations": ["중심 중복"],
    })
    normalized = enforce_phase1_judgment_headers(report, [partial])
    assert "- 개시 상태: 부분 개시" in normalized


def test_report_reference_scope_rejects_documents_outside_final_chain() -> None:
    chain = {
        "total": [1],
        "doc_name_mapping": {"1": "인용발명 1", "0": "인용발명 2"},
        "common_general_knowledge": {"labels": ["F"]},
    }
    assert find_unselected_reference_mentions("인용발명 1과 주지관용을 검토한다.", chain) == []
    assert find_unselected_reference_mentions(
        "차이점은 인용발명 2의 발췌로 보완된다.", chain
    ) == ["인용발명 2"]


def test_independent_report_prompt_uses_locked_global_reference_name() -> None:
    claim = ParsedClaim(
        claim_number=6,
        claim_type="independent",
        text="통신 방법",
        elements=[ClaimElement(label="A", text="패킷을 처리하는 단계")],
    )
    matches = [
        ElementMatch(
            label="A",
            cited_invention_index=0,
            judgment="일부 차이",
            quote="packet processing",
            chunk_id="[0001]",
        )
    ]
    chain = {
        "total": [0],
        "doc_name_mapping": {"1": "인용발명 1", "0": "인용발명 2"},
        "analysis_track": "inventive_step_combination",
    }

    prompt = _make_phase1_prompt(
        claim,
        matches,
        [ExtractedDocument(filename="selected.pdf"), ExtractedDocument(filename="locked.pdf")],
        chain,
        Settings(),
    )
    system = _build_system(Settings(), "independent", chain)

    assert "[인용발명 2]" in prompt
    assert "각 구성요소는 인용발명 2 기준" in prompt
    assert "인용발명 1" not in prompt
    assert "인용발명 1" not in system
    assert find_unselected_reference_mentions(prompt, chain) == []


def test_rejection_impossible_prompt_uses_locked_global_reference_name() -> None:
    claim = ParsedClaim(
        claim_number=6,
        claim_type="independent",
        text="통신 방법",
        elements=[ClaimElement(label="A", text="패킷을 처리하는 단계")],
    )
    matches = [
        ElementMatch(
            label="A",
            cited_invention_index=0,
            judgment="대응 없음",
            quote="",
        )
    ]
    chain = {
        "total": [0],
        "doc_name_mapping": {"1": "인용발명 1", "0": "인용발명 2"},
    }

    with patch(
        "backend.services.report_generator.call_ai",
        new=AsyncMock(return_value="ok"),
    ) as mocked_call:
        result = asyncio.run(
            _generate_rejection_impossible_report(
                claim,
                matches,
                [ExtractedDocument(filename="selected.pdf")],
                Settings(),
                chain,
            )
        )

    prompt, system = mocked_call.await_args.args[:2]
    assert result == "ok"
    assert "허용 문헌: 인용발명 2" in prompt
    assert "(인용발명 2)" in prompt
    assert "인용발명 1" not in prompt
    assert "인용발명 1" not in system
    assert find_unselected_reference_mentions(prompt, chain) == []


def test_used_invention_card_contains_canonical_combination_basis() -> None:
    docs = [
        ExtractedDocument(filename="paper.pdf"),
        ExtractedDocument(filename="US2002.pdf"),
    ]
    chain = {
        "total": [1],
        "doc_name_mapping": {"1": "인용발명 1", "0": "인용발명 2"},
        "reference_roles": {"1": "primary"},
        "analysis_track": "inventive_step_combination",
        "common_general_knowledge": {"labels": ["F"]},
    }

    cards = _used_inventions_for(chain, docs)

    assert cards == [{
        "name": "인용발명 1",
        "filename": "US2002.pdf",
        "role": "primary",
        "basis_label": "인용발명 1 + 주지관용(진보성)",
    }]


def test_common_knowledge_chain_replaces_stale_two_document_rationale() -> None:
    chain_data = {
        "chains": {
            "1": {
                "total": [1],
                "common_general_knowledge": [{"label": "F", "basis": "일반 구성"}],
                "combination_rationale": {
                    "type": "supporting_evidence",
                    "writing_guidance": "문헌 2를 보강 근거로 사용한다.",
                },
            }
        },
        "doc_name_mapping": {"1": "인용발명 1", "0": "인용발명 2"},
    }

    info = get_claim_chain_info(chain_data, 1)

    assert info["combination_rationale_type"] == "common_general_knowledge"
    assert "다른 인용발명은 결합 근거로 사용하지 않는" in info["combination_rationale"]["description"]
    assert "문헌 2" not in info["combination_rationale"]["writing_guidance"]
