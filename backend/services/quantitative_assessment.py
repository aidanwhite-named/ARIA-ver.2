"""기술분야 중립적인 청구항-문헌 정량평가.

점수는 분석 감사(audit)를 위한 보조 지표이며 법적 결론이 아니다. 기술
키워드를 사용하지 않고 구조화된 대비 결과, 중요도 및 출처 존재 여부만 쓴다.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


DEFAULT_LABEL_VALUES = {
    "동일": 1.00, "실질적 동일": 0.85, "일부 차이": 0.55,
    "일부 유사": 0.35, "차이": 0.15, "대응 없음": 0.00,
}


def _importance(value: Any) -> int:
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return 3


def _label(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "")


def _items(cache: Optional[Dict], claim_number: int) -> Dict[str, Dict]:
    values = (cache or {}).get(str(claim_number), [])
    if not isinstance(values, list):
        return {}
    return {_label(v.get("label")): v for v in values if isinstance(v, dict) and v.get("label")}


def _evidence_factor(item: Dict) -> float:
    # 4대 정량 요소: 0.40*개시충족도 + 0.25*문헌근거성 + 0.25*결합관계충족도 + 0.10*구현구체성 - 추론감점
    directness = str(item.get("directness") or "").strip().lower()
    quote = bool(str(item.get("quote") or "").strip())
    location = bool(str(item.get("chunk_id") or item.get("paragraph_no") or "").strip())
    reason = bool(str(item.get("판단_이유") or item.get("similarity_reason") or "").strip())
    has_missing = bool(item.get("missing_limitations"))
    
    disclosure = 1.0 if directness == "direct" else 0.7 if directness == "inferred" else 0.2
    evidence = (0.5 * quote + 0.5 * location) if (quote or location) else 0.2
    relational = 0.9 if (not has_missing and directness == "direct") else 0.5 if not has_missing else 0.3
    concreteness = 1.0 if (reason and quote) else 0.6 if reason else 0.3
    inference_penalty = 0.25 if directness == "inferred" else 0.0

    raw_score = 0.40 * disclosure + 0.25 * evidence + 0.25 * relational + 0.10 * concreteness
    return max(0.0, min(1.0, raw_score - inference_penalty))


def assess_claim(
    claim,
    document_caches: List[Optional[Dict]],
    selected_documents: Optional[Iterable[int]] = None,
    *,
    label_values: Optional[Dict[str, float]] = None,
    coverage_threshold: float = 0.55,
    critical_threshold: float = 0.35,
) -> Dict:
    """평균점수와 필수구성 결손을 분리해 청구항 하나를 평가한다."""
    values = dict(DEFAULT_LABEL_VALUES)
    values.update(label_values or {})
    selected = list(selected_documents) if selected_documents is not None else list(range(len(document_caches)))
    selected = [i for i in selected if 0 <= i < len(document_caches)]
    per_doc = {i: _items(document_caches[i], claim.claim_number) for i in selected}

    rows, uncovered, critical = [], [], []
    total = primary_sum = combined_sum = evidence_sum = complement_weight = 0.0
    for element in claim.elements:
        label = _label(element.label)
        weight = float(_importance(element.importance))
        candidates = []
        for idx in selected:
            item = per_doc[idx].get(label, {})
            candidates.append((max(0.0, min(1.0, float(values.get(item.get("judgment"), 0.0)))), idx, item))
        candidates.sort(key=lambda row: row[0], reverse=True)
        best, best_doc, best_item = candidates[0] if candidates else (0.0, None, {})
        primary_item = per_doc[selected[0]].get(label, {}) if selected else {}
        primary = max(0.0, min(1.0, float(values.get(primary_item.get("judgment"), 0.0))))
        adjusted = best * _evidence_factor(best_item)

        total += weight
        primary_sum += weight * primary
        combined_sum += weight * best
        evidence_sum += weight * adjusted
        if best < coverage_threshold:
            uncovered.append(element.label)
        is_critical_gap = weight >= 4 and best < critical_threshold
        if is_critical_gap:
            critical.append(element.label)
        if selected and best_doc != selected[0] and best > primary:
            complement_weight += weight
        rows.append({
            "label": element.label, "importance": int(weight),
            "primary_score": round(primary * 100, 1),
            "combined_score": round(best * 100, 1),
            "evidence_adjusted_score": round(adjusted * 100, 1),
            "best_document_index": best_doc,
            "has_quote": bool(str(best_item.get("quote") or "").strip()),
            "covered": best >= coverage_threshold,
            "critical_uncovered": is_critical_gap,
        })

    denominator = total or 1.0
    primary_coverage = primary_sum / denominator
    combined_coverage = combined_sum / denominator
    evidence_strength = evidence_sum / denominator
    reliability = (
        2 * combined_coverage * evidence_strength / (combined_coverage + evidence_strength)
        if combined_coverage + evidence_strength else 0.0
    )
    status = (
        "critical_gap" if critical else
        "incomplete_coverage" if uncovered else
        "evidence_review_required" if reliability < 0.70 else
        "coverage_supported"
    )

    gap_coverage_by_doc = {}
    for idx in selected:
        doc_gap_covered = []
        for element in claim.elements:
            if element.label in uncovered or element.label in critical:
                doc_item = per_doc[idx].get(_label(element.label), {})
                doc_sim = max(0.0, min(1.0, float(values.get(doc_item.get("judgment"), 0.0))))
                if doc_sim >= coverage_threshold:
                    doc_gap_covered.append(element.label)
        gap_coverage_by_doc[idx] = doc_gap_covered

    return {
        "method_version": "generic-evidence-score-v1",
        "claim_number": claim.claim_number,
        "selected_documents": selected,
        "metrics": {
            "primary_coverage": round(primary_coverage * 100, 1),
            "combined_coverage": round(combined_coverage * 100, 1),
            "evidence_strength": round(evidence_strength * 100, 1),
            "assessment_reliability": round(reliability * 100, 1),
            "complement_dependency": round(complement_weight / denominator * 100, 1),
        },
        "status": status,
        "uncovered_labels": uncovered,
        "critical_uncovered_labels": critical,
        "gap_coverage_by_doc": gap_coverage_by_doc,
        "elements": rows,
        "interpretation": {
            "score_role": "분석 우선순위와 근거 충실도를 점검하는 보조 지표",
            "not_included": [
                "선행기술 적격일 판단", "결합 동기의 법적 판단",
                "기술분야별 통상의 지식", "청구항 해석의 최종 판단",
            ],
        },
    }


def assess_claims(claims, document_caches: List[Optional[Dict]], chains: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    chain_map = chains or {}
    for claim in claims:
        key = str(claim.claim_number)
        chain = chain_map.get(key, {})
        total = list(chain.get("total") or [])
        assessment = assess_claim(claim, document_caches, selected_documents=total)
        if claim.claim_type == "dependent":
            inherited = list(chain.get("inherited") or [])
            added = list(chain.get("added") or [])
            inherited_assessment = assess_claim(
                claim, document_caches, selected_documents=inherited,
            )
            total_coverage = assessment["metrics"]["combined_coverage"]
            inherited_coverage = inherited_assessment["metrics"]["combined_coverage"]
            decision_trace = dict(chain.get("decision_trace") or {})
            remaining = assessment["uncovered_labels"]
            if not inherited:
                basis_summary = "부모항 인용체인을 확인할 수 없어 명시된 추가 한정만 평가"
            elif not added and not remaining:
                basis_summary = "상속 문헌이 추가 한정까지 대응하여 새 문헌을 추가하지 않음"
            elif added and not remaining:
                basis_summary = "새 문헌 하나가 상속 문헌의 잔여 추가 한정을 보완"
            elif added:
                basis_summary = "새 문헌이 일부 추가 한정을 보완하지만 잔여 미커버가 존재"
            else:
                basis_summary = "잔여 추가 한정을 하나의 새 문헌이 모두 보완하지 못함"
            assessment["dependency"] = {
                "assessment_scope": "additional_limitations",
                "parent_claim": claim.parent_claim,
                "inherited_documents": inherited,
                "added_documents": added,
                "inherited_coverage": inherited_coverage,
                "coverage_after_added_documents": total_coverage,
                "coverage_gain": round(total_coverage - inherited_coverage, 1),
                "remaining_uncovered_labels": assessment["uncovered_labels"],
                "parent_chain_available": bool(inherited),
                "decision_status": decision_trace.get("decision_status", ""),
                "selection_basis": decision_trace.get("selection_basis", ""),
                "basis_summary": basis_summary,
                "decision_trace": decision_trace,
            }
        results[key] = assessment
    return results


def format_assessment_markdown(assessment: Optional[Dict]) -> str:
    """정량평가 데이터는 내부 검색/감사용으로만 유지하고 보고서에는 출력하지 않는다."""
    return ""
