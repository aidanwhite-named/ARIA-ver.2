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
    quote = bool(str(item.get("quote") or "").strip())
    location = bool(str(item.get("chunk_id") or item.get("paragraph_no") or "").strip())
    reason = bool(str(item.get("판단_이유") or item.get("similarity_reason") or "").strip())
    return 0.55 + 0.25 * quote + 0.10 * location + 0.10 * reason


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
    """보고서에 삽입할 중립적 요약을 만든다."""
    if not assessment:
        return ""
    metrics = assessment.get("metrics") or {}
    uncovered = assessment.get("uncovered_labels") or []
    critical = assessment.get("critical_uncovered_labels") or []
    lines = [
        "[정량평가 - 분석 보조지표]",
        "",
        f"- 주 문헌 커버리지: {metrics.get('primary_coverage', 0):.1f}",
        f"- 결합 커버리지: {metrics.get('combined_coverage', 0):.1f}",
        f"- 근거 조정 강도: {metrics.get('evidence_strength', 0):.1f}",
        f"- 평가 신뢰도: {metrics.get('assessment_reliability', 0):.1f}",
        f"- 보조 문헌 의존도: {metrics.get('complement_dependency', 0):.1f}",
        f"- 미커버 구성: {', '.join(map(str, uncovered)) if uncovered else '없음'}",
        f"- 고중요도 미커버 구성: {', '.join(map(str, critical)) if critical else '없음'}",
        "- 주의: 위 수치는 구조화된 구성대비와 출처 존재 여부를 집계한 분석 보조지표이며, "
        "선행기술 적격일·결합 동기·통상의 지식·법적 결론을 포함하지 않습니다.",
    ]
    dependency = assessment.get("dependency") or {}
    if dependency:
        lines.extend([
            "",
            "[종속항 추가한정 평가]",
            "",
            f"- 부모 청구항: 제{dependency.get('parent_claim')}항",
            f"- 상속 문헌 인덱스: {dependency.get('inherited_documents') or '없음'}",
            f"- 추가 문헌 인덱스: {dependency.get('added_documents') or '없음'}",
            f"- 상속 문헌 커버리지: {dependency.get('inherited_coverage', 0):.1f}",
            f"- 추가 문헌 적용 후 커버리지: {dependency.get('coverage_after_added_documents', 0):.1f}",
            f"- 추가 문헌에 의한 증가분: {dependency.get('coverage_gain', 0):.1f}",
            f"- 판단 근거: {dependency.get('basis_summary', '')}",
            f"- 문헌 선정 기준: {dependency.get('selection_basis') or '추가 문헌 없음'}",
            "- 평가 범위: 부모항 전체를 재평가하지 않고 해당 종속항에 새로 부가된 한정을 평가합니다.",
        ])
    return "\n".join(lines)
