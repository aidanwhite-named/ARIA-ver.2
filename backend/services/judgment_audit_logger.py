"""
판단 추적 및 감사 로그 생성 모듈 (Judgment Audit Logger)

특허 청구항 구성대비 및 인용발명 선정 과정에서 LLM 및 알고리즘이 내린 판단 근거,
각 구성요소별 평가 밴드, 미개시 제한, 주/보조 문헌 선정 이유, 최종 결론 도출 배경을
상세하게 기록하여 추후 판단 차이 분석 및 교정(Calibration)에 활용합니다.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from backend.paths import CASES_DIR, REPORTS_DIR, UPLOADS_DIR

logger = logging.getLogger(__name__)


def generate_judgment_audit_log(job_id: str, claim_number: int = 1) -> Optional[Path]:
    """해당 케이스 및 청구항에 대한 상세 판단 추적 감사 로그(JSON & Markdown)를 생성합니다."""
    job_dir = UPLOADS_DIR / job_id
    case_dir = CASES_DIR / job_id

    # 소스 파일 로드
    claims_data = _load_json(job_dir / "claims.json") or _load_json(case_dir / "parsed" / "claims.json") or []
    prior_docs_data = _load_json(job_dir / "prior_docs.json") or _load_json(case_dir / "parsed" / "prior_docs.json") or []
    citation_chain = _load_json(job_dir / "citation_chain.json") or _load_json(case_dir / "parsed" / "citation_chain.json") or {}
    comparison_matrix = _load_json(case_dir / "parsed" / "comparison_matrix.json") or {}
    if not comparison_matrix:
        # job_dir 내 glob 탐색
        comp_dict = {}
        for p in sorted(job_dir.glob("comparisons_*.json")):
            comp_dict[p.stem] = _load_json(p) or {}
        comparison_matrix = comp_dict

    settings = _load_json(Path("backend/settings.json")) or {}

    if not claims_data or not prior_docs_data:
        logger.warning("Audit log skipped for %s: missing claims or prior_docs", job_id)
        return None

    # 청구항 및 인용발명 매핑
    target_claim = next((c for c in claims_data if c.get("claim_number") == claim_number), claims_data[0] if claims_data else {})
    doc_map = {idx: doc for idx, doc in enumerate(prior_docs_data)}

    # 인용 체인 정보
    chain_info = citation_chain.get("chains", {}).get(str(claim_number)) or {}
    family_key = str(chain_info.get("family_root") or claim_number)
    family_info = citation_chain.get("families", {}).get(family_key) or {}
    doc_name_mapping = citation_chain.get("doc_name_mapping") or {}
    inv_scores = citation_chain.get("inv_scores") or {}
    primary_scores = chain_info.get("primary_scores") or citation_chain.get("primary_score_details") or {}

    # 요소별 상세 매칭 수집
    elements = target_claim.get("elements", [])
    element_audit_list = []

    for elem in elements:
        label = elem.get("label", "")
        elem_text = elem.get("text", "")
        elem_importance = elem.get("importance", "3")

        doc_matches = []
        for doc_idx, doc in doc_map.items():
            comp_key = f"comparisons_{doc_idx}"
            doc_comp = comparison_matrix.get(comp_key, {}).get(str(claim_number), [])
            match = next((item for item in doc_comp if item.get("label") == label), {})

            doc_name = doc_name_mapping.get(str(doc_idx)) or doc.get("filename") or f"D{doc_idx + 1}"
            doc_matches.append({
                "doc_idx": doc_idx,
                "doc_name": doc_name,
                "filename": doc.get("filename", ""),
                "publication_no": doc.get("publication_no", ""),
                "judgment": match.get("judgment", "대응 없음"),
                "technical_judgment": (
                    match.get("technical_judgment")
                    or match.get("llm_judgment")
                    or match.get("judgment", "대응 없음")
                ),
                "judgment_adjustment_reason": match.get("judgment_adjustment_reason", ""),
                "directness": match.get("directness", "absent"),
                "evidence_status": match.get(
                    "evidence_status",
                    "verified" if match.get("quote") else "absent",
                ),
                "quote": match.get("quote", ""),
                "quote_translation": match.get("quote_translation", ""),
                "unverified_quote": match.get("unverified_quote", ""),
                "reason": match.get("판단_이유") or match.get("similarity_reason", ""),
                "missing_limitations": match.get("missing_limitations", []),
                "chunk_id": match.get("chunk_id", ""),
            })

        element_audit_list.append({
            "label": label,
            "text": elem_text,
            "importance": elem_importance,
            "doc_matches": doc_matches,
        })

    # 감사 로그 구조체 생성
    audit_record = {
        "job_id": job_id,
        "claim_number": claim_number,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {
            "engine": settings.get("engine", "agy"),
            "model_parser": settings.get("model_parser", ""),
            "model_compare": settings.get("model_compare", ""),
            "model_report": settings.get("model_report", ""),
            "comparison_mode": settings.get("comparison_mode", "mixed"),
        },
        "claim_summary": {
            "claim_number": claim_number,
            "claim_type": target_claim.get("claim_type", "independent"),
            "preamble": target_claim.get("preamble", ""),
            "text": target_claim.get("text", ""),
            "element_count": len(elements),
        },
        "prior_documents": [
            {
                "doc_idx": idx,
                "filename": doc.get("filename", ""),
                "publication_no": doc.get("publication_no", ""),
                "title": doc.get("title", ""),
                "score": inv_scores.get(str(idx), 0.0),
            }
            for idx, doc in doc_map.items()
        ],
        "selection_audit": {
            "primary_idx": chain_info.get("primary_idx"),
            "secondary_idx": chain_info.get("secondary_idx"),
            "total_chain": chain_info.get("total", []),
            "analysis_track": chain_info.get("analysis_track", ""),
            "combination_rationale_type": chain_info.get("combination_rationale_type", ""),
            "score_details": primary_scores,
            "pair_candidates": family_info.get("pair_candidates", []),
        },
        "element_audits": element_audit_list,
    }

    # JSON 로그 저장
    json_path = case_dir / "parsed" / f"judgment_audit_claim{claim_number}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit_record, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown 감사 보고서 생성
    md_content = _build_markdown_audit_report(audit_record)
    md_path = case_dir / "reports" / f"judgment_audit_claim{claim_number}.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_content, encoding="utf-8")

    # REPORTS_DIR에도 복사
    public_md_path = REPORTS_DIR / f"judgment_audit_{job_id}_claim{claim_number}.md"
    public_md_path.write_text(md_content, encoding="utf-8")

    logger.info("Judgment audit log successfully written to %s", md_path)
    return md_path


def _build_markdown_audit_report(audit: dict) -> str:
    lines = []
    lines.append(f"# [판단 추적 및 감사 로그] Case: {audit['job_id']} (청구항 {audit['claim_number']})")
    lines.append(f"* 생성 시각: {audit['timestamp']}")
    lines.append(f"* 사용 엔진 및 모델: {audit['settings']['engine'].upper()} / Parser: `{audit['settings']['model_parser']}` / Compare: `{audit['settings']['model_compare']}` / Report: `시스템 템플릿`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 청구항 구조
    lines.append("## 1. 대상 청구항 구성 요약")
    lines.append(f"**전제부**: {audit['claim_summary']['preamble'] or '없음'}")
    lines.append(f"**청구항 전체 문언**:\n```text\n{audit['claim_summary']['text']}\n```")
    lines.append("")

    # 2. 인용발명 점수 및 주/보조 문헌 선정 이유
    lines.append("## 2. 인용발명 점수 및 채택 이유 (Selection Audit)")
    lines.append("| 문헌 ID | 파일명 | 공개번호/제목 | 단독 적합도 점수 | 최종 역할 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- |")

    sel = audit["selection_audit"]
    total_chain = sel.get("total_chain", [])
    primary_idx = sel.get("primary_idx")
    secondary_idx = sel.get("secondary_idx")

    for doc in audit["prior_documents"]:
        idx = doc["doc_idx"]
        role = "미채택"
        if idx == primary_idx:
            role = "⭐ 주 인용발명 (Primary)"
        elif idx == secondary_idx:
            role = "🔹 보완 인용발명 (Secondary)"
        elif idx in total_chain:
            role = "🔸 예외 보완문헌"

        score_str = f"{doc['score']:.2f}점" if isinstance(doc['score'], (int, float)) else str(doc['score'])
        lines.append(f"| D{idx + 1} | {doc['filename']} | {doc['publication_no'] or doc['title']} | {score_str} | {role} |")

    lines.append("")
    lines.append(f"* **분석 트랙**: `{sel.get('analysis_track', 'N/A')}`")
    lines.append(f"* **결합 유형**: `{sel.get('combination_rationale_type', 'N/A')}`")
    lines.append("")
    pair_candidates = sel.get("pair_candidates") or []
    if pair_candidates:
        lines.append("### 검토된 주문헌·보조문헌 조합")
        lines.append("| 주문헌 | 보조문헌 | 결합 유사도 | 잔여 구성 |")
        lines.append("| :--- | :--- | :--- | :--- |")
        for pair in pair_candidates[:8]:
            primary = pair.get("primary_idx")
            secondary = pair.get("secondary_idx")
            similarity = pair.get("similarity", {}).get("combined_similarity", 0)
            remaining = pair.get("combination_validity", {}).get(
                "remaining_uncovered_labels", []
            )
            lines.append(
                f"| D{primary + 1 if isinstance(primary, int) else '-'} | "
                f"{f'D{secondary + 1}' if isinstance(secondary, int) else '없음'} | "
                f"{similarity} | {', '.join(remaining) if remaining else '없음'} |"
            )
        lines.append("")

    # 3. 구성요소별 인용발명대비 정밀 추적 매트릭스
    lines.append("## 3. 구성요소별 4개 인용발명 정밀 판정 추적 (Element Audit)")

    for elem in audit["element_audits"]:
        lines.append(f"### [구성요소 {elem['label']}] (중요도: {elem['importance']})")
        lines.append(f"> {elem['text']}")
        lines.append("")

        for match in elem["doc_matches"]:
            d_name = match["doc_name"]
            j_str = match["judgment"]
            dir_str = match["directness"]
            technical_judgment = match.get("technical_judgment") or j_str
            adjustment_reason = match.get("judgment_adjustment_reason", "")
            evidence_status = match.get("evidence_status", "absent")
            reason = match["reason"] or "사유 미기재"
            missing = match["missing_limitations"]
            quote = match["quote"]
            trans = match["quote_translation"]
            unverified_quote = match.get("unverified_quote", "")

            icon = "⚪"
            if j_str in ("동일", "95% 이상: 동일"):
                icon = "🔵"
            elif j_str in ("실질적 동일", "실질적동일"):
                icon = "🟢"
            elif j_str in ("일부 차이", "일부차이"):
                icon = "🟠"
            elif j_str in ("일부 유사", "일부유사"):
                icon = "🟡"

            lines.append(f"* **{d_name} ({match['filename']})**: {icon} `{j_str}` (개시 방식: `{dir_str}`)")
            if technical_judgment != j_str or adjustment_reason:
                lines.append(
                    f"  - **판정 보정 추적**: 기술 판단 `{technical_judgment}` → 최종 `{j_str}`"
                    + (f" (`{adjustment_reason}`)" if adjustment_reason else "")
                )
            if evidence_status not in {"verified", "absent"}:
                lines.append(f"  - **증거 검증 상태**: `{evidence_status}`")
            lines.append(f"  - **판단 이유**: {reason}")
            if missing:
                lines.append(f"  - **누락/차이 제한사항**: {', '.join(missing)}")
            if quote:
                if trans:
                    lines.append(f"  - **원문 번역**: {trans}")
                lines.append(f"  - **원문 발췌**: \"{quote[:200]}\" ({match['chunk_id']})")
            if unverified_quote:
                lines.append(
                    f"  - **검증 실패 인용(판정 근거로 미사용)**: \"{unverified_quote[:200]}\""
                )
            lines.append("")

    # 4. 종합 판단 및 교정(Calibration) 가이드
    lines.append("## 4. 판단 교정 및 핵심 격차 분석 (Calibration Guide)")
    lines.append("1. **주 인용발명 선정 교정**: 평균 유사도점수뿐만 아니라 차별적 핵심구성의 직접 발췌 여부에 가중치가 부여되었는지 검토합니다.")
    lines.append("2. **진보성 인정/부정의 핵심 요소**: 차이점이 존재하는 구성요소(예: 비전 AI 기반 형상추출, 제조설비 전송 등)의 누락 여부가 결합 후에도 보완되지 않았는지 최종 점검합니다.")
    lines.append("")

    return "\n".join(lines)


def _load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
