"""
보고서 생성기
- Template A: 단일 인용발명 대비 (필요시 주지관용 검토)
- Template B: 복수 인용발명 결합 검토 (예외적으로 주지관용 입증용 제3문헌)
- 종속항 보고서
- 진보성 부정 불가 보고서
- 신규성 부정 보고서 (차이점 없음)
- 카테고리 동일 청구항 처리
"""
from __future__ import annotations
import json
import logging
import re
from string import Template
from typing import Dict, List, Optional

from backend.models.schemas import (
    ClaimElement, ElementMatch, ExtractedDocument, ParsedClaim,
    Settings
)
from backend.services.ai_engine import call_ai, call_ai_streaming
from backend.services.citation_chain import format_inv_list
from backend.services.prompt_loader import load_prompt, render_prompt

logger = logging.getLogger(__name__)

NO_MATCH_LABELS = {"대응 없음"}


def _extract_first_json_object(text: str) -> Optional[Dict]:
    """Return the first complete JSON object without greedy brace matching."""
    cleaned = re.sub(r"```(?:json)?", "", (text or "").strip()).replace("```", "").strip()
    decoder = json.JSONDecoder()
    start = 0
    while True:
        start = cleaned.find("{", start)
        if start == -1:
            return None
        try:
            value, _end = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            start += 1
            continue
        if isinstance(value, dict):
            return value
        start += 1

# ---------------------------------------------------------------------------
# 기본값 상수 — 프론트엔드 설정창에서 "기본값으로 채우기" 시 사용
# ---------------------------------------------------------------------------

DEFAULT_PHASE1_FORMAT = """\
### [구성요소 (알파벳 기호)] 유사도: (동일 / 실질적 동일 / 일부 차이 / 일부 유사 / 차이 / 대응 없음 중 택1) + 구체적 퍼센트(예: 85%)

- 청구항 구성:

- 인용발명 대응 부분 요약:

- **인용발명** 대응 원문:
  [한국어 원문인 경우] 원문 그대로 발췌 후 (단락 [XXXX]) 병기
  [외국어 원문인 경우] 반드시 아래 2줄 구조 사용 (번역 절대 생략 금지):
    한국어 번역문
    (단락 [XXXX] 또는 본문 N 페이지, "원어 원문 발췌")
  [대응 없음인 경우] (인용발명 1에서 해당 구성 확인 불가)

- 판단 이유:

- 차이점:


### 종합 분석 요약

- 유사점 요약:

- 차이점:

- 결합 논리 및 차이점 극복:

- 결론: (구성대비 결과에 따라 거절 근거를 구성할 수 있는지, 또는 추가 근거가 필요한지 중립적으로 작성)"""

# Template B(복수 인용발명) Phase 1 전용 형식
DEFAULT_PHASE1_FORMAT_COMBO = """\
### [구성요소 (알파벳 기호)] 유사도: (동일 / 실질적 동일 / 일부 차이 / 일부 유사 / 차이 / 대응 없음 중 택1) + 구체적 퍼센트(예: 85%)

- 청구항 구성:

- 인용발명 대응 부분 요약:

- **인용발명** 대응 원문:
  [한국어 원문인 경우] 원문 그대로 발췌 후 (단락 [XXXX]) 병기
  [외국어 원문인 경우] 반드시 아래 2줄 구조 사용 (번역 절대 생략 금지):
    한국어 번역문
    (단락 [XXXX] 또는 본문 N 페이지, "원어 원문 발췌")

- 판단 이유:

- 차이점:


### 📊 종합 분석 요약

- 유사점 요약:

- 결합 논리 및 차이점 극복: (차이점마다 인용발명 2의 해당 발췌를 단락 번호와 함께 인용해 극복 논리 제시. 주지관용기술은 문헌 근거가 없는 차이점에만 최후 수단으로 사용. 예외적 인용발명 3은 표시된 일반 구성의 주지관용성 입증에만 사용)

- 결론: (인용발명 1과 2의 핵심 결합 및 필요한 경우 일반 구성에 대한 제한적 주지관용 근거를 구분하여, 거절 근거 구성 가능 여부를 중립적으로 작성)"""

# Phase 2 제목 — "# [Phase 2]" 경계 마커 뒤에 붙는 문구 (시스템이 경계를 전담, 제목만 사용자 편집)
DEFAULT_PHASE2_TITLE = "📑 최종 보고서"

# Phase 2 독립항 출력 양식 (LLM 없음, Python이 자리표시자를 채움)
# 경계선("# [Phase 2] ...")은 시스템이 자동으로 붙이므로 본문 양식에는 포함하지 않는다.
# 자리표시자: ${inv_header} ${components} ${similar} ${diff} ${conclusion}
# - ${inv_header}  : [인용발명 1] 또는 [인용발명 1] [인용발명 2] 로 자동 치환
# - ${components}  : (A) 동일 85% + 원문 발췌 블록 (구성요소별)
# - ${similar}/${diff}/${conclusion} : Phase 1 종합 분석 요약에서 추출
DEFAULT_PHASE2_FORMAT_SINGLE = """\
${inv_header}

[구성대비]

[구성요소]

${components}

[종합 판단]

[유사점]

${similar}

[차이점]

${diff}

[결론]

${conclusion}"""

DEFAULT_PHASE2_FORMAT_COMBO = """\
${inv_header}

[구성대비]

[구성요소]

${components}

[종합 판단]

[유사점]

${similar}

[결합 논리]

${combination_rationale}

[차이점]

${diff}

[결론]

${conclusion}"""

DEFAULT_PHASE1_FORMAT_DEPENDENT = """\
### [추가 구성 (A)] 유사도: (동일 / 실질적 동일 / 일부 차이 / 일부 유사 / 차이 / 대응 없음 중 택1) + 구체적 퍼센트(예: 85%)

- 청구항 추가 구성:

- **인용발명** 대응 원문:

- 판단 이유:

- 차이점:


### 종합 분석 요약

- 유사점 요약:

- 차이점:

- 결론: (추가 구성의 대응 강도와 남은 차이에 따라 거절 근거 구성 가능 여부를 중립적으로 작성)"""

DEFAULT_PHASE2_FORMAT_DEPENDENT = """\
종속항 ${claim_number}

${review_intro}

${analysis}

${parent_basis}

${conclusion}"""

_BASE_SYSTEM = """당신은 대한민국 특허청 심사관 수준의 특허 분석 전문가입니다.

[절대 금지 표현]
- "신규성이 없다", "신규성이 있다" → 사용 금지
- "진보성이 없다", "진보성이 있다" → 사용 금지
- "특허성이 없다/있다" → 사용 금지

[인용 규칙]
- 한국어 인용: 원문 그대로 기재 후 (단락 [XXXX]) 형식
- 영문 인용(특허·비특허 모두) — 2줄 구조를 반드시 지킬 것:
  1줄: 한국어 번역 (번역 생략 금지)
  2줄: (단락 [XXXX] 또는 본문 N 페이지, "원문 영어 인용")
  예시(비특허):
  첫 번째 단계는 입력 이미지에서 장면 중첩을 찾는 대응 검색 단계로서 겹치는 이미지에서 동일한 점들의 투영을 식별한다.
  (본문 2 페이지, "The first stage is correspondence search which finds scene overlap in the input images...")

[출력 형식]
- 마크다운으로 출력
- 독립항 Phase 1의 각 구성요소는 `### [구성요소 (알파벳)]` 헤더로 시작할 것
- 종속항 Phase 1의 각 추가 구성은 `### [추가 구성 (알파벳)]` 헤더로 시작할 것"""


def _phase1_format_text(combo: bool = False) -> str:
    """Phase 1 출력 형식 템플릿 반환 (파일 우선, 없으면 기본값)."""
    if combo:
        return load_prompt("format_phase1_combo.txt", DEFAULT_PHASE1_FORMAT_COMBO)
    return load_prompt("format_phase1_independent.txt", DEFAULT_PHASE1_FORMAT)


def _build_system(settings: Settings, claim_type: str = "independent") -> str:
    """claim_type: 'independent' | 'combo' | 'dependent'"""
    system = load_prompt("system_report_base.txt", _BASE_SYSTEM)

    if claim_type == "combo":
        fmt = load_prompt("format_phase1_combo.txt", DEFAULT_PHASE1_FORMAT_COMBO)
    elif claim_type == "independent":
        fmt = load_prompt("format_phase1_independent.txt", DEFAULT_PHASE1_FORMAT)
    else:  # dependent
        fmt = load_prompt("format_phase1_dependent.txt", DEFAULT_PHASE1_FORMAT_DEPENDENT)
    system += f"\n\n[출력 형식 템플릿]\n{fmt}"

    return system


# ---------------------------------------------------------------------------
# 독립항 보고서 (Template A / B)
# ---------------------------------------------------------------------------

def _build_context_block(prev_context: Optional[List[Dict]]) -> str:
    """이전 청구항 분석 결과를 프롬프트용 컨텍스트 블록으로 변환"""
    if not prev_context:
        return ""
    lines = [
        "[이전 청구항 분석 컨텍스트 — 발명의 전체 맥락 파악용]",
        "※ 아래 이전 청구항 보고서를 참고하여 동일 발명 내 청구항 간 상호 관계와 기술적 맥락을 파악하고,"
        " 현재 청구항 분석과 일관성을 유지하십시오.",
        "",
    ]
    for entry in prev_context:
        lines.append(f"=== 청구항 {entry['claim_number']} 분석 결과 (Phase 2 요약) ===")
        lines.append(entry.get("phase2_summary", "(요약 없음)"))
        lines.append("")
    lines.append("─" * 60)
    lines.append("")
    return "\n".join(lines)


def _combination_rationale_text(chain_info: Optional[Dict]) -> str:
    if not chain_info:
        return ""
    rationale = chain_info.get("combination_rationale") or {}
    label = rationale.get("label", "")
    type_code = rationale.get("type", "")
    description = rationale.get("description", "")
    warnings = rationale.get("warnings") or []
    lines = []
    if label or type_code:
        rationale_label = label or type_code
        lines.append(f"결합 논리 유형: {rationale_label}".strip())
    if description:
        lines.append(f"판단 취지: {description}")
    for warning in warnings:
        lines.append(f"주의: {warning}")
    conventional_support = chain_info.get("conventional_support") or {}
    if conventional_support:
        mapping = chain_info.get("doc_name_mapping", {})
        doc_idx = conventional_support.get("doc_idx")
        doc_name = (
            mapping.get(str(doc_idx), f"인용발명 {int(doc_idx) + 1}")
            if doc_idx is not None else "추가 문헌"
        )
        labels = ", ".join(f"({label})" for label in conventional_support.get("labels", []))
        if conventional_support.get("position") == 3:
            lines.append(
                f"예외적 제3 인용발명 역할: {doc_name}은 {labels or '일반 구성'}의 "
                "주지관용성을 입증하는 자료로만 사용합니다."
            )
        else:
            lines.append(
                f"주지관용 구성의 문헌 근거: {doc_name}은 {labels or '일반 구성'}의 "
                "통상적 채용을 뒷받침하는 자료입니다."
            )
        lines.append("역할 제한: 이 문헌을 핵심 기술사상이나 새로운 상호작용의 보완 근거로 확대하지 않습니다.")
    common_knowledge = chain_info.get("common_general_knowledge") or []
    if common_knowledge:
        labels = ", ".join(f"({item.get('label', '')})" for item in common_knowledge)
        lines.append(f"문헌 없는 주지관용 검토 대상: {labels}")
        lines.append("문헌 근거 없이 주지관용이라고 단정하지 말고, 통상적 기능·단순 결합 가능성과 추가 입증 필요 여부를 구분합니다.")
    return "\n".join(lines).strip()


def _conventional_policy_prompt_block(chain_info: Optional[Dict]) -> str:
    if not chain_info:
        return ""
    if not chain_info.get("conventional_support") and not chain_info.get("common_general_knowledge"):
        return ""
    lines = []
    conventional_support = chain_info.get("conventional_support") or {}
    if conventional_support:
        mapping = chain_info.get("doc_name_mapping", {})
        doc_idx = conventional_support.get("doc_idx")
        doc_name = mapping.get(str(doc_idx), f"인용발명 {int(doc_idx) + 1}")
        labels = ", ".join(f"({label})" for label in conventional_support.get("labels", []))
        position = conventional_support.get("position")
        role = "예외적 제3문헌" if position == 3 else "주지관용 명시근거 문헌"
        lines.append(f"- {role}: {doc_name}, 대상 구성 {labels}")
        lines.append("- 이 문헌은 위 일반 구성의 통상적 채용을 입증하는 용도로만 사용합니다.")
    common_knowledge = chain_info.get("common_general_knowledge") or []
    if common_knowledge:
        labels = ", ".join(f"({item.get('label', '')})" for item in common_knowledge)
        lines.append(f"- 문헌 없는 주지관용 검토 대상: {labels}")
        lines.append("- 통상적 기능과 단순 결합 가능성을 설명하되, 문헌 근거 없이 주지관용이라고 단정하지 않습니다.")
    policy_text = "\n".join(lines)
    return (
        "[주지관용 구성 적용 정책]\n"
        f"{policy_text}\n"
        "주지관용 구성은 핵심 차이점과 분리하여 작성하고, 단순 결합 이상의 새로운 작용효과를 추정하지 마십시오.\n\n"
    )


def _strip_internal_scoring_notes(text: str) -> str:
    """Remove internal scoring/debug rationale from user-facing reports."""
    if not text:
        return text
    cleaned_lines = []
    for line in text.splitlines():
        if re.search(r"SubScore|score_detail|gap_fill|quote_explicitness|reportability", line, re.IGNORECASE):
            continue
        if re.search(r"공백/약점\s*보완도|발췌\s*근거\s*명시성|보고서\s*작성\s*용이성", line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _format_citation_location(match: ElementMatch, prior_docs: List[ExtractedDocument]) -> str:
    """Format a user-facing paragraph/page citation for an extracted match."""
    chunk_id = match.chunk_id
    if not chunk_id:
        return "(위치 정보 없음)"
    doc = prior_docs[match.cited_invention_index] if match.cited_invention_index < len(prior_docs) else None
    if doc and doc.document_type == "non_patent":
        anchor = chunk_id.replace("[P", "").split("-")[0] if "[P" in chunk_id else chunk_id
        return f"(본문 {anchor} 페이지)"
    return f"(단락 {chunk_id})"


def _combination_rationale_prompt_block(chain_info: Optional[Dict]) -> str:
    if not chain_info:
        return ""
    rationale = chain_info.get("combination_rationale") or {}
    if not rationale:
        return ""
    lines = []
    label = rationale.get("label") or rationale.get("type")
    if label:
        lines.append(f"결합 논리 유형: {label}")
    if rationale.get("description"):
        lines.append(f"판단 취지: {rationale['description']}")
    lines.extend(f"주의: {warning}" for warning in rationale.get("warnings", []))
    text = "\n".join(lines)
    return (
        "[인용발명 결합 논리 유형]\n"
        f"{text}\n"
        "위 유형을 우선 기준으로 삼되, 실제 발췌 근거와 맞지 않으면 남는 차이점을 명시하십시오.\n\n"
    )


def _strip_agent_tool_calls(text: str) -> str:
    """CLI 에이전트가 새어 보낸 도구 호출 줄(update_topic(strategic_intent='...') 등)을 제거한다.

    `이름(인자='...')` 형태로 한 줄을 통째로 차지하는 도구 호출만 제거하고,
    한국어 보고서 본문은 그대로 보존한다(본문은 소문자 식별자+괄호로 시작하지 않음).
    """
    cleaned = re.sub(
        r"(?m)^[ \t]*[a-z][a-z0-9_]*\([a-z_]+\s*=\s*['\"].*\)[ \t]*-*[ \t]*$\n?",
        "",
        text,
    )
    return cleaned.strip()


def _dedupe_phase1_sections(phase1_md: str) -> str:
    """Phase 1 LLM 출력에서 반복된 구성요소 섹션을 제거한다 (첫 번째 등장만 유지)."""
    sections = re.split(r'\n(?=###\s)', "\n" + phase1_md.strip())
    seen_labels: set = set()
    seen_summary: bool = False
    kept: list = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        header = sec.split('\n', 1)[0].strip()
        m = re.match(
            r'###\s+\[?(?:구성요소|추가\s*구성)\s+\(([A-J](?:-\d+)?)\)\]?',
            header,
        )
        if m:
            label = m.group(1)
            if label not in seen_labels:
                seen_labels.add(label)
                kept.append(sec)
        elif '종합 분석 요약' in header:
            if not seen_summary:
                seen_summary = True
                kept.append(sec)
        else:
            kept.append(sec)
    return "\n\n".join(kept)


# Phase 1 유사도 라인 파싱용 — 판정 라벨(긴 것 우선) 및 장식 제거
_SIM_LABELS = r'(?:실질적 동일|일부 유사|일부 차이|대응 없음|동일|차이)'
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"
    "\U00002600-\U000027BF"
    "️‍"
    "]"
)


def _extract_similarity(sec: str) -> tuple:
    """Phase 1 구성요소 섹션에서 판정 라벨과 퍼센트를 추출한다.

    유사도는 구성요소 헤더 줄(`### [구성요소 (A)] 유사도: 동일 85%`)에 표기하는 것이 기본이나,
    구버전처럼 `- 유사도:` 불릿으로 나오는 경우도 함께 처리한다.
    LLM이 `**<span style="color:green">동일</span>** 🟠 91%이상` 처럼 마크다운/HTML/이모지로
    장식해 출력해도 견고하게 판정과 퍼센트를 분리한다.
    퍼센트는 실제 숫자만 인정한다('X%' 같은 미치환 플레이스홀더는 무시).
    """
    # '유사도:' 라벨(헤더든 불릿이든)이 있으면 그 뒤를, 없으면 헤더의 (X)] 뒤 텍스트를 본다.
    m = re.search(r'유사도\s*\*{0,2}\s*[:：]\s*([^\n]+)', sec)
    if m:
        raw = m.group(1)
    else:
        header = sec.split('\n', 1)[0]
        hm = re.search(r'\([A-Ja-j](?:-\d+)?\)\]?\s*([^\n]+)', header)
        raw = hm.group(1) if hm else ""

    cleaned = re.sub(r'<[^>]+>', '', raw)   # HTML 태그 제거
    cleaned = cleaned.replace('*', '')       # 마크다운 볼드 제거
    cleaned = _EMOJI_RE.sub('', cleaned)     # 이모지 제거
    cleaned = cleaned.replace('[', '').replace(']', '')

    jm = re.search(_SIM_LABELS, cleaned)
    judgment = jm.group(0).strip() if jm else ""

    pm = re.search(r'\d+\s*%\s*(?:이상|이하|내외|정도)?', cleaned)
    percent = re.sub(r'\s+', '', pm.group(0)) if pm else ""
    return judgment, percent


async def generate_independent_phase2(
    phase1_md: str,
    claim: ParsedClaim,
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    chain_info: Optional[Dict],
    settings: Settings,
) -> str:
    """Phase 2만 생성 (분리 호출용). Phase 1 결과를 받아 최종 보고서 작성."""
    # 결합(Template B) 여부는 인용 체인이 결정한다(total 길이). 체인이 없을 때만
    # 구성요소 귀속(any idx>0)으로 추정한다. — 주인용발명이 doc 0이 아니어도 정확.
    if chain_info:
        needs_combination = len(chain_info.get("total", [])) > 1
    else:
        needs_combination = any(m.cited_invention_index > 0 for m in matches)

    if needs_combination:
        return await _generate_template_b_phase2(phase1_md, claim, matches, prior_docs, chain_info, settings)
    else:
        return await _generate_template_a_phase2(phase1_md, claim, chain_info, settings)


async def generate_independent_phase1_streaming(
    claim: ParsedClaim,
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    chain_info: Optional[Dict],
    settings: Settings,
    prev_context: Optional[List[Dict]] = None,
    secondary_matches: Optional[List[ElementMatch]] = None,
):
    """Phase 1을 CLI stdout에서 실시간으로 읽어 청크 단위로 yield.
    특수 케이스(부정불가/신규성)는 전체 문자열을 한 번에 yield.
    """
    all_no_match = all(m.judgment in NO_MATCH_LABELS for m in matches)
    if all_no_match:
        result = await _generate_rejection_impossible_report(claim, matches, prior_docs, settings)
        yield result
        return

    all_identical = all(m.judgment == "동일" for m in matches)
    if all_identical:
        primary_idx = (chain_info.get("total") or [0])[0] if chain_info else 0
        full = await _generate_novelty_rejection_report(claim, matches, prior_docs, settings, primary_idx)
        idx = full.find("# [Phase 2]")
        yield full[:idx].strip() if idx >= 0 else full
        return

    # 결합(Template B) 여부는 인용 체인이 결정한다(total 길이). 체인이 없을 때만
    # 구성요소 귀속(any idx>0)으로 추정한다. — 주인용발명이 doc 0이 아니어도 정확.
    if chain_info:
        needs_combination = len(chain_info.get("total", [])) > 1
    else:
        needs_combination = any(m.cited_invention_index > 0 for m in matches)

    if needs_combination:
        system = _build_system(settings, "combo")
        prompt = _make_phase1_b_prompt(claim, matches, prior_docs, chain_info, settings, prev_context,
                                       secondary_matches=secondary_matches)
    else:
        system = _build_system(settings, "independent")
        prompt = _make_phase1_a_prompt(claim, matches, prior_docs, chain_info, settings, prev_context)

    async for chunk in call_ai_streaming(prompt, system, settings, agent="report"):
        yield chunk


def _make_phase1_prompt(
    claim: ParsedClaim,
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    chain_info: Optional[Dict],
    settings: Settings,
    prev_context: Optional[List[Dict]] = None,
    combo: bool = False,
    secondary_matches: Optional[List[ElementMatch]] = None,
) -> str:
    doc_name_mapping = chain_info.get("doc_name_mapping") if chain_info else None
    total_invs = chain_info.get("total", [0, 1]) if chain_info else [0, 1]
    primary_idx = total_invs[0]

    inv1_name = doc_name_mapping.get(str(primary_idx), "인용발명 1") if doc_name_mapping else "인용발명 1"
    inv1_doc = prior_docs[primary_idx] if primary_idx < len(prior_docs) else (prior_docs[0] if prior_docs else None)
    inv1_filename = inv1_doc.filename if inv1_doc else "인용발명 1"

    inv2_block = ""
    if combo:
        weak_judgments = {"일부 차이", "일부 유사", "차이", "대응 없음"}
        primary_by_label = {m.label: m for m in matches if m.cited_invention_index == primary_idx}
        conventional_support = (chain_info or {}).get("conventional_support") or {}
        evidence_blocks = []

        for support_idx in total_invs[1:]:
            support_doc = prior_docs[support_idx] if support_idx < len(prior_docs) else None
            support_name = (
                doc_name_mapping.get(str(support_idx), f"인용발명 {support_idx + 1}")
                if doc_name_mapping else f"인용발명 {support_idx + 1}"
            )
            support_filename = support_doc.filename if support_doc else support_name
            support_elements = [m for m in matches if m.cited_invention_index == support_idx]
            target_conventional_labels = (
                set(conventional_support.get("labels", []))
                if conventional_support.get("doc_idx") == support_idx else set()
            )

            if secondary_matches:
                existing_labels = {m.label for m in support_elements}
                for sm in secondary_matches:
                    if sm.cited_invention_index != support_idx:
                        continue
                    if sm.label in existing_labels or not sm.quote or sm.judgment in NO_MATCH_LABELS:
                        continue
                    pm = primary_by_label.get(sm.label)
                    if sm.label in target_conventional_labels or (
                        pm is not None and pm.judgment in weak_judgments
                    ):
                        support_elements.append(sm)
                        existing_labels.add(sm.label)
                support_elements.sort(key=lambda match: match.label)

            if not support_elements:
                continue
            role_suffix = ""
            role_notice = ""
            if conventional_support.get("doc_idx") == support_idx:
                role_suffix = " - 주지관용 구성 입증자료"
                label_text = ", ".join(f"({label})" for label in sorted(target_conventional_labels))
                role_notice = (
                    f"\n역할 제한: 구성요소 {label_text or '(표시된 일반 구성)'}의 통상적 채용을 "
                    "뒷받침하는 용도로만 사용하고, 핵심 기술사상 보완 근거로 확대하지 않습니다."
                )
            evidence_lines = []
            for match in support_elements:
                evidence_lines.append(f"구성요소 ({match.label}) [{match.judgment}]")
                evidence_lines.append(f"원문 발췌: {match.quote}")
                evidence_lines.append(f"단락/본문 위치: {_format_citation_location(match, prior_docs)}")
                evidence_lines.append("")
            evidence_body = "\n".join(evidence_lines).strip()
            evidence_blocks.append(
                f"[{support_name}{role_suffix}] {support_filename}\n"
                f"[{support_name} 관련 구성요소]{role_notice}\n{evidence_body}"
            )
        inv2_block = "\n\n".join(evidence_blocks)
        if inv2_block:
            inv2_block += "\n"

    elements_text = _format_elements(claim)
    comp_text = _format_component_comparison(matches, prior_docs, primary_idx=primary_idx, doc_name_mapping=doc_name_mapping)
    combination_rationale_block = _combination_rationale_prompt_block(chain_info)
    conventional_policy_block = _conventional_policy_prompt_block(chain_info)
    context_block = _build_context_block(prev_context)
    fmt = _phase1_format_text(combo=combo)
    if combo and len(total_invs) > 2:
        combo_hint = "인용발명 1과 2를 핵심 결합 문헌으로 검토하고, 인용발명 3은 표시된 주지관용 구성의 입증자료로만 제한하여 사용합니다."
    elif combo:
        combo_hint = "인용발명 1과 인용발명 2를 함께 검토하되, 각 구성요소는 인용발명 1 기준으로 우선 작성합니다."
    else:
        combo_hint = "각 구성요소는 인용발명 1 기준으로 작성합니다."

    return render_prompt(
        "prompt_phase1_main.txt",
        context_block=context_block,
        claim_number=str(claim.claim_number),
        claim_text=claim.text,
        elements_text=elements_text,
        inv1_filename=inv1_filename,
        comp_text=comp_text,
        inv2_block=inv2_block,
        combination_rationale_block=combination_rationale_block,
        conventional_policy_block=conventional_policy_block,
        fmt=fmt,
        combo_hint=combo_hint,
    )


def _parse_phase1(phase1_md: str) -> dict:
    """Phase 1 마크다운에서 Phase 2 조립에 필요한 데이터를 추출한다 (LLM 없음)."""
    components: list = []
    summary_similar = ""
    summary_diff = ""
    conclusion = ""

    # 종합 분석 요약 헤더가 ### 없이(또는 다른 헤더 레벨로) 출력되는 경우를 보정해
    # 섹션 분리(### 기준)가 항상 성립하도록 한다.
    phase1_md = re.sub(
        r'(?m)^[ \t]*#{0,6}[ \t]*(?:📊[ \t]*)?종합\s*분석\s*요약[^\n]*$',
        '### 📊 종합 분석 요약',
        phase1_md,
    )

    raw_sections = re.split(r'\n(?=###\s)', "\n" + phase1_md.strip())

    for sec in raw_sections:
        sec = sec.strip()
        if not sec:
            continue
        header_line = sec.split('\n', 1)[0].strip()

        # ── 구성요소 섹션 ──────────────────────────────────────────────────────
        m = re.match(
            r'###\s+\[?(?:구성요소|추가\s*구성)\s+\(([A-J](?:-\d+)?)\)\]?',
            header_line,
        )
        if m:
            label = m.group(1)

            judgment, percent = _extract_similarity(sec)

            # 대응 원문은 단락을 그대로 발췌하므로, 다음 '필드 라벨'이나 섹션 경계가
            # 나오기 전까지 전부 캡처한다. (발췌 내부의 '- ' 줄에서 잘리지 않도록)
            # 대응 원문 추출 — LLM이 다양한 bold/축약 형식으로 출력해도 캡처
            # 지원 형식: "- **인용발명** 대응 원문:", "- **인용발명 대응 원문:**",
            #           "- 인용발명 대응 원문:", "- **인용발명 원문:**", "- 인용발명 원문:"
            _QUOTE_PATTERNS = [
                # 표준: - **인용발명** 대응 원문: / - 인용발명 대응 원문:
                r'-\s*\*{0,2}\s*인용발명\s*\*{0,2}\s*대응\s*원문\s*\*{0,2}\s*:\s*(.+?)',
                # bold 통합: - **인용발명 대응 원문:**
                r'-\s*\*{1,2}\s*인용발명\s*대응\s*원문\s*\*{1,2}\s*:\s*(.+?)',
                # 축약: - 인용발명 원문: / - **인용발명 원문:**
                r'-\s*\*{0,2}\s*인용발명\s*\*{0,2}\s*원문\s*\*{0,2}\s*:\s*(.+?)',
            ]
            _QUOTE_END = (
                r'(?='
                r'\n\s*-\s*\*{0,2}\s*(?:유사도|판단\s*이유|차이점|청구항\s*구성|인용발명\s*대응\s*부분)'
                r'|\n\s*#{1,6}\s'
                r'|\n\s*📊'
                r'|\Z'
                r')'
            )
            quote_m = None
            for _qpat in _QUOTE_PATTERNS:
                quote_m = re.search(_qpat + _QUOTE_END, sec, re.DOTALL)
                if quote_m:
                    break
            quote = quote_m.group(1).strip() if quote_m else ""

            components.append({"label": label, "judgment": judgment, "percent": percent, "quote": quote})

        # ── 종합 분석 요약 섹션 ───────────────────────────────────────────────
        elif '종합 분석 요약' in header_line:
            for _sim_pat in [
                r'-\s*유사점\s*요약\s*:\s*(.+?)(?=\n\s*-\s|\Z)',
                r'-\s*유사점\s*:\s*(.+?)(?=\n\s*-\s|\Z)',
            ]:
                sim_m = re.search(_sim_pat, sec, re.DOTALL)
                if sim_m:
                    summary_similar = sim_m.group(1).strip()
                    break

            for _diff_pat in [
                r'-\s*결합\s*논리(?:\s*및\s*차이점\s*극복)?\s*:\s*(.+?)(?=\n\s*-\s|\Z)',
                r'-\s*차이점\s*극복\s*:\s*(.+?)(?=\n\s*-\s|\Z)',
                r'-\s*차이점\s*:\s*(.+?)(?=\n\s*-\s|\Z)',
            ]:
                diff_m = re.search(_diff_pat, sec, re.DOTALL)
                if diff_m:
                    summary_diff = diff_m.group(1).strip()
                    break

            conc_m = re.search(
                r'-\s*\*{0,2}\s*결론\s*\*{0,2}\s*:\s*(.+?)(?=\n\s*-\s|\Z)',
                sec, re.DOTALL,
            )
            if conc_m:
                conclusion = conc_m.group(1).strip()

    # 결론 폴백: LLM이 `- 결론:` 라벨을 생략한 경우, Phase 1 전체에서 마지막 판단 문장을 사용
    if not conclusion:
        verdicts = re.findall(
            r'[^\n.。]*쉽게\s*발명할\s*수\s*(?:있|없)습니다[.。]?',
            phase1_md,
        )
        if verdicts:
            conclusion = verdicts[-1].strip()

    seen_labels: set = set()
    deduped: list = []
    for c in components:
        if c["label"] not in seen_labels:
            seen_labels.add(c["label"])
            deduped.append(c)

    return {
        "components": deduped,
        "summary_similar": summary_similar,
        "summary_diff": summary_diff,
        "conclusion": conclusion,
    }


def _build_phase2_markdown(
    phase1_md: str,
    claim_number: int,
    inv1_name: str,
    inv2_name: str = "",
    inv3_name: str = "",
    is_combo: bool = False,
    combination_rationale: str = "",
    components_override: str = "",
    settings: Optional[Settings] = None,
) -> str:
    """Phase 1 파싱 결과를 Phase 2 양식에 채워 조립한다 (LLM 호출 없음).

    반환값은 본문만 포함하며, 경계선("# [Phase 2] 제목")은 호출부(analyze.py)가 붙인다.
    양식은 backend/prompts 파일을 우선 사용하고, 파일이 없으면 내장 기본값을 사용한다.
    양식의 자리표시자를 Python이 치환한다:
      ${inv_header} ${components} ${similar} ${combination_rationale} ${diff} ${conclusion}
    각 구성요소 줄((A) 동일 85% + 원문)은 코드가 생성한다.
    """
    phase1_md = _strip_internal_scoring_notes(phase1_md)
    combination_rationale = _strip_internal_scoring_notes(combination_rationale)
    data = _parse_phase1(phase1_md)

    if is_combo and inv2_name:
        inv_header = f"[{inv1_name}] [{inv2_name}]"
        if inv3_name:
            inv_header += f" [{inv3_name} - 주지관용 구성 입증자료]"
    else:
        inv_header = f"[{inv1_name}]"

    if components_override.strip():
        components_block = components_override.strip()
    else:
        comp_lines: list = []
        for comp in data["components"]:
            sim = comp["judgment"]
            if comp.get("percent"):
                sim = f"{sim} {comp['percent']}".strip()
            comp_lines.append(f"({comp['label']}) {sim}".rstrip())
            if comp["quote"]:
                comp_lines.append(comp["quote"])
            comp_lines.append("")
        components_block = "\n".join(comp_lines).strip()

    similar = data["summary_similar"] or \
        "※ Phase 1 탭의 [종합 분석 요약 — 유사점 요약] 항목을 참고하여 직접 작성하십시오."
    diff = data["summary_diff"] or \
        "※ Phase 1 탭의 [종합 분석 요약 — 결합 논리 및 차이점 극복] 항목을 참고하여 직접 작성하십시오."
    conclusion = data["conclusion"] or (
        "※ Phase 1 분석 결과에서 결론 항목을 확인할 수 없습니다. "
        "Phase 1 탭의 종합 분석 요약을 참고하여 결론을 직접 작성하십시오."
    )

    # 양식: 폴더 txt → 내장 기본값
    if is_combo:
        tmpl = load_prompt("format_phase2_combo.txt", DEFAULT_PHASE2_FORMAT_COMBO)
    else:
        tmpl = load_prompt("format_phase2_single.txt", DEFAULT_PHASE2_FORMAT_SINGLE)

    if is_combo and combination_rationale and "${combination_rationale}" not in tmpl:
        diff = f"{combination_rationale}\n\n{diff}".strip()

    rendered = Template(tmpl).safe_substitute(
        inv_header=inv_header,
        components=components_block,
        similar=similar,
        combination_rationale=combination_rationale,
        diff=diff,
        conclusion=conclusion,
    )
    return _strip_internal_scoring_notes(rendered)


async def _generate_template_a_phase2(
    phase1_md: str,
    claim: ParsedClaim,
    chain_info: Optional[Dict],
    settings: Settings,
) -> str:
    """Phase 2 조립 (LLM 없음): Phase 1 파싱 → 마크다운 직접 구성"""
    doc_name_mapping = chain_info.get("doc_name_mapping") if chain_info else None
    primary_idx = chain_info.get("total", [0])[0] if chain_info else 0
    inv1_name = doc_name_mapping.get(str(primary_idx), "인용발명 1") if doc_name_mapping else "인용발명 1"
    return _build_phase2_markdown(phase1_md, claim.claim_number, inv1_name, settings=settings)


def _make_phase1_a_prompt(
    claim: ParsedClaim,
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    chain_info: Optional[Dict],
    settings: Settings,
    prev_context: Optional[List[Dict]] = None,
) -> str:
    return _make_phase1_prompt(claim, matches, prior_docs, chain_info, settings, prev_context, combo=False)


def _make_phase1_b_prompt(
    claim: ParsedClaim,
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    chain_info: Optional[Dict],
    settings: Settings,
    prev_context: Optional[List[Dict]] = None,
    secondary_matches: Optional[List[ElementMatch]] = None,
) -> str:
    return _make_phase1_prompt(claim, matches, prior_docs, chain_info, settings, prev_context,
                               combo=True, secondary_matches=secondary_matches)


async def _generate_template_b_phase2(
    phase1_md: str,
    claim: ParsedClaim,
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    chain_info: Optional[Dict],
    settings: Settings,
) -> str:
    """Phase 2 조립 (LLM 없음, Template B): Phase 1 파싱 → 마크다운 직접 구성"""
    doc_name_mapping = chain_info.get("doc_name_mapping") if chain_info else None
    total_invs = chain_info.get("total", [0, 1]) if chain_info else [0, 1]
    inv1_idx = total_invs[0]
    inv2_idx = total_invs[1] if len(total_invs) > 1 else 1
    inv3_idx = total_invs[2] if len(total_invs) > 2 else None
    inv1_name = doc_name_mapping.get(str(inv1_idx), "인용발명 1") if doc_name_mapping else "인용발명 1"
    inv2_name = doc_name_mapping.get(str(inv2_idx), "인용발명 2") if doc_name_mapping else "인용발명 2"
    inv3_name = (
        doc_name_mapping.get(str(inv3_idx), "인용발명 3")
        if doc_name_mapping and inv3_idx is not None
        else ("인용발명 3" if inv3_idx is not None else "")
    )
    combination_rationale = _combination_rationale_text(chain_info)
    components_block = _format_component_comparison(
        matches,
        prior_docs,
        primary_idx=inv1_idx,
        doc_name_mapping=doc_name_mapping,
    )
    return _build_phase2_markdown(
        phase1_md,
        claim.claim_number,
        inv1_name,
        inv2_name,
        inv3_name,
        is_combo=True,
        combination_rationale=combination_rationale,
        components_override=components_block,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# 신규성 부정 (차이점 없음)
# ---------------------------------------------------------------------------

async def _generate_novelty_rejection_report(
    claim: ParsedClaim,
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
    primary_idx: int = 0,
) -> str:
    inv1_doc = prior_docs[primary_idx] if primary_idx < len(prior_docs) else (prior_docs[0] if prior_docs else None)
    elements_text = _format_elements(claim)
    comp_text = _format_component_comparison(matches, prior_docs, primary_idx=primary_idx)

    prompt = render_prompt(
        "prompt_novelty_rejection.txt",
        claim_number=str(claim.claim_number),
        claim_text=claim.text,
        elements_text=elements_text,
        inv1_filename=inv1_doc.filename if inv1_doc else "인용발명 1",
        comp_text=comp_text,
    )
    return await call_ai(prompt, _build_system(settings, "independent"), settings, agent="report")


# ---------------------------------------------------------------------------
# 진보성 부정 불가
# ---------------------------------------------------------------------------

async def _generate_rejection_impossible_report(
    claim: ParsedClaim,
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
) -> str:
    claim_type = "independent" if claim.claim_type == "independent" else "dependent"
    parent_prefix = f"제{claim.parent_claim}항에 있어서 " if claim.parent_claim else ""
    evidence_lines = []
    for match in matches:
        doc_name = f"인용발명 {match.cited_invention_index + 1}"
        evidence_lines.append(f"({match.label}) {match.judgment} ({doc_name})")
        if match.quote:
            evidence_lines.append(f"원문 발췌: {match.quote}")
            evidence_lines.append(f"출처: {_format_citation_location(match, prior_docs)}")
        if match.similarity_reason:
            evidence_lines.append(f"판단 이유: {match.similarity_reason}")
        evidence_lines.append("")

    prompt = render_prompt(
        "prompt_rejection_impossible.txt",
        claim_number=str(claim.claim_number),
        claim_text=claim.text,
        parent_prefix=parent_prefix,
        elements_text=_format_elements(claim),
        comp_text="\n".join(evidence_lines).strip() or "(확인된 대응 근거 없음)",
    )
    result = await call_ai(prompt, _build_system(settings, claim_type), settings, agent="report")
    return result


# ---------------------------------------------------------------------------
# 종속항 보고서
# ---------------------------------------------------------------------------

# 보조 문헌 발췌를 함께 제공할 "약한 판정" — 독립항 소프트 보강과 동일 기준
_WEAK_JUDGMENTS = {"일부 차이", "일부 유사", "차이"}


def _dependent_quote_lines(
    matches: List[ElementMatch],
    secondary_matches: Optional[List[ElementMatch]],
    mapping: Optional[Dict],
) -> str:
    """종속항 프롬프트용 [대응 구성 데이터] 발췌 라인을 생성한다.

    구성요소별 최선 매치에 더해, 최선 판정이 '일부 차이' 이하로 약할 때는
    체인 보조 문헌의 발췌를 함께 제공한다. 최선 선정이 동점에서 주인용발명으로
    귀속되면 보조 문헌의 명시 개시가 프롬프트에 도달하지 못해, 문헌 근거가
    있는데도 자명성 논거로 빠지는 문제를 막는다.
    """
    relevant = [m for m in matches if m.judgment not in NO_MATCH_LABELS and m.quote]
    best_by_label = {m.label: m for m in relevant}
    seen = {(m.label, m.cited_invention_index) for m in relevant}
    for sm in secondary_matches or []:
        if not sm.quote or sm.judgment in NO_MATCH_LABELS:
            continue
        if (sm.label, sm.cited_invention_index) in seen:
            continue
        best = best_by_label.get(sm.label)
        if best is None or best.judgment in _WEAK_JUDGMENTS:
            relevant.append(sm)
            seen.add((sm.label, sm.cited_invention_index))
    relevant.sort(key=lambda m: m.label)
    return "\n".join(
        f"({m.label}) [{format_inv_list([m.cited_invention_index], mapping)} {m.chunk_id}]: {m.quote}"
        for m in relevant
    )


def _dependent_parent_context_status(
    claim: ParsedClaim,
    chain_info: Optional[Dict],
) -> tuple[bool, str]:
    parent_available = bool((chain_info or {}).get("parent_available", True))
    parent_num = claim.parent_claim
    if parent_available:
        return True, f"부모 청구항: 제{parent_num or 1}항 (기존 인용 체인 상속)"
    return False, (
        f"부모 청구항 제{parent_num}항은 입력 또는 이전 컨텍스트에서 확인되지 않음. "
        "부모항의 실체나 거절이유를 추정하지 말고, '에 있어서' 뒤의 추가 기술 특징 자체만 대비"
    )


def _extract_dependent_conclusion(phase1_md: str) -> str:
    """Extract the explicit Phase 1 conclusion without inventing a verdict."""
    match = re.search(
        r"(?ms)^\s*-\s*\*{0,2}결론\*{0,2}\s*:\s*(.+?)"
        r"(?=\n\s*-\s|\n\s*#{1,6}\s|\Z)",
        phase1_md or "",
    )
    return match.group(1).strip() if match else ""


async def generate_dependent_report(
    claim: ParsedClaim,
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    chain_info: Optional[Dict],
    settings: Settings,
    prev_context: Optional[List[Dict]] = None,
    secondary_matches: Optional[List[ElementMatch]] = None,
) -> str:
    # 부정불가 자동 전환
    all_no_match = all(m.judgment in NO_MATCH_LABELS for m in matches)
    coverage_incomplete = bool(
        chain_info and chain_info.get("coverage_complete") is False
    )
    if all_no_match or coverage_incomplete:
        return await _generate_rejection_impossible_report(claim, matches, prior_docs, settings)

    parent_num = claim.parent_claim or 1
    inherited_invs = chain_info.get("inherited", [0]) if chain_info else [0]
    added_invs = chain_info.get("added", []) if chain_info else []
    total_invs = chain_info.get("total", inherited_invs) if chain_info else inherited_invs
    mapping = chain_info.get("doc_name_mapping") if chain_info else None
    parent_available, parent_context_status = _dependent_parent_context_status(claim, chain_info)

    inherited_str = format_inv_list(inherited_invs, mapping) if inherited_invs else ("인용발명 1" if parent_available else "없음 (부모항 미제공)")
    # added가 비어 있으면 상속 문헌이 추가 구성까지 커버한다는 뜻 — 담당도 상속 문헌
    current_inv_str = format_inv_list(added_invs, mapping) if added_invs else (inherited_str if parent_available else (format_inv_list(total_invs, mapping) if total_invs else "대응 문헌 없음"))
    final_str = format_inv_list(total_invs, mapping) if total_invs else ("인용발명 1" if parent_available else "대응 문헌 없음")
    added_inv_str = format_inv_list(added_invs, mapping) if added_invs else "없음"
    coverage_status = "모든 추가 구성 대응 확인"

    # 현재 종속항에서 새로 추가된 인용발명의 대응 내용
    added_doc = None
    if added_invs and added_invs[0] < len(prior_docs):
        added_doc = prior_docs[added_invs[0]]

    # 해당 구성요소 인용 — 발췌마다 출처 인용발명을 병기 (체인 문헌이 여럿일 수 있음)
    quotes_text = _dependent_quote_lines(matches, secondary_matches, mapping)

    phase1_dep_fmt = load_prompt("format_phase1_dependent.txt", DEFAULT_PHASE1_FORMAT_DEPENDENT)

    context_block = _build_context_block(prev_context)
    prompt = render_prompt(
        "prompt_dependent_report.txt",
        context_block=context_block,
        claim_number=str(claim.claim_number),
        claim_text=claim.text,
        parent_num=str(parent_num),
        parent_context_status=parent_context_status,
        inherited_str=inherited_str,
        current_inv_str=current_inv_str,
        added_inv_str=added_inv_str,
        final_str=final_str,
        coverage_status=coverage_status,
        added_doc_filename=added_doc.filename if added_doc else current_inv_str,
        quotes_text=quotes_text if quotes_text else "(대응 구성 확인 필요 — 인용발명 원문 기반으로 직접 판단)",
        phase1_dep_fmt=phase1_dep_fmt,
    )
    return await call_ai(prompt, _build_system(settings, "dependent"), settings, agent="report")


def generate_dependent_phase2(
    phase1_md: str,
    claim: ParsedClaim,
    chain_info: Optional[Dict],
    settings: Settings,
) -> str:
    """종속항 Phase 2 조립 (LLM 없음): Phase 1 분석문과 체인 정보를 템플릿에 치환한다."""
    parent_num = claim.parent_claim or 1
    inherited_invs = chain_info.get("inherited", [0]) if chain_info else [0]
    added_invs = chain_info.get("added", []) if chain_info else []
    total_invs = chain_info.get("total", inherited_invs) if chain_info else inherited_invs
    mapping = chain_info.get("doc_name_mapping") if chain_info else None
    parent_available, parent_context_status = _dependent_parent_context_status(claim, chain_info)

    inherited_str = format_inv_list(inherited_invs, mapping) if inherited_invs else ("인용발명 1" if parent_available else "없음 (부모항 미제공)")
    current_inv_str = format_inv_list(added_invs, mapping) if added_invs else (inherited_str if parent_available else (format_inv_list(total_invs, mapping) if total_invs else "대응 문헌 없음"))
    final_str = format_inv_list(total_invs, mapping) if total_invs else current_inv_str

    if parent_available:
        review_intro = (
            f"제{parent_num}항에 있어서, 청구항 {claim.claim_number}의 추가 구성은 "
            f"아래와 같이 {current_inv_str}에 의해 검토됩니다."
        )
        parent_basis = (
            f"부모 청구항의 기존 대비에서 사용된 인용발명 조합은 {inherited_str}이며, "
            f"청구항 {claim.claim_number}의 추가 구성까지 고려한 검토 조합은 {final_str}입니다."
        )
    else:
        review_intro = (
            f"청구항 {claim.claim_number}이 인용하는 제{parent_num}항은 입력 또는 이전 컨텍스트에서 "
            f"확인되지 않아, '에 있어서' 뒤의 추가 기술 특징을 {current_inv_str}과 직접 대비합니다."
        )
        parent_basis = (
            "확인되지 않은 부모 청구항의 실체나 거절이유는 추정하지 않고, "
            f"청구항 {claim.claim_number}에 직접 기재된 추가 기술 특징만을 판단 대상으로 삼았습니다."
        )
    analysis = re.sub(
        rf"(?m)^\s*###\s+청구항\s+{claim.claim_number}\s*$\n*",
        "",
        phase1_md.strip(),
    ).strip()
    if not analysis:
        analysis = "※ Phase 1 추가 구성 대비 분석 결과를 확인할 수 없습니다."

    coverage_incomplete = bool(
        chain_info and chain_info.get("coverage_complete") is False
    )
    phase1_conclusion = _extract_dependent_conclusion(analysis)
    if not parent_available:
        conclusion = (
            f"청구항 {claim.claim_number}의 추가 기술 특징에 대한 대비 결과는 위와 같으나, "
            f"인용하는 제{parent_num}항이 확인되지 않아 청구항 전체의 거절 근거 구성 가능 여부는 "
            "현재 정보만으로 판단할 수 없습니다."
        )
    elif coverage_incomplete:
        uncovered = ", ".join(chain_info.get("uncovered_labels", [])) or "일부 추가 구성"
        conclusion = (
            f"청구항 {claim.claim_number}의 {uncovered}에 대해서는 부모 체인에 새 인용발명 "
            "1개만 추가하는 범위에서 대응 근거가 완성되지 않으므로, 위 인용발명 조합만으로 "
            "통상의 기술자가 쉽게 발명할 수 있다고 보기 어렵습니다."
        )
    elif phase1_conclusion:
        conclusion = phase1_conclusion
    else:
        conclusion = (
            f"청구항 {claim.claim_number}의 최종 판단은 위 추가 구성 대비 결과와 부모 청구항의 "
            f"판단을 함께 고려하여 검토할 필요가 있습니다. 검토 대상 인용발명 조합은 {final_str}입니다."
        )

    tmpl = load_prompt("format_phase2_dependent.txt", DEFAULT_PHASE2_FORMAT_DEPENDENT)

    return Template(tmpl).safe_substitute(
        claim_number=str(claim.claim_number),
        parent_num=str(parent_num),
        parent_context_status=parent_context_status,
        inherited_str=inherited_str,
        current_inv_str=current_inv_str,
        final_str=final_str,
        review_intro=review_intro,
        parent_basis=parent_basis,
        analysis=analysis,
        conclusion=conclusion,
    )


async def generate_dependent_reports_batch(
    claims_data: List[tuple],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
    prev_context: Optional[List[Dict]] = None,
) -> str:
    """여러 종속항을 한 번의 LLM 호출로 처리한다.

    claims_data: (claim, matches, chain_info, secondary_matches) 튜플 리스트.
    `===청구항 N===` 구분선으로 종속항별 보고서가 이어진 원시 통합 출력을 반환한다(분리는 라우터 담당).
    종속항은 독립항보다 단순해 한 호출에 묶어도 품질 손실이 작고, 시스템/양식 오버헤드를 1회로 줄인다.
    """
    blocks = []
    for claim, matches, chain_info, secondary_matches in claims_data:
        parent_num = claim.parent_claim or 1
        inherited_invs = chain_info.get("inherited", [0]) if chain_info else [0]
        added_invs = chain_info.get("added", []) if chain_info else []
        total_invs = chain_info.get("total", inherited_invs) if chain_info else inherited_invs
        mapping = chain_info.get("doc_name_mapping") if chain_info else None
        parent_available, parent_context_status = _dependent_parent_context_status(claim, chain_info)

        inherited_str = format_inv_list(inherited_invs, mapping) if inherited_invs else ("인용발명 1" if parent_available else "없음 (부모항 미제공)")
        # added가 비어 있으면 상속 문헌이 추가 구성까지 커버한다는 뜻 — 담당도 상속 문헌
        current_inv_str = format_inv_list(added_invs, mapping) if added_invs else (inherited_str if parent_available else (format_inv_list(total_invs, mapping) if total_invs else "대응 문헌 없음"))
        final_str = format_inv_list(total_invs, mapping) if total_invs else ("인용발명 1" if parent_available else "대응 문헌 없음")
        added_inv_str = format_inv_list(added_invs, mapping) if added_invs else "없음"
        uncovered_labels = (chain_info or {}).get("uncovered_labels", [])
        if (chain_info or {}).get("coverage_complete") is False:
            coverage_status = (
                "대응 불충분 — 새 인용발명 1개만으로 커버되지 않은 추가 구성: "
                + (", ".join(uncovered_labels) or "확인 필요")
            )
        else:
            coverage_status = "모든 추가 구성 대응 확인"

        added_doc = None
        if added_invs and added_invs[0] < len(prior_docs):
            added_doc = prior_docs[added_invs[0]]

        quotes_text = _dependent_quote_lines(matches, secondary_matches, mapping)

        blocks.append(
            f"===청구항 {claim.claim_number}===\n"
            f"[청구항 {claim.claim_number} 원문]\n{claim.text}\n\n"
            f"[인용 체인 정보]\n"
            f"- {parent_context_status}\n"
            f"- 이 종속항에서 새로 추가된 인용발명: {added_inv_str} (최대 1개)\n"
            f"- 이 종속항 담당 인용발명: {current_inv_str} "
            f"({added_doc.filename if added_doc else current_inv_str})\n"
            f"- 최종 결합 발명: {final_str}\n"
            f"- 단일 추가 문헌 커버 상태: {coverage_status}\n\n"
            f"[담당 인용발명 대응 구성 데이터]\n"
            f"{quotes_text if quotes_text else '(대응 구성 확인 필요 — 인용발명 원문 기반으로 직접 판단)'}"
        )

    phase1_dep_fmt = load_prompt("format_phase1_dependent.txt", DEFAULT_PHASE1_FORMAT_DEPENDENT)

    context_block = _build_context_block(prev_context)
    prompt = render_prompt(
        "prompt_dependent_report_batch.txt",
        context_block=context_block,
        claim_blocks="\n\n".join(blocks),
        phase1_dep_fmt=phase1_dep_fmt,
    )
    return await call_ai(prompt, _build_system(settings, "dependent"), settings, agent="report")


# ---------------------------------------------------------------------------
# 카테고리 동일 청구항
# ---------------------------------------------------------------------------

def generate_category_same_report(
    original_claim_num: int,
    same_claim_nums: List[int],
    original_report: str,
) -> str:
    suffix_lines = [original_report, "\n---\n"]
    for n in same_claim_nums:
        suffix_lines.append(
            f"청구항 {n} 발명은 청구항 {original_claim_num} 발명과 카테고리만 상이할 뿐 "
            f"실질적으로 동일한 발명으로 동일한 구성대비 판단 근거가 적용됩니다."
        )
    return "\n".join(suffix_lines)


def build_rejected_inventions_section(
    claim: ParsedClaim,
    prior_docs: List[ExtractedDocument],
    chain_info: Optional[Dict],
    job_dir: str,
) -> str:
    """채택되지 않은(탈락) 인용발명의 청구항 대비 결과를 보고서 말미 섹션으로 조립한다.

    chain_info.total(보고서에 실제 사용된 인용발명)에 없는 인용발명을 대상으로,
    comparisons_{idx}.json 의 독립 판정을 읽어 '대비했으나 채택 안 됨'을 명시한다.
    per_doc 대비에서 각 인용발명이 독립 판정될 때 의미가 있다(LLM 호출 없음).
    """
    from backend.services.citation_extractor import load_comparisons

    if not chain_info or len(prior_docs) <= 1:
        return ""

    used = set(chain_info.get("total", []))
    doc_name_mapping = chain_info.get("doc_name_mapping", {})
    # 자기 청구항 키 우선, 없으면(구버전 작업) 부모 독립항 키 폴백.
    claim_key = str(claim.claim_number)
    parent_key = (
        str(claim.parent_claim)
        if claim.claim_type == "dependent" and claim.parent_claim
        else None
    )

    blocks = []
    for doc_idx in range(len(prior_docs)):
        if doc_idx in used:
            continue
        cache = load_comparisons(job_dir, doc_idx)
        items = None
        if cache:
            items = cache.get(claim_key) or (cache.get(parent_key) if parent_key else None)
        if not items:
            continue  # 이 인용발명을 해당 청구항과 대비한 기록이 없음 → 생략
        inv_name = doc_name_mapping.get(str(doc_idx), f"인용발명 {doc_idx + 1}")
        judged = " · ".join(
            f"({it.get('label', '')}) {it.get('judgment', '대응 없음')}" for it in items
        )
        has_corr = any(it.get("judgment") not in NO_MATCH_LABELS for it in items)
        note = (
            "주인용발명이 개시하지 못한 구성요소를 보완하지 못해 채택되지 않았습니다."
            if has_corr
            else "청구항의 어떤 구성요소도 개시하지 않아 채택되지 않았습니다."
        )
        blocks.append(f"**{inv_name}** ({prior_docs[doc_idx].filename})\n{judged}\n→ {note}")

    if not blocks:
        return ""
    return (
        "## 그 외 검토한 인용발명\n\n"
        "아래 인용발명도 청구항 구성요소와 대비하였으나 주된 거절근거로 채택되지 않았습니다.\n\n"
        + "\n\n".join(blocks)
    )


async def parse_manual_claim_locally(
    claim_text: str,
    claim_number: int,
    claim_type: str,
    parent_claim: Optional[int],
) -> ParsedClaim:
    """사용자가 붙여넣은 청구항 1개를 LLM 없이 구성요소로 분해한다."""
    clean_text = claim_text.strip()
    inferred_parent = parent_claim
    if inferred_parent is None:
        # 1차: 어미 패턴 정확 매칭 ("제N항에 있어서" / "제N항의")
        m = re.search(r"제\s*(\d+)\s*항(?:에\s*있어서|의)", clean_text)
        if m:
            candidate = int(m.group(1))
            if candidate != claim_number:
                inferred_parent = candidate

    if inferred_parent is None:
        # 2차: 어두 오타·생략 대비 — 앞부분 150자 안에 "제N항" + N < 현재 번호
        # finditer로 전체 후보를 보는 이유: "제10항. 제1항에 있이서…"처럼 자기 번호가
        # 맨 앞에 오면 re.search 첫 결과가 자기 자신(N == 현재)이라 조건에 걸린다.
        for m2 in re.finditer(r"제\s*(\d+)\s*항", clean_text[:150]):
            candidate = int(m2.group(1))
            if candidate < claim_number:
                inferred_parent = candidate
                break

    if inferred_parent is None:
        # 3차: 후미형 종속항 — 어두가 독립항처럼 생겼지만 본문/후미에
        # "제X항 [내지/또는/및 제Y항]을 포함하는/인용하는/에 따른" 형태로 참조하는 경우.
        # 독립항이 다른 청구항을 단순 언급하는 것과 구분하기 위해 의존 동사구를 필수로 요구한다.
        m3 = re.search(
            r"제\s*(\d+)\s*항"
            r"(?:\s*(?:내지|또는|및)\s*제\s*\d+\s*항)?"
            r"\s*(?:(?:을|를)\s*(?:포함|인용|청구|참조|기재)\s*(?:하는|하여|하고)?|에\s*따른|에\s*의한)",
            clean_text,
        )
        if m3:
            candidate = int(m3.group(1))
            if candidate < claim_number:
                inferred_parent = candidate

    resolved_type = claim_type
    if inferred_parent and claim_type == "independent":
        resolved_type = "dependent"

    return _enhanced_parse_manual_claim(clean_text, claim_number, resolved_type, inferred_parent)


def _enhanced_parse_manual_claim(
    claim_text: str,
    claim_number: int,
    claim_type: str,
    parent_claim: Optional[int],
) -> ParsedClaim:
    """개선된 정규식 기반 청구항 파싱.
    어두/어미 추출 → 세미콜론/줄바꿈 분리 → 서브구성(A-1) 감지."""
    LABELS = "ABCDEFGHIJ"
    text = claim_text.strip()

    # 1. 어두(preamble) 추출: "~~에 있어서," 패턴
    preamble: Optional[str] = None
    preamble_end = 0
    m_pre = re.search(r'^(.*?에\s*있어서)\s*[,，、]\s*', text, re.DOTALL)
    if m_pre:
        preamble = m_pre.group(1).strip()
        preamble_end = m_pre.end()

    # 2. 어미(closing) 추출: "특징으로 하는/포함하는 [장치/방법/...]" 패턴
    closing: Optional[str] = None
    closing_start = len(text)
    _CLOSING_RE = re.compile(
        r'(?:^|[\n,])\s*(?=(?:을|를)\s*포함(?:하는|하며|하고)?\s*'
        r'(?:장치|방법|시스템|프로그램|단말|서버|기기|컴퓨터|기록\s*매체|네트워크|데이터베이스)'
        r'|특징으로\s*하는\s*(?:장치|방법|시스템|프로그램|단말|서버|기기|컴퓨터|기록\s*매체))',
        re.DOTALL,
    )
    m_cl = _CLOSING_RE.search(text, preamble_end)
    if m_cl:
        candidate = text[m_cl.start():].strip().lstrip(',').strip()
        # 어미가 텍스트 뒤쪽 1/3 이내에 있는지 확인 (너무 앞이면 무시)
        if m_cl.start() > preamble_end + max(10, (len(text) - preamble_end) // 3):
            closing = candidate
            closing_start = m_cl.start()

    # 3. 중간(body) 추출
    body = text[preamble_end:closing_start].strip().strip(',').strip()
    if not body:
        body = text

    # 4a-0. 후행 라벨: "...입력받는 (a)단계", "...제거하는 (c) 단계 및"처럼 라벨이
    #        분절 '끝'에 오는 한국식 청구항. "상기 (a) 단계에서 ..." 같은 이전 단계
    #        참조는 라벨로 보지 않는다(뒤에 '에서'가 오면 제외).
    trailing = list(re.finditer(r'\(\s*([A-Ja-j])\s*\)\s*단계(?!\s*에서)(?:\s*및)?', body))
    if len(trailing) >= 2:
        elements = []
        prev_end = 0
        for i, mk in enumerate(trailing[:10]):
            # 분절 안의 '(x)단계' 토큰(마커·참조 모두)은 '단계'로 정리하고, 분절 경계에
            # 남는 연결어 '및'은 앞뒤 모두 떼어낸다.
            seg = re.sub(r'\(\s*[A-Ja-j]\s*\)\s*단계', '단계', body[prev_end:mk.end()])
            seg = re.sub(r'^\s*및\s*|\s*및\s*$', '', seg.strip().strip(',').strip()).strip()
            prev_end = mk.end()
            if not seg:
                continue
            imp = "5" if i == 0 else ("3" if i < 3 else "2")
            elements.append(ClaimElement(label=LABELS[i], text=seg, importance=imp))
        if len(elements) >= 2:
            return ParsedClaim(
                claim_number=claim_number, claim_type=claim_type,
                parent_claim=parent_claim, text=text, elements=elements,
                preamble=preamble, closing=closing, split_method="trailing_labeled",
            )

    # 4a. 기존 (A)/(B) 명시적 라벨이 있는 경우 — 우선 처리
    labeled = re.findall(
        r'(?:^|\n|\s)[\(\[]([A-Ja-j])[\)\]]\s*(.*?)(?=(?:\n|\s)[\(\[][A-Ja-j][\)\]]|\Z)',
        body, flags=re.DOTALL,
    )
    if labeled:
        elements = []
        for i, (lbl, content) in enumerate(labeled[:10]):
            content = content.strip(' \n;')
            if not content:
                continue
            imp = "5" if i == 0 else ("3" if i < 3 else "2")
            elements.append(ClaimElement(label=lbl.upper(), text=content, importance=imp))
        if elements:
            return ParsedClaim(
                claim_number=claim_number, claim_type=claim_type,
                parent_claim=parent_claim, text=text, elements=elements,
                preamble=preamble, closing=closing, split_method="labeled",
            )

    # 4b. 세미콜론 분리
    parts = [p.strip() for p in re.split(r'[;；]', body) if p.strip()]

    # 4c. 줄바꿈 분리 (세미콜론 없을 때)
    if len(parts) <= 1:
        parts = [p.strip() for p in body.split('\n') if p.strip() and len(p.strip()) > 3]

    split_method = "regex"
    if len(parts) <= 1:
        # 4d. fallback — 단일 텍스트 블록 (LLM 강화 대상)
        parts = [body]
        split_method = "fallback"

    # 5. 구성요소 레이블 + 서브구성(A-1) 감지
    elements: List[ClaimElement] = []
    label_idx = 0
    component_names: Dict[str, str] = {}  # label → 핵심 명사

    for part in parts[:10]:
        if not part:
            continue
        clean = part.rstrip('; ')

        # 단어 하나짜리 예외 (프로세서, 메모리 등)
        if _is_single_word_component(clean):
            elements.append(ClaimElement(label="_", text=clean, importance="2"))
            continue

        # 서브구성 감지: "상기 [이전 구성 이름]" 패턴
        sub_of = _find_sub_component(clean, component_names)
        if sub_of:
            elements.append(ClaimElement(
                label=f"{sub_of}-1", text=clean, importance="3",
                is_sub=True, parent_label=sub_of,
            ))
            continue

        # 일반 구성요소 레이블 할당
        if label_idx >= len(LABELS):
            break
        label = LABELS[label_idx]
        name = _extract_component_name(clean)
        if name:
            component_names[label] = name
        imp = "5" if label_idx == 0 else ("3" if label_idx < 3 else "2")
        elements.append(ClaimElement(label=label, text=clean, importance=imp))
        label_idx += 1

    if not elements:
        elements = [ClaimElement(label="A", text=text, importance="3")]

    return ParsedClaim(
        claim_number=claim_number, claim_type=claim_type,
        parent_claim=parent_claim, text=text, elements=elements,
        preamble=preamble, closing=closing, split_method=split_method,
    )


def _is_single_word_component(text: str) -> bool:
    """단어 하나짜리 구성(프로세서, 메모리 등) 여부 판별."""
    clean = text.strip('; ')
    if len(clean) > 15:
        return False
    # 서술어(동사/형용사) 어미가 없으면 단순 명사구
    if re.search(r'하는|하며|하고|이며|이고|되는|수행|포함|연결|처리|변환|전송|수신|저장|판단|생성|제공|구비', clean):
        return False
    return True


def _extract_component_name(text: str) -> Optional[str]:
    """구성요소 텍스트에서 핵심 명사(구성 이름) 추출."""
    # "...하는 [명사]" — 명사가 구성 이름
    m = re.search(r'(?:하는|하기\s*위한|구비된|구비한)\s+([\w가-힣]+(?:\s+[\w가-힣]+){0,2})\s*[;,]?\s*$', text.strip())
    if m:
        return m.group(1).strip()
    # "...포함하는 [명사]"
    m = re.search(r'포함하는\s+([\w가-힣]+(?:\s+[\w가-힣]+){0,2})\s*[;,]?\s*$', text.strip())
    if m:
        return m.group(1).strip()
    return None


def _find_sub_component(text: str, component_names: Dict[str, str]) -> Optional[str]:
    """'상기 [이전 구성 이름]' 패턴으로 서브구성 여부 감지."""
    for label, name in component_names.items():
        if name and re.search(rf'상기\s+{re.escape(name)}', text):
            return label
    return None


# ---------------------------------------------------------------------------
# LLM 강화 함수 (하이브리드)
# ---------------------------------------------------------------------------

_SYSTEM_ENHANCE_PURPOSE = """당신은 대한민국 특허 분석 전문가입니다.
주어진 청구항들을 읽고 발명의 목적과 효과를 간결하게 추출하세요.

출력은 반드시 아래 JSON 형식만 사용하세요:
{"purpose": "발명의 목적 (2-3문장)", "effects": "발명의 효과 및 이점 (2-3문장)"}"""


async def enhance_purpose_effects_with_llm(
    claims_text: str,
    settings: "Settings",
) -> dict:
    """독립항 텍스트로부터 LLM으로 발명의 목적/효과를 추출한다."""
    prompt = render_prompt("prompt_enhance_purpose.txt", claims_text=claims_text[:4000])

    try:
        response = await call_ai(prompt, load_prompt("system_enhance_purpose.txt", _SYSTEM_ENHANCE_PURPOSE), settings, agent="parser")
        data = _extract_first_json_object(response)
        if data:
            return {
                "purpose": data.get("purpose", ""),
                "effects": data.get("effects", ""),
                "extracted_by": "llm",
            }
    except Exception as e:
        logger.error(f"LLM purpose/effects enhance error: {e}")

    return {"purpose": "", "effects": "", "extracted_by": "llm_error"}


_SYSTEM_ENHANCE_CLAIM = """당신은 대한민국 특허 청구항 분석 전문가입니다.
주어진 청구항의 구성요소를 분해하세요.

규칙:
1. 세미콜론이나 줄바꿈 없이 연결된 경우도 의미 단위로 분해
2. (A), (B), (C)... 순서로 라벨링
3. 어두("에 있어서") → preamble 보존
4. 어미("특징으로 하는 장치/방법") → closing 보존
5. 단어 하나짜리(프로세서, 메모리 등) → label="_"
6. 이전 구성을 "상기 X"로 참조하는 구성 → "A-1", "B-1" 등 서브라벨

출력 형식 (JSON):
{
  "preamble": "어두 (없으면 null)",
  "closing": "어미 (없으면 null)",
  "elements": [
    {"label": "A", "text": "구성 내용", "importance": "5", "is_sub": false, "parent_label": null},
    {"label": "A-1", "text": "서브구성 내용", "importance": "3", "is_sub": true, "parent_label": "A"}
  ]
}"""


async def enhance_claim_parsing_with_llm(
    claim: ParsedClaim,
    settings: "Settings",
) -> ParsedClaim:
    """split_method=fallback인 청구항을 LLM으로 재파싱한다."""
    prompt = render_prompt("prompt_enhance_claim.txt", claim_number=str(claim.claim_number), claim_text=claim.text)

    try:
        response = await call_ai(prompt, load_prompt("system_enhance_claim.txt", _SYSTEM_ENHANCE_CLAIM), settings, agent="parser")
        data = _extract_first_json_object(response)
        if data:
            elements = [ClaimElement(**e) for e in data.get("elements", []) if e.get("label")]
            if elements:
                return claim.model_copy(update={
                    "elements": elements,
                    "preamble": data.get("preamble") or claim.preamble,
                    "closing": data.get("closing") or claim.closing,
                    "split_method": "llm",
                })
    except Exception as e:
        logger.error(f"LLM claim enhance error: {e}")

    return claim


# ---------------------------------------------------------------------------
# 카테고리 동일 청구항 감지
# ---------------------------------------------------------------------------

_SYSTEM_CATEGORY = """당신은 특허 청구항 비교 전문가입니다.
주어진 청구항들 중 카테고리(장치/방법/시스템)만 다르고 기술적 내용이 실질적으로 동일한 쌍을 찾으세요.

출력 형식 (JSON): {"same_pairs": {"카테고리동일청구항번호": 원본청구항번호}}
예시: {"same_pairs": {"11": 1, "12": 2}}
동일한 쌍이 없으면: {"same_pairs": {}}
반드시 JSON만 출력"""


async def detect_category_same_claims(
    claims: List[ParsedClaim],
    settings: Settings,
) -> Dict[str, int]:
    if len(claims) < 2:
        return {}

    claims_summary = "\n".join(
        f"청구항 {c.claim_number} ({c.claim_type}): {c.text[:200]}"
        for c in claims
    )

    prompt = render_prompt("prompt_category_detect.txt", claims_summary=claims_summary[:4000])

    try:
        response = await call_ai(prompt, load_prompt("system_category.txt", _SYSTEM_CATEGORY), settings, agent="category")
        data = _extract_first_json_object(response)
        if data:
            return {str(k): int(v) for k, v in data.get("same_pairs", {}).items()}
    except Exception as e:
        logger.warning(f"Category same detection error: {e}")
    return {}


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------

def _format_elements(claim: ParsedClaim) -> str:
    if not claim.elements:
        return claim.text
    return "\n".join(f"({e.label}) {e.text}" for e in claim.elements)


def _format_component_comparison(
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    primary_idx: int = 0,
    doc_name_mapping: Optional[Dict[str, str]] = None,
) -> str:
    lines = []
    for m in matches:
        # 주인용발명(primary_idx)이 아닌 인용발명에 대응된 구성요소는 이 인용발명의
        # [구성요소] 섹션에서 '대응 없음'으로 처리하고 quote/chunk_id를 비운다.
        # (결합(Template B)이면 보조인용발명 발췌는 별도 inv2_block으로 전달된다.)
        # 이전 조건(> max_inv_idx and != 0)은 주인용발명을 doc 0으로 가정했기 때문에,
        # 체인이 doc 0이 아닌 문헌을 주인용으로 선정하면 다른 문헌 발췌가 섞였다.
        is_secondary = m.cited_invention_index != primary_idx

        if is_secondary:
            judgment_display = "대응 없음"
            quote_display = ""
            chunk_display = ""
        else:
            judgment_display = m.judgment
            quote_display = m.quote
            chunk_display = m.chunk_id

        doc_name = "인용발명 1"
        if doc_name_mapping:
            doc_name = doc_name_mapping.get(str(m.cited_invention_index), f"인용발명 {m.cited_invention_index + 1}")

        if is_secondary:
            lines.append(f"({m.label}) {judgment_display} (인용발명 1에서 직접 대응 구성 미확인)")
        else:
            lines.append(f"({m.label}) {judgment_display} ({doc_name})")
        if quote_display:
            lines.append(quote_display)
        if chunk_display:
            doc = prior_docs[m.cited_invention_index] if m.cited_invention_index < len(prior_docs) else None
            if doc and doc.document_type == "non_patent":
                anchor = chunk_display.replace("[P", "").split("-")[0] if "[P" in chunk_display else "?"
                lines.append(f'(본문 {anchor} 페이지)')
            else:
                lines.append(f"(단락 {chunk_display})")
        lines.append("")
    return "\n".join(lines)

