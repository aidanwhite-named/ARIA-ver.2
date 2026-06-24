"""
gap_search.py — 보완문서 검색 (인용발명 미커버 구성요소 웹검색)

두 단계로 동작:
  1. find_uncovered_elements() — LLM 없음, 즉시 반환
     comparisons_{i}.json 전체를 읽어 어떤 인용발명도 개시하지 못한 구성요소를 찾음
  2. web_search_gap_documents() — LLM 1회 호출 (버튼 클릭 시에만)
     LLM이 웹검색 도구로 미커버 구성요소를 개시하는 보완문서 후보를 직접 탐색
"""
from __future__ import annotations

import json
import logging
import re
from typing import List

from backend.models.schemas import ParsedClaim, Settings
from backend.services.ai_engine import call_ai
from backend.services.citation_chain import _JUDGMENT_SCORE, _PRIMARY_COVER_THRESHOLD
from backend.services.citation_extractor import load_comparisons, normalize_label

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 공개 API 1: 미커버 구성요소 탐지 (LLM 없음)
# ---------------------------------------------------------------------------

def find_uncovered_elements(
    job_dir: str,
    claim: ParsedClaim,
    doc_filenames: List[str],
) -> dict:
    """전체 인용발명 비교 캐시에서, 어떤 문헌도 '일부 차이' 이상으로
    개시하지 못한 구성요소를 중요도순으로 반환한다.

    Returns:
        {
          "claim_number": int,
          "analyzed": bool,        # 비교 캐시 존재 여부
          "uncovered": [ {label, text, importance, best_judgment, best_doc} ],
          "covered":   [ {label, text, judgment, doc} ],
        }
    """
    claim_key = str(claim.claim_number)
    # 종속항은 부모 독립항 캐시를 우선 사용 (get_matches_from_cache와 동일 규칙)
    target_key = claim_key
    if claim.claim_type == "dependent" and claim.parent_claim:
        target_key = str(claim.parent_claim)

    # 구성요소 라벨 → 문헌별 (judgment, doc_idx) 수집
    per_label: dict[str, list[tuple[int, str, int]]] = {}  # label → [(score, judgment, doc_idx)]
    analyzed = False
    for doc_idx in range(len(doc_filenames)):
        cache = load_comparisons(job_dir, doc_idx)
        if not cache:
            continue
        items = cache.get(claim_key) or cache.get(target_key)
        if not isinstance(items, list):
            continue
        analyzed = True
        for item in items:
            label = normalize_label(item.get("label", ""))
            judgment = item.get("judgment", "대응 없음")
            score = _JUDGMENT_SCORE.get(judgment, 0)
            per_label.setdefault(label, []).append((score, judgment, doc_idx))

    if not analyzed:
        return {"claim_number": claim.claim_number, "analyzed": False,
                "uncovered": [], "covered": []}

    uncovered, covered = [], []
    for elem in claim.elements:
        label = normalize_label(elem.label)
        entries = per_label.get(label, [])
        if entries:
            best_score, best_judgment, best_doc = max(entries, key=lambda e: e[0])
        else:
            best_score, best_judgment, best_doc = 0, "대응 없음", -1
        best_doc_name = doc_filenames[best_doc] if 0 <= best_doc < len(doc_filenames) else ""

        row = {
            "label": elem.label,
            "text": elem.text,
            "importance": elem.importance,
            "best_judgment": best_judgment,
            "best_doc": best_doc_name,
        }
        if best_score < _PRIMARY_COVER_THRESHOLD:
            uncovered.append(row)
        else:
            covered.append(row)

    # 중요도(별점) 내림차순 — 진보성에 중요한 구성을 검색 우선 타깃으로
    def _imp(r):
        try:
            return int(r["importance"])
        except (ValueError, TypeError):
            return 3
    uncovered.sort(key=_imp, reverse=True)

    return {
        "claim_number": claim.claim_number,
        "analyzed": True,
        "uncovered": uncovered,
        "covered": covered,
    }


# ---------------------------------------------------------------------------
# 공개 API 2: LLM 웹검색으로 보완문서 후보 탐색
# ---------------------------------------------------------------------------

_WEB_SYSTEM = """당신은 특허 선행기술 조사 전문가입니다.
웹검색 도구를 사용하여, 현재 인용발명들이 개시하지 못한 청구항 구성요소를 개시하는 보완 선행문헌을 직접 찾아야 합니다.
검색을 모두 마친 뒤, 최종 답변은 JSON 형식으로만 작성하세요. 다른 설명이나 마크다운 코드블록은 사용하지 마세요."""

_WEB_PROMPT_TMPL = """청구항 {claim_number}의 진보성 부정을 위해 추가 선행문헌(보완문서)이 필요합니다.
결합 거절에 쓸 보조인용발명이므로, 아래 [검색 대상] 구성요소를 개시하는 특허문헌·논문을 웹검색으로 직접 찾아주세요.

[검색 대상 — 어떤 인용발명도 개시하지 못한 구성요소 (중요도 ★ 순)]
{uncovered_text}

[기술분야 컨텍스트]
{field_text}

검색 방법:
- 구성요소마다 핵심 기술 특징어를 뽑아 한국어·영문으로 웹검색을 수행하세요.
- 특허문헌이 우선이므로 patents.google.com 결과를 우선 활용하세요 (예: site:patents.google.com 검색).
- 중요도(★)가 높은 구성요소부터 검색하고, 구성요소당 후보 문헌을 2~3개 찾으세요.
- 후보 문헌 페이지를 열어 해당 기술 특징이 실제로 개시되어 있는지 확인하세요.

최종 답변 JSON:
{{
  "results": [
    {{
      "label": "구성요소 라벨",
      "feature_ko": "검색한 기술 특징 한 줄 요약",
      "queries_used": ["실제 사용한 검색어"],
      "documents": [
        {{
          "title": "문헌 제목",
          "number": "공개/등록번호 (특허가 아니면 출처명)",
          "url": "문헌 링크",
          "summary": "이 문헌이 해당 구성요소를 개시하는 내용 요약 (2문장 이내)",
          "relevance": "high|medium|low"
        }}
      ]
    }}
  ]
}}

규칙:
- 웹검색 결과에서 실제로 확인한 문헌만 포함하세요. URL·번호를 지어내지 마세요.
- 해당 구성요소를 개시할 가능성이 낮은 문헌은 제외하세요. 적합한 문헌이 없으면 documents를 빈 배열로 두세요.
- 반드시 유효한 JSON 객체만 반환, 설명 텍스트 없음"""


def _format_uncovered(uncovered: List[dict]) -> str:
    lines = []
    for u in uncovered:
        try:
            stars = "★" * max(1, min(5, int(u["importance"])))
        except (ValueError, TypeError):
            stars = "★★★"
        lines.append(f"- ({u['label']}) [{stars}] {u['text']}")
        lines.append(f"  현재 최고 판정: {u['best_judgment']}"
                     + (f" ({u['best_doc']})" if u['best_doc'] else ""))
    return "\n".join(lines)


async def web_search_gap_documents(
    claim: ParsedClaim,
    gap_result: dict,
    settings: Settings,
) -> dict:
    """LLM이 웹검색 도구로 미커버 구성요소를 개시하는 보완문서 후보를 직접 찾는다."""
    uncovered = gap_result.get("uncovered", [])
    if not uncovered:
        return {"claim_number": claim.claim_number, "results": []}

    # 검색 세션이 과도하게 길어지지 않도록 중요도 상위 4개만 타깃
    targets = uncovered[:4]

    field_text = claim.preamble or (claim.elements[0].text if claim.elements else claim.text[:120])

    prompt = _WEB_PROMPT_TMPL.format(
        claim_number=claim.claim_number,
        uncovered_text=_format_uncovered(targets),
        field_text=field_text,
    )

    try:
        raw = await call_ai(
            prompt=prompt,
            system=_WEB_SYSTEM,
            settings=settings,
            agent="compare",   # 판단 품질이 필요한 작업 — 구성대비 모델 재사용
            web_search=True,
        )
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            result = json.loads(raw.strip())
        except json.JSONDecodeError:
            # 웹검색 응답은 JSON 뒤에 출처 목록 등이 붙는 경우가 있어 중괄호 범위만 추출
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end <= start:
                raise
            result = json.loads(raw[start:end + 1])
        if not isinstance(result, dict):
            raise ValueError("응답이 객체 형태가 아닙니다")
        result["claim_number"] = claim.claim_number
        return result
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"gap web search LLM parse error: {e}")
        return {
            "claim_number": claim.claim_number,
            "results": [],
            "error": str(e),
        }
