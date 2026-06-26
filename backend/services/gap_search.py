"""
gap_search.py - 보완문서 검색 (인용발명 미커버 구성요소 웹검색)

동작:
  1. find_uncovered_elements() - LLM 없이 즉시 미커버 구성요소를 계산
  2. web_search_gap_documents() - LLM 웹검색을 우선 시도하고, 실패 시 HTTP 검색으로 폴백
"""
from __future__ import annotations

import json
import logging
import re
from html import unescape
from typing import List
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx

from backend.models.schemas import ParsedClaim, Settings
from backend.services.ai_engine import call_ai
from backend.services.citation_chain import _JUDGMENT_SCORE, _PRIMARY_COVER_THRESHOLD
from backend.services.citation_extractor import load_comparisons, normalize_label

logger = logging.getLogger(__name__)

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/137.0.0.0 Safari/537.36"
    )
}
_SEARCH_TIMEOUT = httpx.Timeout(20.0, connect=10.0)
_MAX_TARGETS = 4
_MAX_DOCS_PER_TARGET = 3


def find_uncovered_elements(
    job_dir: str,
    claim: ParsedClaim,
    doc_filenames: List[str],
) -> dict:
    """비교 캐시에서 미커버 구성요소를 중요도 순으로 반환한다."""
    claim_key = str(claim.claim_number)
    target_key = claim_key
    if claim.claim_type == "dependent" and claim.parent_claim:
        target_key = str(claim.parent_claim)

    per_label: dict[str, list[tuple[int, str, int]]] = {}
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
            judgment = item.get("judgment", "없음")
            score = _JUDGMENT_SCORE.get(judgment, 0)
            per_label.setdefault(label, []).append((score, judgment, doc_idx))

    if not analyzed:
        return {
            "claim_number": claim.claim_number,
            "analyzed": False,
            "uncovered": [],
            "covered": [],
        }

    uncovered, covered = [], []
    for elem in claim.elements:
        label = normalize_label(elem.label)
        entries = per_label.get(label, [])
        if entries:
            best_score, best_judgment, best_doc = max(entries, key=lambda e: e[0])
        else:
            best_score, best_judgment, best_doc = 0, "없음", -1
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


_WEB_SYSTEM = """당신은 특허 선행기술 조사 전문가입니다.
웹검색 도구를 사용하여, 현재 인용발명들이 개시하지 못한 청구항 구성요소를 개시하는 보완 선행문헌을 직접 찾아야 합니다.
검색을 모두 마친 후 최종 응답은 JSON 형식으로만 작성하세요. 다른 설명이나 마크다운 코드블록은 사용하지 마세요."""

_WEB_PROMPT_TMPL = """청구항 {claim_number}의 진보성 부정을 위해 추가 선행문헌(보완문서)이 필요합니다.
결합 논리의 보조인용발명 후보로 아래 [검색 대상] 구성요소를 개시하는 특허문헌/논문을 웹검색으로 직접 찾아주세요.

[검색 대상: 현재 어떤 인용발명도 개시하지 못한 구성요소 (중요도 순)]
{uncovered_text}

[기술분야 컨텍스트]
{field_text}

검색 방법:
- 구성요소마다 핵심 기술 특징을 뽑아 국문/영문으로 웹검색을 수행하세요.
- 특허문헌은 우선적으로 patents.google.com 결과를 확인하세요.
- 중요도가 높은 구성요소부터 검색하고, 구성요소마다 후보 문헌을 2~3개 찾으세요.
- 후보 문헌 페이지를 읽고 실제로 해당 기술 특징이 개시되어 있는지 확인하세요.

최종 응답 JSON:
{{
  "results": [
    {{
      "label": "구성요소 라벨",
      "feature_ko": "검색한 기술 특징 한 줄 요약",
      "queries_used": ["실제 사용한 검색어"],
      "documents": [
        {{
          "title": "문헌 제목",
          "number": "공개/등록번호",
          "url": "문헌 링크",
          "summary": "해당 문헌이 구성요소를 개시하는 이유 요약",
          "relevance": "high|medium|low"
        }}
      ]
    }}
  ]
}}"""

_VERIFY_SYSTEM = """당신은 특허 선행기술 후보 검증 전문가입니다.
1차 검색 결과가 실제로 청구항의 미대응 구성요소를 개시하는지 다시 검증합니다.
최종 응답은 JSON 객체만 반환하세요."""

_VERIFY_PROMPT_TMPL = """청구항 {claim_number}의 미대응 구성요소에 대한 1차 검색 후보를 다시 검증해주세요.

[검증 대상 미대응 구성요소]
{uncovered_text}

[1차 후보 결과]
{candidate_text}

검증 기준:
- direct: 문헌이 해당 구성요소를 직접 또는 매우 명확하게 개시
- functional: 용어는 다르지만 기능적으로 유사
- weak: 관련은 있으나 개시라고 보기 어려움
- unsupported: 확인 불가

출력 JSON:
{{
  "results": [
    {{
      "label": "구성요소 라벨",
      "documents": [
        {{
          "number": "문헌 번호",
          "verification_status": "direct|functional|weak|unsupported",
          "confidence": "high|medium|low",
          "reason": "판단 이유 1~2문장",
          "quote": "가능하면 근거 문구"
        }}
      ]
    }}
  ]
}}"""


def _format_uncovered(uncovered: List[dict]) -> str:
    lines = []
    for u in uncovered:
        try:
            stars = "★" * max(1, min(5, int(u["importance"])))
        except (ValueError, TypeError):
            stars = "중요도 미상"
        lines.append(f"- ({u['label']}) [{stars}] {u['text']}")
        lines.append(
            f"  현재 최고 판정: {u['best_judgment']}"
            + (f" ({u['best_doc']})" if u["best_doc"] else "")
        )
    return "\n".join(lines)


def _format_candidates_for_verification(results: List[dict]) -> str:
    lines = []
    for target in results:
        lines.append(f"- ({target.get('label', '')}) {target.get('feature_ko', '')}".strip())
        for doc in (target.get("documents") or [])[:4]:
            title = doc.get("title", "")
            number = doc.get("number", "")
            url = doc.get("url", "")
            summary = doc.get("summary", "")
            relevance = doc.get("relevance", "")
            lines.append(f"  - title={title} / number={number} / relevance={relevance} / url={url}")
            if summary:
                lines.append(f"    summary={summary}")
    return "\n".join(lines)


def _extract_json_object(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            raise
        parsed = json.loads(raw[start:end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(text or "")).strip()


def _extract_keywords(text: str, max_terms: int = 6) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-_/]{1,}|[가-힣]{2,}", text or "")
    seen: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.append(token)
        if len(seen) >= max_terms:
            break
    return seen


def _build_queries(target: dict, field_text: str) -> list[str]:
    feature_terms = _extract_keywords(target.get("text", ""), max_terms=6)
    field_terms = _extract_keywords(field_text, max_terms=4)
    quoted_feature = " ".join(f'"{term}"' for term in feature_terms[:3])
    loose_feature = " ".join(feature_terms[:5])
    field_suffix = " ".join(field_terms[:3])
    queries = [
        f"site:patents.google.com {quoted_feature} {field_suffix}".strip(),
        f"site:patents.google.com {loose_feature} patent".strip(),
        f"{loose_feature} {field_suffix} patent".strip(),
    ]
    deduped: list[str] = []
    for query in queries:
        normalized = re.sub(r"\s+", " ", query).strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _decode_duckduckgo_href(href: str) -> str:
    if not href:
        return ""
    href = unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("/l/?"):
        parsed = urlparse("https://duckduckgo.com" + href)
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unescape(target)
    return href


def _extract_patent_number(url: str, title: str) -> str:
    match = re.search(r"/patent/([^/?#]+)", url or "")
    if match:
        return match.group(1)
    match = re.search(r"\b([A-Z]{2}\d[\dA-Z]*)\b", title or "")
    return match.group(1) if match else ""


async def _search_duckduckgo(query: str) -> list[dict]:
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    async with httpx.AsyncClient(headers=_HTTP_HEADERS, timeout=_SEARCH_TIMEOUT, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
    html = response.text
    results: list[dict] = []
    pattern = re.compile(
        r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    for href, title_html in pattern.findall(html):
        real_url = _decode_duckduckgo_href(href)
        if "patents.google.com/patent/" not in real_url:
            continue
        title = _clean_text(re.sub(r"<[^>]+>", " ", title_html))
        if not title:
            continue
        results.append(
            {
                "title": title,
                "url": real_url,
                "number": _extract_patent_number(real_url, title),
            }
        )
    return results


async def _search_target_documents(target: dict, field_text: str) -> tuple[list[str], list[dict]]:
    queries = _build_queries(target, field_text)
    docs: list[dict] = []
    seen_urls: set[str] = set()
    for query in queries:
        try:
            hits = await _search_duckduckgo(query)
        except Exception as exc:
            logger.warning("gap fallback search failed for query=%s: %s", query, exc)
            continue
        for hit in hits:
            if hit["url"] in seen_urls:
                continue
            seen_urls.add(hit["url"])
            docs.append(
                {
                    "title": hit["title"],
                    "number": hit["number"],
                    "url": hit["url"],
                    "summary": f"검색어 '{query}'로 찾은 patents.google.com 후보 문헌입니다.",
                    "relevance": "medium",
                    "source": "http_fallback",
                }
            )
            if len(docs) >= _MAX_DOCS_PER_TARGET:
                break
        if len(docs) >= _MAX_DOCS_PER_TARGET:
            break
    return queries, docs


async def _fallback_http_gap_search(claim: ParsedClaim, targets: List[dict], field_text: str) -> dict:
    results = []
    for target in targets:
        queries, docs = await _search_target_documents(target, field_text)
        results.append(
            {
                "label": target.get("label", ""),
                "feature_ko": target.get("text", ""),
                "queries_used": queries,
                "documents": docs,
            }
        )
    return {
        "claim_number": claim.claim_number,
        "results": results,
        "fallback_used": True,
        "verification_applied": False,
    }


async def _verify_gap_documents(
    claim: ParsedClaim,
    targets: List[dict],
    search_result: dict,
    settings: Settings,
) -> dict:
    prompt = _VERIFY_PROMPT_TMPL.format(
        claim_number=claim.claim_number,
        uncovered_text=_format_uncovered(targets),
        candidate_text=_format_candidates_for_verification(search_result.get("results", []) or []),
    )
    raw = await call_ai(
        prompt=prompt,
        system=_VERIFY_SYSTEM,
        settings=settings,
        agent="compare",
        web_search=False,
    )
    return _extract_json_object(raw)


def _merge_verification(search_result: dict, verification: dict) -> dict:
    by_label = {
        item.get("label"): {doc.get("number"): doc for doc in item.get("documents", []) or []}
        for item in verification.get("results", []) or []
    }
    for target in search_result.get("results", []) or []:
        verified_docs = by_label.get(target.get("label"), {})
        merged_docs = []
        for doc in target.get("documents", []) or []:
            verified = verified_docs.get(doc.get("number"))
            if verified:
                doc["verification_status"] = verified.get("verification_status", "")
                doc["confidence"] = verified.get("confidence", doc.get("relevance", ""))
                doc["verification_reason"] = verified.get("reason", "")
                doc["verification_quote"] = verified.get("quote", "")
            else:
                doc["verification_status"] = "unsupported"
                doc["confidence"] = "low"
                doc["verification_reason"] = "2차 검증에서 해당 구성요소 개시를 확인하지 못했습니다."
                doc["verification_quote"] = ""
            if doc.get("verification_status") in {"direct", "functional", "weak"}:
                merged_docs.append(doc)
        target["documents"] = merged_docs
        target["verified_count"] = len(merged_docs)
    return search_result


async def web_search_gap_documents(
    claim: ParsedClaim,
    gap_result: dict,
    settings: Settings,
) -> dict:
    """LLM 웹검색을 우선 시도하고, 실패 시 HTTP 검색으로 폴백한다."""
    uncovered = gap_result.get("uncovered", [])
    if not uncovered:
        return {"claim_number": claim.claim_number, "results": []}

    targets = uncovered[:_MAX_TARGETS]
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
            agent="compare",
            web_search=True,
        )
        result = _extract_json_object(raw)
        try:
            verification = await _verify_gap_documents(claim, targets, result, settings)
            result = _merge_verification(result, verification)
            result["verification_applied"] = True
        except Exception as exc:
            logger.warning("gap search verification skipped: %s", exc)
            result["verification_applied"] = False
            result["verification_error"] = str(exc)
        result["claim_number"] = claim.claim_number
        return result
    except Exception as exc:
        logger.warning("gap LLM web search failed, switching to HTTP fallback: %s", exc)
        fallback = await _fallback_http_gap_search(claim, targets, field_text)
        fallback["error"] = str(exc)
        return fallback
