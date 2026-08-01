"""주/보조 인용발명 최종 판정 (LLM-A / LLM-B).

설계 원칙
---------
계산은 알고리즘이, 기술 해석은 LLM이, 결론 조립은 다시 파이썬이 담당한다.

- 후보 목록은 citation_chain의 순수 함수가 확정한다. LLM은 그 목록 **안에서만**
  고를 수 있고, 커버리지·유사도 수치를 다시 계산하지 못한다.
- 주인용 선정(LLM-A)과 보조문헌 기술 검토(LLM-B)를 분리한다. 한 번의 호출로
  둘을 다 시키면 결론을 먼저 정하고 이유를 역산하는 편향이 생긴다.
- LLM-B는 결론이 아니라 사실 항목(기술분야 인접성·입출력 호환성·치환 가능성·
  기술적 모순·명시적 시사)만 출력한다. 결합 근거 등급은 그 값들로부터
  `_derive_combination_basis()`가 결정론적으로 조립한다.
- 어떤 검증이라도 실패하면 알고리즘 1위로 폴백하고 사유를 기록한다.

재현성
------
ARIA2는 세 엔진 모두 CLI(`claude -p` / `agy --print` / `codex exec`) 호출이라
temperature를 노출하지 않는다. 따라서 캐시가 사실상 유일한 재현성 수단이며,
사건 폴더가 아니라 프로젝트 데이터 디렉터리에 해시 키로 저장한다. 히스토리를
지워도 같은 입력이면 같은 판정이 재사용된다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from backend.models.schemas import ExtractedDocument, ParsedClaim, Settings
from backend.paths import ADJUDICATION_CACHE_DIR
from backend.services.ai_engine import call_ai
from backend.services.citation_chain import (
    _compute_family_context,
    _claim_family_groups,
    _load_cache,
    shortlist_primary_candidates,
    shortlist_secondary_candidates,
)
from backend.services.citation_extractor import normalize_label
from backend.services.prompt_loader import load_prompt, render_prompt

logger = logging.getLogger(__name__)

# 프롬프트 본문이 바뀌면 기존 캐시를 재사용하면 안 되므로 함께 해시에 넣는다.
ADJUDICATION_PROMPT_VERSION = 1

_FIELD_ADJACENCY = {"same", "adjacent", "distant"}
_IO_COMPATIBILITY = {"compatible", "requires_adaptation", "incompatible"}
_SUBSTITUTION = {"substitutable", "addable", "neither"}
_CONFIDENCE = {"high", "medium", "low"}


# ---------------------------------------------------------------------------
# JSON 응답 파싱
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> Optional[Dict]:
    """응답에서 첫 번째 완결된 JSON 객체를 꺼낸다.

    CLI 엔진은 코드펜스나 서두 문장을 덧붙이는 경우가 있어 그대로 파싱하면
    실패한다. 중괄호 균형을 세어 첫 객체만 잘라낸다.
    """
    if not text:
        return None
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.MULTILINE)
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(cleaned)):
        char = cleaned[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(cleaned[start:index + 1])
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def _as_doc_index(value, num_docs: int) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        index = int(value)
    except (TypeError, ValueError):
        return None
    return index if 0 <= index < num_docs else None


def _as_enum(value, allowed: set, default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


# ---------------------------------------------------------------------------
# 캐시
# ---------------------------------------------------------------------------

def _cache_key(
    claim: ParsedClaim,
    caches: Dict[int, Optional[Dict]],
    shortlist: Dict,
    stage: str,
    model: str,
) -> str:
    """청구항·비교판정·후보목록·프롬프트·모델이 모두 같을 때만 재사용한다."""
    payload = {
        "stage": stage,
        "claim": claim.model_dump() if hasattr(claim, "model_dump") else str(claim),
        "comparisons": {str(key): value for key, value in sorted(caches.items())},
        "shortlist": shortlist,
        "prompt_version": ADJUDICATION_PROMPT_VERSION,
        "model": model,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _cache_read(key: str) -> Optional[Dict]:
    path = ADJUDICATION_CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _cache_write(key: str, value: Dict) -> None:
    try:
        ADJUDICATION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        (ADJUDICATION_CACHE_DIR / f"{key}.json").write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError:
        logger.warning("판정 캐시를 저장하지 못했습니다 (key=%s).", key[:12])


# ---------------------------------------------------------------------------
# 프롬프트 입력 구성
# ---------------------------------------------------------------------------

def _doc_label(prior_docs: List[ExtractedDocument], doc_idx: int) -> str:
    if 0 <= doc_idx < len(prior_docs):
        return f"doc_idx {doc_idx} — {prior_docs[doc_idx].filename}"
    return f"doc_idx {doc_idx}"


def _format_core_elements(claim: ParsedClaim, context: Dict) -> str:
    root_key = context["root_key"]
    dynamic = context["dynamic_weights"]
    lines: List[str] = []
    for element in claim.elements:
        label = normalize_label(element.label)
        weight = dynamic.get((root_key, label))
        if weight is None or weight < 4.0:
            continue
        text = re.sub(r"\s+", " ", element.text or "").strip()
        lines.append(f"- ({element.label}) [가중치 {weight:.1f}] {text}")
    return "\n".join(lines) or "- (차별적 핵심으로 분류된 구성이 없습니다)"


def _format_primary_candidates(
    shortlist: Dict,
    claim: ParsedClaim,
    caches: Dict[int, Optional[Dict]],
    prior_docs: List[ExtractedDocument],
) -> str:
    root_key = str(claim.claim_number)
    element_by_label = {normalize_label(e.label): e for e in claim.elements}
    blocks: List[str] = []
    for candidate in shortlist.get("candidates", []):
        doc_idx = candidate["doc_idx"]
        cache = caches.get(doc_idx) or {}
        items = cache.get(root_key, []) if isinstance(cache.get(root_key), list) else []
        lines = [
            f"### {_doc_label(prior_docs, doc_idx)}",
            f"- 알고리즘 순위: {candidate['algorithm_rank']}"
            + ("  (핵심 직접개시로 강제 포함됨)" if candidate.get("forced_include") else ""),
            f"- 차별적 핵심 직접 개시도: {candidate['distinctive_direct_coverage']}",
            f"- 핵심 구성 커버리지: {candidate['core_coverage']}",
            f"- 핵심 공백 비중: {candidate['critical_gap_weight']}",
        ]
        quoted = 0
        for item in items:
            label = normalize_label(item.get("label"))
            element = element_by_label.get(label)
            quote = re.sub(r"\s+", " ", str(item.get("quote") or "")).strip()
            if not element or not quote:
                continue
            weight = candidate.get("core_disclosure_labels") or []
            marker = " ★핵심" if label in weight else ""
            lines.append(
                f"  · ({item.get('label')}){marker} [{item.get('judgment', '대응 없음')}] "
                f'"{quote[:220]}" (chunk {item.get("chunk_id", "-")})'
            )
            quoted += 1
            if quoted >= 8:
                break
        if quoted == 0:
            lines.append("  · (원문 발췌 근거 없음)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_gap_elements(secondary_shortlist: Dict, claim: ParsedClaim) -> str:
    element_by_label = {normalize_label(e.label): e for e in claim.elements}
    lines: List[str] = []
    for entry in secondary_shortlist.get("hard_gaps", []):
        label = entry.split(":", 1)[-1]
        element = element_by_label.get(normalize_label(label))
        text = re.sub(r"\s+", " ", (element.text if element else "")).strip()
        lines.append(f"- ({label}) [미커버] {text}")
    for entry in secondary_shortlist.get("soft_gaps", []):
        label = entry.split(":", 1)[-1]
        element = element_by_label.get(normalize_label(label))
        text = re.sub(r"\s+", " ", (element.text if element else "")).strip()
        lines.append(f"- ({label}) [차이 잔존] {text}")
    return "\n".join(lines) or "- (남은 공백 없음)"


def _format_secondary_candidates(
    secondary_shortlist: Dict, prior_docs: List[ExtractedDocument]
) -> str:
    blocks: List[str] = []
    for candidate in secondary_shortlist.get("candidates", []):
        doc_idx = candidate["doc_idx"]
        lines = [
            f"### {_doc_label(prior_docs, doc_idx)}",
            f"- 알고리즘 순위: {candidate['algorithm_rank']}",
            f"- 주인용 대비 보완 점수: {candidate['sub_score']}",
            f"- 핵심 공백 보완량: {candidate['critical_gap_evidence_gain']}",
        ]
        for filled in candidate.get("filled", [])[:8]:
            quote = re.sub(r"\s+", " ", str(filled.get("quote") or "")).strip()
            lines.append(
                f"  · ({filled['label']}) [{filled['gap_type']}] [{filled['judgment']}] "
                f'"{quote[:220]}" (chunk {filled.get("chunk_id", "-")})'
            )
            motivation = re.sub(r"\s+", " ", str(filled.get("motivation_quote") or "")).strip()
            if motivation:
                lines.append(f'      결합 시사 후보 발췌: "{motivation[:180]}"')
        if not candidate.get("filled"):
            lines.append("  · (공백에 대한 직접 발췌 없음)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 결합 근거 조립 (결정론)
# ---------------------------------------------------------------------------

def _derive_combination_basis(review: Dict, valid_quote_refs: set) -> str:
    """LLM이 낸 사실 항목으로부터 결합 근거 등급을 파이썬이 결정한다.

    LLM에 `obvious` 같은 결론을 직접 내게 하지 않는 이유는, 결론을 먼저 정하고
    나머지 항목을 사후에 맞춰 쓰는 편향을 막기 위해서다.
    """
    if review.get("technical_conflict"):
        return "unproven"
    quote_ref = review.get("explicit_suggestion_quote_ref")
    if quote_ref and str(quote_ref) in valid_quote_refs:
        return "explicit"
    if (
        review.get("field_adjacency") in {"same", "adjacent"}
        and review.get("io_compatibility") == "compatible"
        and review.get("substitution_feasibility") in {"substitutable", "addable"}
    ):
        return "implicit"
    return "unproven"


def _collect_quote_refs(candidate: Optional[Dict]) -> set:
    refs = set()
    for filled in (candidate or {}).get("filled", []):
        chunk_id = str(filled.get("chunk_id") or "").strip()
        if chunk_id:
            refs.add(chunk_id)
    return refs


# ---------------------------------------------------------------------------
# LLM-A: 주인용 선정
# ---------------------------------------------------------------------------

async def _adjudicate_primary(
    claim: ParsedClaim,
    context: Dict,
    shortlist: Dict,
    caches: Dict[int, Optional[Dict]],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
) -> Dict:
    allowed = {candidate["doc_idx"] for candidate in shortlist.get("candidates", [])}
    algorithm_top1 = shortlist.get("algorithm_top1")
    fallback = {
        "primary_idx": algorithm_top1,
        "source": "algorithm",
        "fallback_reason": None,
        "reason": "",
        "confidence": "",
        "critical_feature_basis": [],
    }

    prompt = render_prompt(
        "prompt_primary_selection.txt",
        claim_number=claim.claim_number,
        claim_text=re.sub(r"\s+", " ", claim.text or "").strip(),
        core_elements=_format_core_elements(claim, context),
        candidates=_format_primary_candidates(shortlist, claim, caches, prior_docs),
    )
    if not prompt.strip():
        fallback["fallback_reason"] = "prompt_missing"
        return fallback

    try:
        raw = await call_ai(prompt, load_prompt("system_adjudicate.txt"), settings, agent="compare")
    except Exception as error:  # CLI 실패는 판정 실패로 처리하고 알고리즘을 따른다.
        logger.warning("LLM-A 주인용 판정 호출 실패: %s", error)
        fallback["fallback_reason"] = f"call_failed: {type(error).__name__}"
        return fallback

    parsed = _extract_json_object(raw)
    if parsed is None:
        fallback["fallback_reason"] = "unparsable_response"
        return fallback

    selected = _as_doc_index(parsed.get("selected_primary_idx"), len(prior_docs))
    if selected is None or selected not in allowed:
        fallback["fallback_reason"] = f"out_of_shortlist: {parsed.get('selected_primary_idx')!r}"
        return fallback

    return {
        "primary_idx": selected,
        "source": "llm",
        "fallback_reason": None,
        "reason": str(parsed.get("reason") or "").strip(),
        "confidence": _as_enum(parsed.get("confidence"), _CONFIDENCE, "medium"),
        "critical_feature_basis": [
            str(label) for label in parsed.get("critical_feature_basis") or [] if str(label).strip()
        ],
        "runner_up_idx": _as_doc_index(parsed.get("runner_up_idx"), len(prior_docs)),
        "runner_up_gap": str(parsed.get("runner_up_gap") or "").strip(),
    }


# ---------------------------------------------------------------------------
# LLM-B: 보조문헌 기술 검토
# ---------------------------------------------------------------------------

async def _adjudicate_secondary(
    claim: ParsedClaim,
    primary_idx: int,
    secondary_shortlist: Dict,
    prior_docs: List[ExtractedDocument],
    settings: Settings,
) -> Dict:
    candidates = secondary_shortlist.get("candidates", [])
    allowed = {candidate["doc_idx"] for candidate in candidates}
    algorithm_top1 = secondary_shortlist.get("algorithm_top1")
    by_idx = {candidate["doc_idx"]: candidate for candidate in candidates}
    fallback = {
        "secondary_idx": algorithm_top1,
        "source": "algorithm",
        "fallback_reason": None,
        "combination_basis": "unproven",
        "technical_review": {},
        "reason": "",
        "confidence": "",
    }

    prompt = render_prompt(
        "prompt_secondary_review.txt",
        claim_number=claim.claim_number,
        claim_text=re.sub(r"\s+", " ", claim.text or "").strip(),
        primary_document=_doc_label(prior_docs, primary_idx),
        gap_elements=_format_gap_elements(secondary_shortlist, claim),
        candidates=_format_secondary_candidates(secondary_shortlist, prior_docs),
    )
    if not prompt.strip():
        fallback["fallback_reason"] = "prompt_missing"
        return fallback

    try:
        raw = await call_ai(prompt, load_prompt("system_adjudicate.txt"), settings, agent="compare")
    except Exception as error:
        logger.warning("LLM-B 보조문헌 검토 호출 실패: %s", error)
        fallback["fallback_reason"] = f"call_failed: {type(error).__name__}"
        return fallback

    parsed = _extract_json_object(raw)
    if parsed is None:
        fallback["fallback_reason"] = "unparsable_response"
        return fallback

    raw_selected = parsed.get("selected_secondary_idx")
    # 명시적 null은 "적합한 보조문헌 없음"이라는 유효한 판정이다.
    if raw_selected is None:
        selected = None
    else:
        selected = _as_doc_index(raw_selected, len(prior_docs))
        if selected is None or selected not in allowed:
            fallback["fallback_reason"] = f"out_of_shortlist: {raw_selected!r}"
            return fallback
        if selected == primary_idx:
            fallback["fallback_reason"] = "secondary_equals_primary"
            return fallback

    review = {
        "field_adjacency": _as_enum(parsed.get("field_adjacency"), _FIELD_ADJACENCY, "distant"),
        "io_compatibility": _as_enum(
            parsed.get("io_compatibility"), _IO_COMPATIBILITY, "requires_adaptation"
        ),
        "substitution_feasibility": _as_enum(
            parsed.get("substitution_feasibility"), _SUBSTITUTION, "neither"
        ),
        "technical_conflict": bool(parsed.get("technical_conflict")),
        "technical_conflict_reason": str(parsed.get("technical_conflict_reason") or "").strip(),
        "explicit_suggestion_quote_ref": (
            str(parsed.get("explicit_suggestion_quote_ref")).strip()
            if parsed.get("explicit_suggestion_quote_ref") else None
        ),
        "filled_labels": [
            str(label) for label in parsed.get("filled_labels") or [] if str(label).strip()
        ],
    }
    valid_refs = _collect_quote_refs(by_idx.get(selected))
    quote_ref = review["explicit_suggestion_quote_ref"]
    if quote_ref and quote_ref not in valid_refs:
        # 실재하지 않는 chunk_id는 결합 근거로 인정하지 않는다. 판정 자체를
        # 버리지는 않고 근거만 무효화해 `implicit` 이하로 떨어뜨린다.
        review["explicit_suggestion_quote_ref"] = None
        review["quote_ref_rejected"] = quote_ref

    return {
        "secondary_idx": selected,
        "source": "llm",
        "fallback_reason": None,
        "combination_basis": _derive_combination_basis(review, valid_refs),
        "technical_review": review,
        "reason": str(parsed.get("reason") or "").strip(),
        "confidence": _as_enum(parsed.get("confidence"), _CONFIDENCE, "medium"),
    }


# ---------------------------------------------------------------------------
# 감사 기록
# ---------------------------------------------------------------------------

def _build_audit(
    shortlist: Dict,
    primary_decision: Dict,
    secondary_shortlist: Optional[Dict],
    secondary_decision: Optional[Dict],
) -> Dict:
    candidates = shortlist.get("candidates", [])
    rank_by_idx = {c["doc_idx"]: c["algorithm_rank"] for c in candidates}
    score_by_idx = {c["doc_idx"]: c["distinctive_direct_coverage"] for c in candidates}
    algorithm_top1 = shortlist.get("algorithm_top1")
    chosen_primary = primary_decision.get("primary_idx")

    reason_codes: List[str] = []
    if chosen_primary is not None and chosen_primary != algorithm_top1:
        reason_codes.append("llm_overrode_primary")
        if score_by_idx.get(chosen_primary, 0.0) < score_by_idx.get(algorithm_top1, 0.0):
            reason_codes.append("chose_lower_direct_disclosure")
    if primary_decision.get("fallback_reason"):
        reason_codes.append("primary_fallback")
    if (secondary_decision or {}).get("fallback_reason"):
        reason_codes.append("secondary_fallback")
    if (secondary_decision or {}).get("combination_basis") == "unproven":
        reason_codes.append("combination_unproven")

    secondary_top1 = (secondary_shortlist or {}).get("algorithm_top1")
    chosen_secondary = (secondary_decision or {}).get("secondary_idx")

    return {
        "shortlist": candidates,
        "algorithm_top1": algorithm_top1,
        "algorithm_rank_of_choice": rank_by_idx.get(chosen_primary),
        "llm_selected_primary": chosen_primary if primary_decision.get("source") == "llm" else None,
        "score_margin": round(
            abs(score_by_idx.get(algorithm_top1, 0.0) - score_by_idx.get(chosen_primary, 0.0)), 4
        ),
        "changed_primary": chosen_primary != algorithm_top1,
        "changed_secondary": chosen_secondary != secondary_top1,
        "agreed_with_algorithm": (
            chosen_primary == algorithm_top1 and chosen_secondary == secondary_top1
        ),
        "primary_source": primary_decision.get("source"),
        "primary_confidence": primary_decision.get("confidence"),
        "primary_fallback_reason": primary_decision.get("fallback_reason"),
        "secondary_source": (secondary_decision or {}).get("source"),
        "secondary_confidence": (secondary_decision or {}).get("confidence"),
        "secondary_fallback_reason": (secondary_decision or {}).get("fallback_reason"),
        "secondary_rejected_for_explicit_risk": (secondary_shortlist or {}).get(
            "rejected_for_explicit_risk", {}
        ),
        "critical_reason_codes": reason_codes,
    }


# ---------------------------------------------------------------------------
# 공개 진입점
# ---------------------------------------------------------------------------

async def adjudicate_family(
    family_claims: List[ParsedClaim],
    caches: Dict[int, Optional[Dict]],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
) -> Optional[Dict]:
    """독립항 하나에 대해 주인용·보조문헌을 확정한다.

    후보가 1개뿐이면 해당 단계의 LLM 호출을 건너뛴다. 문헌이 2개인 사건에서는
    LLM-A가 고를 것이 없으므로 자동으로 LLM-B만 수행된다.
    """
    num_docs = len(prior_docs)
    context = _compute_family_context(family_claims, caches, num_docs)
    if not context:
        return None
    root = context["root"]

    shortlist = shortlist_primary_candidates(family_claims, caches, num_docs, context)
    if not shortlist:
        return None

    # 신규성 게이트를 통과한 사건은 결합 자체를 하지 않으므로 판정 대상이 아니다.
    if shortlist.get("novelty_selected_document") is not None:
        return {
            "root_claim": root.claim_number,
            "primary_idx": shortlist["novelty_selected_document"],
            "secondary_idx": None,
            "skipped": "novelty_single_reference",
        }

    model = getattr(settings, "model_compare", "") or getattr(settings, "engine", "")
    cache_key = _cache_key(root, caches, shortlist, "primary", model)
    cached = _cache_read(cache_key)
    if cached is not None:
        cached.setdefault("audit", {})["cache_hit"] = True
        return cached

    if shortlist.get("needs_adjudication"):
        primary_decision = await _adjudicate_primary(
            root, context, shortlist, caches, prior_docs, settings
        )
    else:
        primary_decision = {
            "primary_idx": shortlist.get("algorithm_top1"),
            "source": "algorithm",
            "fallback_reason": "single_eligible_candidate",
            "reason": "",
            "confidence": "",
            "critical_feature_basis": [],
        }

    primary_idx = primary_decision.get("primary_idx")
    if primary_idx is None:
        return None

    secondary_shortlist = shortlist_secondary_candidates(
        root, caches, num_docs, primary_idx, context["weights"]
    )
    secondary_decision: Optional[Dict] = None
    if secondary_shortlist.get("needs_adjudication"):
        secondary_decision = await _adjudicate_secondary(
            root, primary_idx, secondary_shortlist, prior_docs, settings
        )

    # LLM-B가 "적합한 보조문헌 없음"을 명시적으로 판정한 경우와, 애초에 판정을
    # 돌리지 않았거나 폴백한 경우를 구분한다. 전자는 기술 검토가 결합을 부정한
    # 것이므로 알고리즘이 보조문헌을 도로 끼워 넣으면 안 된다.
    secondary_explicitly_none = bool(
        secondary_decision
        and secondary_decision.get("source") == "llm"
        and secondary_decision.get("secondary_idx") is None
    )

    result = {
        "root_claim": root.claim_number,
        "primary_idx": primary_idx,
        "secondary_idx": (secondary_decision or {}).get("secondary_idx"),
        "secondary_explicitly_none": secondary_explicitly_none,
        "primary_decision": primary_decision,
        "secondary_decision": secondary_decision or {},
        "combination_basis": (secondary_decision or {}).get("combination_basis", "unproven"),
        "technical_review": (secondary_decision or {}).get("technical_review", {}),
        "audit": _build_audit(
            shortlist, primary_decision, secondary_shortlist, secondary_decision
        ),
    }
    result["audit"]["cache_hit"] = False
    _cache_write(cache_key, result)
    return result


async def adjudicate_all(
    job_dir: str,
    claims: List[ParsedClaim],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
    progress=None,
) -> Dict[str, Dict]:
    """모든 독립항 패밀리에 대해 판정을 수행하고 root claim 번호로 반환한다.

    실패한 패밀리는 결과에서 빠지며, 그 경우 citation_chain이 기존 알고리즘
    선정을 그대로 사용한다.
    """
    num_docs = len(prior_docs)
    if num_docs < 2:
        return {}
    caches = {index: _load_cache(job_dir, index) for index in range(num_docs)}
    family_groups, _orphans = _claim_family_groups(claims)

    decisions: Dict[str, Dict] = {}
    for root_number, family_claims in sorted(family_groups.items()):
        if progress:
            await progress(f"청구항 {root_number} 인용발명 판정 중…")
        try:
            decision = await adjudicate_family(family_claims, caches, prior_docs, settings)
        except Exception as error:
            logger.warning("청구항 %s 인용발명 판정 실패: %s", root_number, error)
            continue
        if decision:
            decisions[str(root_number)] = decision
    return decisions
