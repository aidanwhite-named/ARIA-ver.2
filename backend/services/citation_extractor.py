"""
인용 추출 파이프라인 — 인용발명 원문(全文)을 Claude에 직접 전달하여 구성요소 대비.

[최적화 구조]
- 비교 단계에서 모든 문헌을 한 번에 비교하고 comparisons_{doc_idx}.json 캐시
- 보고서 생성 시에는 캐시에서 로드만 함(인용발명 원문 재전송 없음)
- 혼합 모드의 큰 독립항 행렬은 핵심 구성 선별 1회와 주 후보 범용 구성 배치 1회로 처리
- 정밀 모드는 모든 문헌과 모든 구성을 한 번의 통합 호출로 처리
"""
from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from backend.models.schemas import ClaimElement, ElementMatch, EvidenceSpan, ExtractedDocument, ParsedClaim, Settings
from backend.services.ai_engine import call_ai
from backend.services.prompt_loader import load_prompt, render_prompt

logger = logging.getLogger(__name__)


class CompareFailed(Exception):
    """구성대비 LLM 호출 또는 응답 파싱 실패를 나타낸다.

    실제로 인용발명에 대응 내용이 없어 나온 정상 JSON 결과와
    CLI 호출/파싱 실패로 인한 빈 결과를 구분하기 위한 예외다.
    이 예외가 발생하면 빈 비교 결과를 캐시하지 않고 호출부에 오류를 전달한다.
    """


# 엔진별 입력 예산 (relevant, hard, hybrid_total, hybrid_min).
# Claude CLI는 긴 stdin에서 매우 드물게 보수적으로 단절된다.
# Gemini는 100만 토큰 컨텍스트에서 인용발명 원문을 그대로 넣어도 단절을 피할 수 있다.
# 문헌 길이나 엔진별 입력 한계로 응답이 중간에 끊기는 상황을 줄이기 위한
# 보수적 예산이다. 구성대비 정확도를 해치지 않는 범위에서 안정성을 우선한다.
_ENGINE_BUDGETS = {
    "gemini": (300_000, 400_000, 300_000, 30_000),
    "agy": (300_000, 400_000, 300_000, 30_000),
    "claude": (45_000, 60_000, 55_000, 5_000),
}
_DEFAULT_BUDGET = (45_000, 60_000, 55_000, 5_000)
_CHUNK_SIZE = 1_200
_CACHE_META_KEY = "_meta"
_CACHE_SCHEMA_VERSION = 18
_MIXED_TOTAL_BUDGET = 80_000
_MIXED_MIN_DOC_BUDGET = 8_000
_CORE_SCREEN_TOTAL_BUDGET = 30_000
_CORE_SCREEN_MIN_DOC_BUDGET = 3_000
_GENERIC_PRIMARY_TOTAL_BUDGET = 20_000
_GENERIC_PRIMARY_MIN_DOC_BUDGET = 3_000
_CORE_FIRST_MIN_DOCS = 3
_CORE_FIRST_MIN_MATRIX_CELLS = 20
_CORE_FIRST_PRIMARY_CANDIDATES = 2
_DEFAULT_DEPENDENT_CANDIDATE_DOC_LIMIT = 3
_FALSE_NEGATIVE_REVIEW_MAX_DOCS = 5
_FALSE_NEGATIVE_REVIEW_MIN_OVERLAP = 0.55
_JUDGMENT_RANK = {
    "동일": 5,
    "실질적 동일": 4,
    "일부 차이": 3,
    "일부 유사": 2,
    "차이": 1,
    "대응 없음": 0,
}

_SELECTION_OR_RE = re.compile(
    r"(?:또는|중\s*적어도\s*하나|하나\s*이상|및/또는|\band/or\b|\bor\b|at\s+least\s+one\s+of)",
    re.IGNORECASE,
)
_SELECTION_CONDITION_RE = re.compile(
    r"(?:에\s*따라|를\s*고려하여|을\s*고려하여|에\s*기초하여|를\s*판단하여|을\s*판단하여|"
    r"인\s*경우|에\s*대응하여|선택|전환|분기|종류|유형|모드|"
    r"based\s+on|according\s+to|depending\s+on|in\s+response\s+to|select|switch|branch|"
    r"type|mode|condition|alternative)",
    re.IGNORECASE,
)
_SELECTION_STRUCTURE_KEYWORDS = [
    "상위개념", "복수", "대안", "선택", "전환", "분기", "조건", "기준", "판단", "종류", "유형", "모드",
    "category", "type", "mode", "condition", "criterion", "determine", "select", "switch", "branch",
    "alternative", "multiple",
]

_GENERIC_COMPONENT_RE = re.compile(
    r"(?:"
    r"\bcpu\b|\bprocessor\b|\bcontroller\b|\bmemory\b|\bstorage\b|"
    r"\binterface\b|\btransceiver\b|\bdisplay\b|\bbattery\b|\binput\b|"
    r"\boutput\b|\bsource\b|\bamplifier\b|\btransducer\b|\bsensor\b|"
    r"프로세서|제어부|제어기|컨트롤러|메모리|저장부|통신부|송수신부|"
    r"인터페이스|입력부|출력부|표시부|디스플레이|전원부|배터리|센서|"
    r"입력\s*신호|출력\s*신호|신호\s*소스|앰프|증폭기|변환기"
    r")",
    re.IGNORECASE,
)
_NON_GENERIC_LIMITATION_RE = re.compile(
    r"(?:\d|수치|범위|이상|이하|초과|미만|보다\s*(?:크|작)|비율|상대|"
    r"피드백|학습|암호|복호|보정|적응|동기|임계|조건|선택|전환|분기|"
    r"상호\s*작용|연동|왜곡|고조파|에너지|트랜지스터|부하|효과|개선|"
    r"based\s+on|in\s+response\s+to|greater\s+than|less\s+than|ratio|"
    r"relative|feedback|adaptive|threshold|synchron|encrypt|decrypt|"
    r"calibrat|distortion|harmonic|transistor|loaded|improv|effect)",
    re.IGNORECASE,
)

# 통합 비교에서 한 문헌의 핵심 문단이 다른 문헌들 사이에 묻혀 ``대응 없음``으로
# 끝나는 것을 막기 위한 다국어 기술 개념 축이다. 이는 대응을 자동 인정하는 규칙이
# 아니라, 명시 원문이 있는 문헌만 개별 재검증 호출로 올리는 용도다.
_TECHNICAL_CONCEPT_PATTERNS = {
    "harmonic": re.compile(r"고조파|harmonic", re.IGNORECASE),
    "distortion": re.compile(r"왜곡|distortion|non[- ]?linear|overdrive", re.IGNORECASE),
    "adjustment": re.compile(
        r"조정|조절|가변|변경|설정|보정|adjust|vary|control|setting|calibrat",
        re.IGNORECASE,
    ),
    "relative_order": re.compile(
        r"상대|비율|보다|차수|에너지|스펙트럼|relative|ratio|proportion|"
        r"energy|spectrum|order|2nd|second|3rd|third|4th|fourth|5th|fifth",
        re.IGNORECASE,
    ),
    "circuit": re.compile(
        r"회로|트랜지스터|증폭기|피드백|circuit|transistor|jfet|fet|amplifier|feedback",
        re.IGNORECASE,
    ),
    "load": re.compile(r"부하|load|resistor|저항", re.IGNORECASE),
    "signal_path": re.compile(r"입력|출력|단계|input|output|gate|drain|stage", re.IGNORECASE),
}
_GENERIC_FUNCTION_RE = re.compile(
    r"(?:제공|공급|수신|송신|저장|표시|출력|입력|증폭|변환|처리|"
    r"provide|supply|receive|transmit|store|display|output|input|amplif|convert|process)",
    re.IGNORECASE,
)


def _importance_value(value: object) -> int:
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return 3


def _is_batchable_generic_element(element: ClaimElement) -> bool:
    """Return True only for a simple, low-weight element safe to defer.

    `importance` alone is not trusted: numerical, comparative, conditional,
    relational and effect-linked limitations always stay in the core screen.
    """
    text = " ".join((element.text or "").split())
    if not text or len(text) > 140:
        return False
    if _importance_value(element.importance) > 3 or bool(element.is_sub):
        return False
    if _is_conditioned_selection_text(text):
        return False
    if not _GENERIC_COMPONENT_RE.search(text):
        return False
    if _NON_GENERIC_LIMITATION_RE.search(text):
        return False
    functions = {match.lower() for match in _GENERIC_FUNCTION_RE.findall(text)}
    if len(functions) > 3:
        return False
    if len(re.findall(r"(?:및|또는|그리고|\band\b|\bor\b)", text, re.IGNORECASE)) > 1:
        return False
    return True


def _is_conditioned_selection_text(text: str) -> bool:
    return bool(_SELECTION_OR_RE.search(text or "") and _SELECTION_CONDITION_RE.search(text or ""))


def _technical_concepts(text: str) -> set[str]:
    """Return the technical concept axes explicitly present in Korean or English text."""
    source = text or ""
    return {
        name for name, pattern in _TECHNICAL_CONCEPT_PATTERNS.items()
        if pattern.search(source)
    }


def _needs_false_negative_review(element: ClaimElement) -> bool:
    """Identify a compound technical limitation that should not be dismissed in one batch."""
    text = " ".join((element.text or "").split())
    concepts = _technical_concepts(text)
    return bool(text) and (
        len(text) > 100
        or len(concepts - {"signal_path"}) >= 2
        or _is_conditioned_selection_text(text)
    )


def _best_document_concept_overlap(doc: ExtractedDocument, element: ClaimElement) -> float:
    """Measure whether one source chunk carries the element's unusual technical axes.

    The score is intentionally only a *review trigger*.  It does not produce a
    judgment and never substitutes for a model-provided quotation.
    """
    expected = _technical_concepts(element.text)
    expected.discard("signal_path")  # input/output alone is not a differentiating clue.
    if len(expected) < 2:
        return 0.0

    best = 0.0
    for _chunk_id, chunk_text in _doc_chunks(doc):
        found = _technical_concepts(chunk_text)
        overlap = len(expected & found) / len(expected)
        best = max(best, overlap)
    return best


def _false_negative_review_candidates(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    doc_results: List[List[Dict]],
) -> List[tuple[int, List[ClaimElement]]]:
    """Find a small set of documents whose compound-feature evidence merits retry.

    An integrated response can be syntactically complete while missing every
    relevant passage in one document.  We only retry a document where a single
    source chunk has multiple technical axes from a compound element and the
    integrated response supplied no quotation for that element.
    """
    candidates: List[tuple[float, int, List[ClaimElement]]] = []
    for doc_idx, doc in enumerate(prior_docs):
        items_by_label = {
            normalize_label(item.get("label", "")): item
            for item in (doc_results[doc_idx] if doc_idx < len(doc_results) else [])
        }
        review_elements: List[ClaimElement] = []
        score = 0.0
        for element in elements:
            if not _needs_false_negative_review(element):
                continue
            item = items_by_label.get(normalize_label(element.label), {})
            if item.get("quote") or item.get("judgment") not in {"대응 없음", "차이"}:
                continue
            overlap = _best_document_concept_overlap(doc, element)
            if overlap < _FALSE_NEGATIVE_REVIEW_MIN_OVERLAP:
                continue
            review_elements.append(element)
            score += overlap * max(1, _importance_value(element.importance))
        if review_elements:
            candidates.append((score, doc_idx, review_elements))

    candidates.sort(key=lambda value: (-value[0], value[1]))
    return [
        (doc_idx, review_elements)
        for _score, doc_idx, review_elements in candidates[:_FALSE_NEGATIVE_REVIEW_MAX_DOCS]
    ]


def _merge_precision_review_results(existing: List[Dict], reviewed: List[Dict]) -> None:
    """Keep a recheck only when it upgrades a previously unsupported judgment."""
    by_label = {
        normalize_label(item.get("label", "")): index
        for index, item in enumerate(existing)
    }
    for item in reviewed:
        index = by_label.get(normalize_label(item.get("label", "")))
        if index is None:
            continue
        previous = existing[index]
        old_rank = _JUDGMENT_RANK.get(previous.get("judgment", "대응 없음"), 0)
        new_rank = _JUDGMENT_RANK.get(item.get("judgment", "대응 없음"), 0)
        if item.get("quote") and new_rank > old_rank:
            upgraded = dict(item)
            upgraded["precision_review"] = True
            existing[index] = upgraded


def _partition_core_first_elements(
    elements: List[ClaimElement],
) -> tuple[List[ClaimElement], List[ClaimElement]]:
    generic = [element for element in elements if _is_batchable_generic_element(element)]
    generic_labels = {normalize_label(element.label) for element in generic}
    core = [
        element for element in elements
        if normalize_label(element.label) not in generic_labels
    ]
    return core, generic

# 한국어 청구항과 영문 인용발명을 혼합 비교할 때, 한국어 토큰만으로 문헌을
# 압축하면 직접 대응하는 영문 실시예가 입력에서 통째로 빠질 수 있다. 자주 쓰이는
# 기능 축을 영문 검색어로 확장하되, 최종 대응 판단은 LLM이 원문 전체 문맥에서 한다.
_KO_EN_CLAIM_KEYWORD_GROUPS = (
    (("텍스트", "문자", "자막", "발화"),
     ("text", "subtitle", "caption", "transcript", "utterance", "sentence")),
    (("메타데이터", "메타 데이터"), ("metadata", "meta-data")),
    (("분할", "분리", "구획"), ("divid", "split", "segment", "boundary")),
    (("세그먼트",), ("segment", "block", "topic", "short")),
    (("연속", "인접", "이어"),
     ("adjacent", "continu", "consecutive", "temporal", "align", "similarity")),
    (("씬", "장면"), ("scene", "shot")),
    (("그룹", "병합", "묶"), ("group", "cluster", "merge", "align")),
    (("맥락", "문맥", "의미"), ("context", "semantic", "topic")),
    (("영상", "비디오", "미디어"), ("video", "media")),
    (("시청", "재생", "탐색"), ("view", "watch", "playback", "presentation", "navigation")),
    (("서비스", "제공", "검색"), ("service", "provide", "present", "search")),
)


def _budgets(engine: str) -> tuple[int, int, int, int]:
    return _ENGINE_BUDGETS.get((engine or "").lower(), _DEFAULT_BUDGET)


def _full_doc_text(doc: ExtractedDocument) -> str:
    chunks = _doc_chunks(doc)
    return "\n".join(f"{cid} {text}" for cid, text in chunks)


def select_candidate_doc_indices_for_elements(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
    max_docs: Optional[int] = None,
) -> List[int]:
    """Return prior documents that must be compared for a dependent claim.

    Dependent-claim citation chains are selected from the comparison cache. If a
    document is skipped here, it cannot later be chosen as the newly added
    rejection reference for that dependent claim. Therefore dependent claims
    compare against every uploaded prior document; RAG may still compact each
    document's text inside the comparison prompt, but it must not route whole
    documents out of the candidate set.
    """
    if not prior_docs:
        return []
    return list(range(len(prior_docs)))

# quote(인용문) 길이가 너무 길면 LLM에게 전달할 인용문 앞부분과 뒷부분만 남기고
# 중간부분을 생략한다. 이 길이를 넘으면 앞뒤로 잘라내고 줄임표(...)가
# 앞뒤에 붙도록 변환해서 ' ... '으로 대체한다.
# LLM이 직접 유니코드 말줄임표로 반환한 경우도 ASCII '...'으로 정규화한다.
_QUOTE_MAX_CHARS = 350
_QUOTE_HEAD_CHARS = 190
_QUOTE_TAIL_CHARS = 140
_ELLIPSIS = " ... "
_HIGH_JUDGMENTS = {"동일", "실질적 동일"}
_COMPOSITE_MISSING_RE = re.compile(
    r"(?:제\s*[12]\s*|최종|결합|조합|함께|모두|각각|별도|"
    r"산출|계산|판단|선택|전환|제어\s*로직|알고리즘|이미지\s*프레임|프레임\s*기반|"
    r"second|first|final|combine|combination|respectively|separate|frame|algorithm|logic)",
    re.IGNORECASE,
)
_NON_DISCLOSURE_RE = re.compile(
    r"(?:확인되지|명시되지|개시되지|부재|차이|불충분|추론|"
    r"not\s+disclosed|not\s+confirmed|missing|absent|insufficient|inferred)",
    re.IGNORECASE,
)
_TERMINOLOGY_ONLY_RE = re.compile(
    r"(?:용어|표현|명칭).{0,20}(?:"
    r"차이(?:만|\s*뿐|\s*불과)|"
    r"차이\s*외(?:에|에는)?\s*(?:실질적으로\s*)?(?:동일|같)"
    r")",
    re.IGNORECASE,
)


def _cap_judgment_for_coverage(
    judgment: str,
    directness: str,
    missing_limitations: list[str],
    reason: str,
) -> str:
    """Clamp over-optimistic judgments when sub-limitations are not directly covered.

    The comparison prompt may find a related paragraph and still overstate the final
    judgment.  This post-processor keeps the expert rule stable: identical/substantial
    identity is allowed only when every material sub-limitation and its relationship
    are directly supported by excerpts.
    """
    directness = (directness or "").strip().lower()
    missing_text = " ".join(missing_limitations or [])
    terminology_only = bool(_TERMINOLOGY_ONLY_RE.search(reason or ""))
    coverage_problem = bool(missing_limitations) or (
        not terminology_only and bool(_NON_DISCLOSURE_RE.search(reason or ""))
    )

    if directness == "absent":
        if judgment in {"동일", "실질적 동일", "일부 차이"}:
            return "일부 유사"
        return judgment

    if directness == "inferred" and judgment in _HIGH_JUDGMENTS:
        return "일부 차이"

    if not coverage_problem:
        return judgment

    if judgment in _HIGH_JUDGMENTS:
        if (
            len(missing_limitations or []) >= 2
            or _COMPOSITE_MISSING_RE.search(missing_text)
            or _COMPOSITE_MISSING_RE.search(reason or "")
        ):
            return "일부 유사"
        return "일부 차이"

    if judgment == "일부 차이" and (
        len(missing_limitations or []) >= 2
        or _COMPOSITE_MISSING_RE.search(missing_text)
    ):
        return "일부 유사"

    return judgment


def _judgment_adjustment_reason(
    llm_judgment: str,
    final_judgment: str,
    directness: str,
    missing_limitations: list[str],
    reason: str,
) -> str:
    """Return an auditable reason when local policy changes the LLM judgment."""
    if llm_judgment == final_judgment:
        return ""
    normalized_directness = (directness or "").strip().lower()
    if normalized_directness == "absent":
        return "directness_absent"
    if normalized_directness == "inferred" and llm_judgment in _HIGH_JUDGMENTS:
        return "directness_inferred"
    missing_text = " ".join(missing_limitations or [])
    if (
        len(missing_limitations or []) >= 2
        or _COMPOSITE_MISSING_RE.search(missing_text)
        or _COMPOSITE_MISSING_RE.search(reason or "")
    ):
        return "multiple_or_composite_missing_limitations"
    if missing_limitations or _NON_DISCLOSURE_RE.search(reason or ""):
        return "missing_or_undisclosed_limitation"
    return "coverage_policy_cap"


def _shorten_quote(quote: str) -> str:
    """길이가 너무 긴 인용문을 앞뒤로 잘라내고 중간에 ' ... '으로 대체한다."""
    q = (quote or "").strip().replace("…", "...")
    if len(q) <= _QUOTE_MAX_CHARS:
        return q
    head = q[:_QUOTE_HEAD_CHARS].rsplit(" ", 1)[0].rstrip() or q[:_QUOTE_HEAD_CHARS]
    tail = q[-_QUOTE_TAIL_CHARS:].split(" ", 1)[-1].lstrip() or q[-_QUOTE_TAIL_CHARS:]
    return f"{head}{_ELLIPSIS}{tail}"


def _normalize_evidence(raw_evidence: object, fallback_quote: str = "", fallback_chunk_id: str = "") -> list[dict]:
    """Normalize optional multi-paragraph evidence while preserving legacy quote behavior."""
    evidence: list[dict] = []
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            quote = _shorten_quote(str(item.get("quote", "") or ""))
            if not quote:
                continue
            evidence.append({
                "limitation": str(item.get("limitation", "") or "").strip(),
                "quote": quote,
                "quote_translation": _shorten_quote(str(item.get("quote_translation", "") or "")),
                "chunk_id": str(item.get("chunk_id", "") or "").strip(),
            })
            if len(evidence) >= 5:
                break

    if not evidence and fallback_quote:
        evidence.append({
            "limitation": "대표 근거",
            "quote": fallback_quote,
            "chunk_id": str(fallback_chunk_id or "").strip(),
        })
    return evidence


def _evidence_spans(raw_evidence: object) -> list[EvidenceSpan]:
    return [EvidenceSpan(**item) for item in _normalize_evidence(raw_evidence)]


def normalize_label(label: str) -> str:
    """구성요소 레이블을 알파벳 대문자 + 선택적 숫자(-숫자 형태)로 정규화한다.

    청구항 분해/LLM 비교에서 레이블이 'A', '(A) 방법', '(a)', 'A-1' 등 다양하게
    표기되더라도 동일 구성요소로 묶이도록 한다. 표기 변환에 따른 캐시 조회
    실패('일치 없음')로 인해 불필요한 LLM 호출이 발생하는 문제를 방지하기 위함이다."""
    raw = str(label or "").strip()
    unwrapped = re.sub(r"^[\s(\[{]+|[\s)\]}:：._-]+$", "", raw).strip().upper()
    if unwrapped in {"P", "PRE", "PREAMBLE", "전제부"}:
        return "P"
    if re.match(r"^[\s(\[{]*P[\s)\]}]*(?=$|\s|[:：._-])", raw, re.IGNORECASE):
        return "P"
    m = re.match(
        r"^[\s(\[{]*([A-Ja-j])\s*(?:-\s*(\d+))?[\s)\]}]*(?=$|\s|[:：._-])",
        raw,
    )
    if not m:
        return raw.upper()
    base = m.group(1).upper()
    return f"{base}-{m.group(2)}" if m.group(2) else base


_COMPARISON_LABELS = tuple("ABCDEFGHIJ")


def _comparison_safe_elements(elements: List[ClaimElement]) -> List[ClaimElement]:
    """Return elements with unique labels suitable for comparison prompts."""
    safe_elements: List[ClaimElement] = []
    used: set[str] = set()
    auto_idx = 0

    for elem in elements:
        label = normalize_label(elem.label)
        if not re.fullmatch(r"(?:P|[A-J](?:-\d+)?)", label or "") or label in used:
            while auto_idx < len(_COMPARISON_LABELS) and _COMPARISON_LABELS[auto_idx] in used:
                auto_idx += 1
            label = _COMPARISON_LABELS[auto_idx] if auto_idx < len(_COMPARISON_LABELS) else f"X{len(used) + 1}"
            auto_idx += 1
        used.add(label)
        if label != elem.label:
            safe_elements.append(elem.model_copy(update={"label": label}))
        else:
            safe_elements.append(elem)

    return safe_elements


def _build_doc_text(
    doc: ExtractedDocument,
    elements: Optional[List[ClaimElement]] = None,
    max_chars: Optional[int] = None,
    engine: str = "",
    settings: Optional[Settings] = None,
) -> str:
    """
    대응관계 텍스트를 LLM 입력용으로 최적화해 반환.

    우선순위:
    1. doc.paragraphs (파일번호[XXXX] 기준 문단/페이지별 분할, chunk_id 포함 가능)
    2. doc.raw_text 단순 텍스트 대안 (청크 없을 경우)

    max_chars: 이 함수에서 잘라낼 최대 길이 사용. 호출자가 직접 지정 시 사용.
    """
    chunks = _doc_chunks(doc)
    if not chunks:
        return ""

    if settings is not None:
        engine = settings.engine
    relevant_default, hard_default, _, _ = _budgets(engine)
    hard_limit = min(max_chars, hard_default) if max_chars else hard_default
    relevant_limit = min(max_chars, relevant_default) if max_chars else relevant_default

    full_text = "\n".join(f"{cid} {text}" for cid, text in chunks)
    if not elements:
        return full_text[:hard_limit]

    # RAG branch removed. Directly using full text or keyword context.

    if len(full_text) <= hard_limit:
        return full_text[:hard_limit]

    keywords = _claim_keywords(elements)
    if not keywords:
        return full_text[:hard_limit]

    scored = []
    for order, (chunk_id, text) in enumerate(chunks):
        lowered = text.lower()
        score = sum(1 for kw in keywords if kw in lowered)
        if score:
            scored.append((score, order, chunk_id, text))

    if not scored:
        logger.info(f"{doc.filename}: no keyword hits, using first {relevant_limit} chars")
        return full_text[:relevant_limit]

    selected_orders = {0}
    total = len(chunks[0][0]) + len(chunks[0][1]) + 2
    # break와 continue: 잘라낸 문단 앞에는 반드시 이전 문단의 연결 맥락이 필요하다.
    # 홀수 인덱스의 문단이 캐시에 없는 상황을 막기 위해, break를 쓰면 잘라낸 이후 문단으로
    # 건너뛰게 되어서 문단 연결이 깨지는 문제가 있었음.
    for score, order, _chunk_id, text in sorted(scored, key=lambda x: (-x[0], x[1])):
        item_len = len(text) + 20
        if total + item_len > relevant_limit:
            continue
        selected_orders.add(order)
        total += item_len

    selected = [
        f"{chunk_id} {text}"
        for order, (chunk_id, text) in enumerate(chunks)
        if order in selected_orders
    ]
    result = "\n".join(selected)
    logger.info(
        f"{doc.filename}: reduced LLM context {len(full_text)} -> {len(result)} chars "
        f"({len(selected)}/{len(chunks)} chunks)"
    )
    return result


def _doc_chunks(doc: ExtractedDocument) -> List[tuple[str, str]]:
    if doc.paragraphs:
        return [
            (para_id, text.strip())
            for para_id, text in doc.paragraphs.items()
            if text and text.strip()
        ]

    if doc.pages:
        chunks = []
        for page_num, page_text in doc.pages.items():
            text = (page_text or "").strip()
            if not text:
                continue
            for idx in range(0, len(text), _CHUNK_SIZE):
                chunk = text[idx:idx + _CHUNK_SIZE].strip()
                if chunk:
                    chunks.append((f"[P{page_num}-{idx // _CHUNK_SIZE + 1}]", chunk))
        return chunks

    raw = doc.raw_text or ""
    return [
        (f"[T{idx // _CHUNK_SIZE + 1}]", raw[idx:idx + _CHUNK_SIZE].strip())
        for idx in range(0, len(raw), _CHUNK_SIZE)
        if raw[idx:idx + _CHUNK_SIZE].strip()
    ]


def _build_hybrid_docs_block(
    prior_docs: List[ExtractedDocument],
    elements: List[ClaimElement],
    engine: str = "",
    settings: Optional[Settings] = None,
    total_budget_override: Optional[int] = None,
    min_doc_budget_override: Optional[int] = None,
) -> str:
    """Build one compact, chat-like comparison context from all prior documents."""
    if not prior_docs:
        return ""

    if settings is not None:
        engine = settings.engine
    _, _, hybrid_total, hybrid_min = _budgets(engine)
    mixed_mode = _comparison_mode(getattr(settings, "comparison_mode", "")) == "mixed"
    if mixed_mode:
        hybrid_total = min(hybrid_total, _MIXED_TOTAL_BUDGET)
        hybrid_min = min(hybrid_min, _MIXED_MIN_DOC_BUDGET)
    if total_budget_override is not None:
        hybrid_total = min(hybrid_total, max(1_000, int(total_budget_override)))
    if min_doc_budget_override is not None:
        hybrid_min = min(hybrid_min, max(500, int(min_doc_budget_override)))

    full_blocks = [
        f"[doc_index={doc_idx}] {doc.filename}\n{_full_doc_text(doc)}"
        for doc_idx, doc in enumerate(prior_docs)
    ]
    full_docs_block = "\n\n---\n\n".join(full_blocks)
    if not mixed_mode and len(full_docs_block) <= hybrid_total:
        logger.info(
            f"Hybrid comparison: using full text for all {len(prior_docs)} docs "
            f"({len(full_docs_block)} chars)"
        )
        return full_docs_block

    # The integrated mode must keep every cited document in the one prompt. If
    # the combined full text is too large, divide the input budget across all
    # documents and compact each one independently (RAG first, keyword fallback).
    separator_chars = len("\n\n---\n\n") * max(0, len(prior_docs) - 1)
    header_chars = sum(
        len(f"[doc_index={doc_idx}] {doc.filename}\n")
        for doc_idx, doc in enumerate(prior_docs)
    )
    available_text_chars = max(0, hybrid_total - separator_chars - header_chars)
    per_doc_budget = max(hybrid_min, available_text_chars // len(prior_docs))
    blocks = []
    for doc_idx, doc in enumerate(prior_docs):
        # per_doc_budget은 _build_doc_text 내부 제한값으로 직접 전달한다.
        # 예전에는 최종 결과를 [:per_doc_budget]으로 다시 잘랐지만,
        # 문서 순서대로 뽑은 텍스트가 뒤에서 한 번 더 잘리면 문맥 손실이 커질 수 있다.
        doc_text = _build_doc_text(
            doc,
            elements,
            max_chars=per_doc_budget,
            engine=engine,
            settings=settings,
        )
        blocks.append(
            f"[doc_index={doc_idx}] {doc.filename}\n"
            f"{doc_text}"
        )
    return "\n\n---\n\n".join(blocks)


def _claim_keywords(elements: List[ClaimElement]) -> List[str]:
    text = " ".join(e.text for e in elements)
    lowered_claim = text.lower()
    tokens = re.findall(r"[A-Za-z0-9가-힣]{2,}", text.lower())
    stopwords = {
        "하는", "하고", "하며", "포함", "포함하는", "구비", "구비하는", "상기",
        "및", "또는", "위해", "위한", "방법", "장치", "시스템", "단계",
        "the", "and", "for", "with", "that", "this", "from", "into", "wherein",
    }
    seen = set()
    keywords = []
    selection_structural_keywords: List[str] = []
    for element in elements:
        element_text = element.text or ""
        if _SELECTION_OR_RE.search(element_text) and _SELECTION_CONDITION_RE.search(element_text):
            selection_structural_keywords.extend(_SELECTION_STRUCTURE_KEYWORDS)

    # 선택식+조건/분기 구성은 개별 대안명(A/B)을 1차 검색축으로 삼으면
    # A만 강한 문헌 또는 B만 강한 문헌이 먼저 뽑힐 수 있다. 따라서 도메인
    # 중립적인 상위 구조 토큰을 앞에 두고, 원문 토큰은 보조 검색축으로 둔다.
    for token in selection_structural_keywords:
        token = token.lower()
        if token in stopwords or token in seen:
            continue
        seen.add(token)
        keywords.append(token)

    for triggers, expansions in _KO_EN_CLAIM_KEYWORD_GROUPS:
        if not any(trigger in lowered_claim for trigger in triggers):
            continue
        for token in expansions:
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)

    for token in tokens:
        if token in stopwords or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords[:80]


_SYSTEM_BATCH = """당신은 특허 구성대비 전문가입니다.
청구항 구성요소와 인용발명의 원문을 중립적으로 비교하십시오.
판정은 동일, 실질적 동일, 일부 차이, 일부 유사, 차이, 대응 없음 중 하나만 사용합니다.
quote는 인용발명의 원문을 그대로 인용하고, 판단_이유에는 대응점과 차이만 간결하게 적습니다.
분석 과정이나 설명문 없이 요청된 JSON 배열만 출력하십시오."""


# ---------------------------------------------------------------------------
# 인용 검증: 원문 문자열 대조만 수행 (LLM 호출 없음)
# ---------------------------------------------------------------------------

# 寃利??먯젙
_VERIFIED = "원문 확인"
_PARTIAL = "일부 일치(요약 또는 생략 가능성)"
_NOT_FOUND = "원문 미확인 — 인용문 검토 필요"
_EMPTY = "인용 없음"
_SHORT = "인용문이 너무 짧아 검증 불가"


def _probe_status(probe_text: str, corpus: str) -> Optional[str]:
    """발췌 구간이 corpus 안에 존재하는지 판정한다: 'verified' | 'partial' | None.
    앞 70자 완전 일치면 verified, 앞 30자 일치 또는 단어 60%+ 일치면 partial."""
    probe_full = probe_text[:70].lower()
    if probe_full and probe_full in corpus:
        return "verified"
    probe_short = probe_text[:30].lower()
    if probe_short and probe_short in corpus:
        return "partial"
    words = [w for w in probe_full.split() if len(w) >= 3]
    if words:
        ratio = sum(1 for w in words if w in corpus) / len(words)
        if ratio >= 0.60:
            return "partial"
    return None


def verify_quotes(
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    min_quote_len: int = 15,
) -> List[Dict]:
    """
    각 ElementMatch의 quote가 실제 문서에 존재하는지 문자열 검색으로 검증한다.
    LLM 호출 없이 즉시 수행한다.

    반환값: [{"label": "A", "status": "verified"|"partial"|"not_found"|"empty"|"short",
             "icon": "info", "message": "..."}]
    """
    results = []
    corpus_cache: Dict[int, str] = {}  # 문서 전체 텍스트 캐시 — 같은 문서를 1회만

    for m in matches:
        label = m.label
        quote = (m.quote or "").strip()

        if not quote:
            results.append({"label": label, "status": "empty",
                             "icon": "info", "message": f"({label}) {_EMPTY}"})
            continue

        if len(quote) < min_quote_len:
            results.append({"label": label, "status": "short",
                             "icon": "warning", "message": f"({label}) {_SHORT}: \"{quote}\""})
            continue

        # 해당 인용발명 문서 텍스트 수집
        doc_idx = m.cited_invention_index
        doc = prior_docs[doc_idx] if 0 <= doc_idx < len(prior_docs) else None
        if doc is None:
            results.append({"label": label, "status": "no_doc",
                             "icon": "warning", "message": f"({label}) 인용발명 문서를 찾을 수 없음"})
            continue

        # 검색 대상 텍스트는 paragraphs + pages + raw_text를 모두 합친다.
        if doc_idx not in corpus_cache:
            corpus_cache[doc_idx] = (
                " ".join(doc.paragraphs.values()) + " "
                + " ".join(doc.pages.values()) + " "
                + doc.raw_text
            ).lower()
        search_corpus = corpus_cache[doc_idx]

        # '...'으로 축약된 발췌문은 각 구간을 나눠 검증한다. 축약이 없으면 전체 인용문에
        # 대해 기존 규칙(앞 70/30자 + 단어 일치율)으로 동일하게 처리한다.
        segments = [
            s.strip() for s in re.split(r"\s*(?:…|\.{3,})\s*", quote)
            if len(s.strip()) >= min_quote_len
        ]
        if not segments:
            segments = [quote]

        seg_statuses = [_probe_status(seg, search_corpus) for seg in segments]
        if all(s == "verified" for s in seg_statuses):
            results.append({"label": label, "status": "verified",
                             "icon": "info", "message": f"({label}) {_VERIFIED}"})
        elif any(s in ("verified", "partial") for s in seg_statuses):
            found = sum(1 for s in seg_statuses if s in ("verified", "partial"))
            results.append({"label": label, "status": "partial",
                             "icon": "warning",
                             "message": f"({label}) {_PARTIAL} (인용 구간 {found}/{len(seg_statuses)} 확인)"})
        else:
            results.append({"label": label, "status": "not_found",
                             "icon": "info", "message": f"({label}) {_NOT_FOUND}"})

    return results


# ---------------------------------------------------------------------------
# 보고서 생성 단계: 캐시에서 로드
# ---------------------------------------------------------------------------

def load_comparisons(job_dir: str, doc_idx: int) -> Optional[Dict]:
    """저장된 비교 결과 로드"""
    path = Path(job_dir) / f"comparisons_{doc_idx}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid comparison cache %s: %s", path, exc)
        return None


def _comparison_mode(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"hybrid", "precision"}:
        return "hybrid"
    if normalized in {"mixed", "fast", "fast_hybrid"}:
        return "mixed"
    return "per_doc"


def _cache_is_compatible(
    cache: Optional[Dict],
    comparison_mode: Optional[str] = None,
) -> bool:
    if not cache:
        return False
    meta = cache.get(_CACHE_META_KEY, {})
    if meta.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return False
    if comparison_mode is not None:
        cached_mode = _comparison_mode(meta.get("comparison_mode", "per_doc"))
        if cached_mode != _comparison_mode(comparison_mode):
            return False
    return True


def reset_incompatible_comparison_caches(
    job_dir: str,
    num_docs: int,
    settings: Settings,
) -> bool:
    """Clear derived comparison caches when their input strategy changed.

    Cache metadata is stored per document, while a file can contain several
    claims. Clearing the whole derived cache prevents old per-document results
    from being mixed with new integrated-mode results in citation-chain scoring.
    """
    expected_mode = _comparison_mode(getattr(settings, "comparison_mode", "per_doc"))
    reset_any = False

    for doc_idx in range(num_docs):
        path = Path(job_dir) / f"comparisons_{doc_idx}.json"
        cache = load_comparisons(job_dir, doc_idx)
        if not cache:
            continue
        meta = cache.get(_CACHE_META_KEY, {})
        cached_mode = _comparison_mode(meta.get("comparison_mode", "per_doc"))
        incompatible = (
            meta.get("schema_version") != _CACHE_SCHEMA_VERSION
            or cached_mode != expected_mode
        )
        if not incompatible:
            continue

        fresh_cache = {
            _CACHE_META_KEY: {
                "schema_version": _CACHE_SCHEMA_VERSION,
                "comparison_mode": expected_mode,
            }
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(fresh_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
        reset_any = True

    return reset_any


def get_matches_from_cache(
    claim: ParsedClaim,
    prior_docs: List[ExtractedDocument],
    job_dir: str,
    allowed_docs: Optional[List[int]] = None,
    comparison_mode: Optional[str] = None,
) -> tuple[List[ElementMatch], bool]:
    """캐시에서 해당 청구항의 ElementMatch 목록을 반환한다.

    종속항도 청구항 번호를 캐시 키로 사용한다. 전달된 인용발명 비교 결과가 모두
    있으면 새로 분석하지 않고, 하나라도 없으면 해당 문헌만 다시 비교한다.
    """
    num_docs = len(prior_docs)
    claim_key = str(claim.claim_number)

    cached_doc_count = 0
    doc_results = []
    for doc_idx in range(num_docs):
        cache = load_comparisons(job_dir, doc_idx)
        if _cache_is_compatible(cache, comparison_mode) and claim_key in cache:
            doc_results.append(cache[claim_key])
            cached_doc_count += 1
        else:
            doc_results.append([])

    elements = _comparison_safe_elements(claim.elements)
    return _select_best_matches(elements, doc_results, num_docs, allowed_docs), cached_doc_count == num_docs


def get_cached_doc_indices(
    job_dir: str,
    claim_number: int,
    num_docs: int,
    comparison_mode: Optional[str] = None,
) -> set[int]:
    """Return active document indices that already have comparison cache for a claim."""
    claim_key = str(claim_number)
    cached: set[int] = set()
    for doc_idx in range(num_docs):
        cache = load_comparisons(job_dir, doc_idx)
        if _cache_is_compatible(cache, comparison_mode) and claim_key in cache:
            cached.add(doc_idx)
    return cached


async def analyze_claim_elements_for_docs(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    doc_indices: List[int],
    settings: Settings,
    job_dir: Optional[str] = None,
    claim_number: Optional[int] = None,
) -> None:
    """Compare one claim only against selected prior documents and cache the results.

    This is used when a refreshed job reuses comparison cache for unchanged PDFs and
    only newly added PDFs need an extra LLM comparison.
    """
    elements = _comparison_safe_elements(elements)
    for doc_idx in doc_indices:
        if doc_idx < 0 or doc_idx >= len(prior_docs):
            continue
        result = await _batch_judge_for_doc(elements, prior_docs[doc_idx], doc_idx, settings)
        if job_dir is not None and claim_number is not None:
            _merge_into_cache(job_dir, doc_idx, str(claim_number), result, settings)
            logger.info(
                f"[partial cache saved] comparisons_{doc_idx}.json claim {claim_number} "
                f"({len(result)} elements)"
            )


# ---------------------------------------------------------------------------
# 대응 분석: 캐시가 없으면 즉시 비교하고 결과를 캐시
# ---------------------------------------------------------------------------

async def analyze_claim_elements(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
    job_dir: Optional[str] = None,
    claim_number: Optional[int] = None,
) -> List[ElementMatch]:
    """구성요소를 인용발명별로 비교하고 필요하면 comparisons_{doc_idx}.json에 캐시한다."""
    elements = _comparison_safe_elements(elements)
    num_docs = len(prior_docs)
    doc_results = []
    for doc_idx in range(num_docs):
        result = await _batch_judge_for_doc(elements, prior_docs[doc_idx], doc_idx, settings)
        doc_results.append(result)

        if job_dir is not None and claim_number is not None:
            _merge_into_cache(job_dir, doc_idx, str(claim_number), result, settings)
            logger.info(
                f"[cache saved] comparisons_{doc_idx}.json claim {claim_number} "
                f"({len(result)} elements)"
            )

    return _select_best_matches(elements, doc_results, num_docs)


_CORE_SCREEN_SIMILARITY = {
    "동일": 1.00,
    "실질적 동일": 0.85,
    "일부 차이": 0.55,
    "일부 유사": 0.35,
    "차이": 0.15,
    "대응 없음": 0.00,
}


def _comparison_placeholder(element: ClaimElement) -> Dict:
    return {
        "label": element.label,
        "found": False,
        "judgment": "대응 없음",
        "quote": "",
        "quote_translation": "",
        "chunk_id": "",
        "판단_이유": "",
        "directness": "absent",
        "missing_limitations": [],
        "evidence": [],
        "not_evaluated": True,
        "evaluation_status": "not_evaluated_low_importance",
        "skip_reason": "주 인용발명 후보가 아닌 문헌의 단순 범용 구성 비교를 생략함",
    }


def _merge_flat_comparison_results(
    doc_results: List[List[Dict]],
    results: List[Dict],
    local_to_global: Optional[Dict[int, int]] = None,
) -> None:
    mapping = local_to_global or {}
    for item in results:
        try:
            local_idx = int(item.get("doc_index", 0))
        except (TypeError, ValueError):
            continue
        doc_idx = mapping.get(local_idx, local_idx)
        if doc_idx < 0 or doc_idx >= len(doc_results):
            continue
        label = normalize_label(item.get("label", ""))
        target_idx = next(
            (
                idx for idx, target in enumerate(doc_results[doc_idx])
                if normalize_label(target.get("label", "")) == label
            ),
            None,
        )
        if target_idx is None:
            continue
        stored = dict(item)
        stored.pop("doc_index", None)
        stored["not_evaluated"] = False
        stored["evaluation_status"] = "evaluated"
        stored.pop("skip_reason", None)
        doc_results[doc_idx][target_idx] = stored


def _core_primary_candidate_indices(
    core_results: List[Dict],
    core_elements: List[ClaimElement],
    num_docs: int,
) -> List[int]:
    """Keep the strongest core candidates and every possible novelty candidate."""
    by_doc: Dict[int, Dict[str, Dict]] = {idx: {} for idx in range(num_docs)}
    for item in core_results:
        try:
            doc_idx = int(item.get("doc_index", 0))
        except (TypeError, ValueError):
            continue
        if 0 <= doc_idx < num_docs:
            by_doc[doc_idx][normalize_label(item.get("label", ""))] = item

    ranked: List[tuple[float, float, int, int]] = []
    novelty_candidates: List[int] = []
    for doc_idx in range(num_docs):
        weighted_score = 0.0
        covered_weight = 0.0
        total_weight = 0.0
        possible_novelty = True
        for element in core_elements:
            weight = float(_importance_value(element.importance))
            total_weight += weight
            item = by_doc.get(doc_idx, {}).get(normalize_label(element.label), {})
            similarity = _CORE_SCREEN_SIMILARITY.get(item.get("judgment", "대응 없음"), 0.0)
            directness = str(item.get("directness") or "direct").strip().lower()
            if not item.get("quote") or directness == "absent":
                evidence_factor = 0.0
            elif directness == "inferred":
                evidence_factor = 0.65
            else:
                evidence_factor = 1.0
            weighted_score += weight * similarity * evidence_factor
            if similarity >= 0.55 and evidence_factor > 0:
                covered_weight += weight
            if not (
                item.get("judgment") in {"동일", "실질적 동일"}
                and bool(item.get("quote"))
                and directness in {"", "direct"}
                and not item.get("missing_limitations")
            ):
                possible_novelty = False
        score = weighted_score / total_weight if total_weight else 0.0
        coverage = covered_weight / total_weight if total_weight else 0.0
        ranked.append((score, coverage, -doc_idx, doc_idx))
        if possible_novelty:
            novelty_candidates.append(doc_idx)

    ranked.sort(reverse=True)
    selected = list(novelty_candidates)
    for _score, _coverage, _tie, doc_idx in ranked[:_CORE_FIRST_PRIMARY_CANDIDATES]:
        if doc_idx not in selected:
            selected.append(doc_idx)
    return selected or ([0] if num_docs else [])


async def _judge_core_first_mixed(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
) -> Optional[List[List[Dict]]]:
    """Run a compact core screen, then batch generic elements for primary candidates."""
    core_elements, generic_elements = _partition_core_first_elements(elements)
    matrix_cells = len(elements) * len(prior_docs)
    if (
        len(prior_docs) < _CORE_FIRST_MIN_DOCS
        or matrix_cells < _CORE_FIRST_MIN_MATRIX_CELLS
        or not core_elements
        or len(generic_elements) < 2
    ):
        return None

    core_results = await _batch_judge_hybrid(
        core_elements,
        prior_docs,
        settings,
        total_budget_override=_CORE_SCREEN_TOTAL_BUDGET,
        min_doc_budget_override=_CORE_SCREEN_MIN_DOC_BUDGET,
    )
    candidate_indices = _core_primary_candidate_indices(
        core_results,
        core_elements,
        len(prior_docs),
    )
    candidate_docs = [prior_docs[idx] for idx in candidate_indices]
    generic_results = await _batch_judge_hybrid(
        generic_elements,
        candidate_docs,
        settings,
        total_budget_override=_GENERIC_PRIMARY_TOTAL_BUDGET,
        min_doc_budget_override=_GENERIC_PRIMARY_MIN_DOC_BUDGET,
    )

    doc_results = [
        [_comparison_placeholder(element) for element in elements]
        for _ in prior_docs
    ]
    _merge_flat_comparison_results(doc_results, core_results)
    _merge_flat_comparison_results(
        doc_results,
        generic_results,
        {local_idx: doc_idx for local_idx, doc_idx in enumerate(candidate_indices)},
    )
    logger.info(
        "Core-first mixed comparison: %s core + %s generic elements, "
        "%s/%s documents received generic comparison",
        len(core_elements),
        len(generic_elements),
        len(candidate_indices),
        len(prior_docs),
    )
    return doc_results


async def analyze_claim_elements_hybrid(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
    job_dir: Optional[str] = None,
    claim_number: Optional[int] = None,
    doc_index_map: Optional[List[int]] = None,
    core_first: bool = False,
) -> List[ElementMatch]:
    """
    Compare one claim against all prior documents and cache the full matrix.

    The normal path uses one integrated call.  For sufficiently large independent
    claims in mixed mode, ``core_first`` may use a core-screening call followed by
    one generic-element call limited to the retained primary candidates.

    Both paths store an explicit per-document, per-element judgment matrix.
    Citation-chain scoring depends on comparisons_{doc_idx}.json representing
    each document's own coverage, not only the globally best document per element.
    """
    elements = _comparison_safe_elements(elements)
    num_docs = len(prior_docs)
    original_doc_indices = doc_index_map or list(range(num_docs))
    if num_docs <= 1:
        if not prior_docs:
            return _select_best_matches(elements, [], 0)
        original_idx = original_doc_indices[0] if original_doc_indices else 0
        result = await _batch_judge_for_doc(elements, prior_docs[0], original_idx, settings)
        if job_dir is not None and claim_number is not None:
            _merge_into_cache(job_dir, original_idx, str(claim_number), result, settings)
        return [
            match.model_copy(update={"cited_invention_index": original_idx})
            for match in _select_best_matches(elements, [result], 1)
        ]

    doc_results = [
        [
            {
                "label": elem.label,
                "found": False,
                "judgment": "대응 없음",
                "quote": "",
                "chunk_id": "",
                "판단_이유": "",
            }
            for elem in elements
        ]
        for _ in range(num_docs)
    ]

    try:
        staged_results = (
            await _judge_core_first_mixed(elements, prior_docs, settings)
            if core_first and _comparison_mode(getattr(settings, "comparison_mode", "")) == "mixed"
            else None
        )
        hybrid_results = None if staged_results is not None else await _batch_judge_hybrid(
            elements, prior_docs, settings
        )
    except CompareFailed as exc:
        logger.warning(
            "Hybrid comparison failed; falling back to per-document comparison: %s",
            exc,
        )
        for doc_idx, doc in enumerate(prior_docs):
            original_idx = original_doc_indices[doc_idx] if doc_idx < len(original_doc_indices) else doc_idx
            doc_results[doc_idx] = await _batch_judge_for_doc(elements, doc, original_idx, settings)
    except Exception as e:
        logger.error(f"Hybrid batch judge error: {e}")
        raise CompareFailed(f"하이브리드 구성대비 LLM 호출 실패: {e}") from e
    else:
        if staged_results is not None:
            doc_results = staged_results
        else:
            _merge_flat_comparison_results(doc_results, hybrid_results or [])

        # A syntactically complete integrated response can still overlook every
        # compound-feature passage in one source.  Retry only documents where a
        # single source chunk contains multiple technical axes from a missed
        # element; the retry can add evidence but never fabricates a match.
        for doc_idx, review_elements in _false_negative_review_candidates(
            elements, prior_docs, doc_results
        ):
            try:
                reviewed = await _batch_judge_for_doc(
                    review_elements,
                    prior_docs[doc_idx],
                    doc_idx,
                    settings,
                    precision_review=True,
                )
            except CompareFailed as exc:
                logger.warning(
                    "Precision review skipped for doc[%s] %s: %s",
                    doc_idx,
                    prior_docs[doc_idx].filename,
                    exc,
                )
                continue
            _merge_precision_review_results(doc_results[doc_idx], reviewed)
            logger.info(
                "Precision review completed for doc[%s] %s (%s elements)",
                doc_idx,
                prior_docs[doc_idx].filename,
                len(review_elements),
            )

    if job_dir is not None and claim_number is not None:
        for doc_idx, results in enumerate(doc_results):
            original_idx = original_doc_indices[doc_idx] if doc_idx < len(original_doc_indices) else doc_idx
            _merge_into_cache(job_dir, original_idx, str(claim_number), results, settings)
            logger.info(
                f"[hybrid cache saved] comparisons_{original_idx}.json claim {claim_number} "
                f"({len(results)} elements)"
            )

    matches = _select_best_matches(elements, doc_results, num_docs)
    return [
        match.model_copy(update={
            "cited_invention_index": original_doc_indices[match.cited_invention_index]
            if match.cited_invention_index < len(original_doc_indices)
            else match.cited_invention_index
        })
        for match in matches
    ]


def _merge_into_cache(
    job_dir: str,
    doc_idx: int,
    claim_key: str,
    results: List[Dict],
    settings: Optional[Settings] = None,
) -> None:
    """comparisons_{doc_idx}.json에 claim_key 결과를 병합 저장한다.
    기존 다른 청구항 캐시는 보존하고, 해당 키만 덮어쓴다."""
    path = Path(job_dir) / f"comparisons_{doc_idx}.json"
    cache: Dict = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    cache[claim_key] = results
    if settings is not None:
        cache[_CACHE_META_KEY] = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "comparison_mode": _comparison_mode(getattr(settings, "comparison_mode", "per_doc")),
        }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


async def _batch_judge_for_doc(
    elements: List[ClaimElement],
    doc: ExtractedDocument,
    doc_idx: int,
    settings: Settings,
    *,
    precision_review: bool = False,
) -> List[Dict]:
    full_text = _build_doc_text(doc, elements, engine=settings.engine, settings=settings)

    elements_text = "\n".join(f"({e.label}) {e.text}" for e in elements)

    prompt = render_prompt(
        "prompt_compare_single.txt",
        doc_filename=doc.filename,
        elements_text=elements_text,
        full_text=full_text,
        review_instruction=(
            "\n[정밀 재검증]\n"
            "앞선 통합 비교에서 복합 구성의 부분 대응이 누락될 수 있어, 이 문헌만 다시 검토합니다. "
            "구성 전체가 정확히 일치하지 않아도 조절·회로·고조파 관계 등 하나 이상의 기술적 근거가 "
            "원문에 있으면 found=true와 적절한 부분 판정을 사용하고, 남은 제한은 missing_limitations에 "
            "명시하십시오. 원문 근거가 전혀 없을 때만 대응 없음으로 하십시오.\n"
            if precision_review else ""
        ),
    )

    return await _call_and_parse_comparison(
        prompt,
        elements,
        settings,
        context=f"인용발명 {doc_idx + 1} 구성대비",
    )


# ---------------------------------------------------------------------------
# 하이브리드 비교
# ---------------------------------------------------------------------------

async def _batch_judge_hybrid(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
    total_budget_override: Optional[int] = None,
    min_doc_budget_override: Optional[int] = None,
) -> List[Dict]:
    docs_block = _build_hybrid_docs_block(
        prior_docs,
        elements,
        engine=settings.engine,
        settings=settings,
        total_budget_override=total_budget_override,
        min_doc_budget_override=min_doc_budget_override,
    )
    elements_text = "\n".join(f"({e.label}) {e.text}" for e in elements)
    doc_list = "\n".join(
        f"- doc_index={idx}: {doc.filename}"
        for idx, doc in enumerate(prior_docs)
    )

    prompt = render_prompt(
        "prompt_compare_hybrid.txt",
        doc_list=doc_list,
        elements_text=elements_text,
        docs_block=docs_block,
    )

    return await _call_and_parse_comparison(
        prompt,
        elements,
        settings,
        expected_doc_indices=list(range(len(prior_docs))),
        context="하이브리드 구성대비",
    )


async def _call_and_parse_comparison(
    prompt: str,
    elements: List[ClaimElement],
    settings: Settings,
    *,
    expected_doc_indices: Optional[List[int]] = None,
    context: str,
) -> List[Dict]:
    """Call the comparison model once and validate its recovered response locally."""
    system = load_prompt("system_compare.txt", _SYSTEM_BATCH)
    try:
        response = await call_ai(prompt, system, settings, agent="compare")
    except Exception as exc:
        raise CompareFailed(f"{context} LLM 호출 실패: {exc}") from exc

    try:
        return _parse_json_array(response, elements, expected_doc_indices)
    except CompareFailed as exc:
        raise CompareFailed(f"{context} 응답 형식 검증 실패: {exc}") from exc


def _select_best_matches(
    elements: List[ClaimElement],
    doc_results: List[List[Dict]],
    num_docs: int,
    allowed_docs: Optional[List[int]] = None,
) -> List[ElementMatch]:
    # allowed_docs가 있으면 보고서에서 채택한 인용발명만 후보로 사용한다.
    # 없으면 전체 문서에서 선택한다(준비/비교 단계 기본 동작).
    fallback_idx = allowed_docs[0] if allowed_docs else 0
    # Primary document first so it wins ties ??doc[0] is not always the primary.
    if allowed_docs:
        priority_order = [d for d in allowed_docs if d < num_docs]
    else:
        priority_order = list(range(num_docs))
    matches = []
    for elem in elements:
        best_match, best_rank, best_doc_idx = None, -1, fallback_idx
        for doc_idx in priority_order:
            if doc_idx >= len(doc_results):
                continue
            item = next(
                (m for m in doc_results[doc_idx]
                 if normalize_label(m.get("label")) == normalize_label(elem.label)),
                None,
            )
            if item is None:
                continue
            rank = _JUDGMENT_RANK.get(item.get("judgment", "대응 없음"), 0)
            if rank > best_rank:
                best_rank, best_match, best_doc_idx = rank, item, doc_idx

        if best_match and best_rank > 0:
            matches.append(ElementMatch(
                label=elem.label,
                found=bool(best_match.get("found", False)),
                quote=_shorten_quote(best_match.get("quote", "")),
                quote_translation=_shorten_quote(best_match.get("quote_translation", "")),
                chunk_id=best_match.get("chunk_id", ""),
                judgment=best_match.get("judgment", "대응 없음"),
                cited_invention_index=best_doc_idx,
                similarity_reason=best_match.get("판단_이유", best_match.get("similarity_reason", "")),
                evidence=_evidence_spans(best_match.get("evidence", [])),
                directness=best_match.get("directness", "direct" if best_match.get("quote") else "absent"),
                missing_limitations=best_match.get("missing_limitations", []),
                motivation_quote=_shorten_quote(best_match.get("motivation_quote", "")),
                combination_risk=best_match.get("combination_risk", "uncertain"),
                combination_risk_reason=best_match.get("combination_risk_reason", ""),
            ))
        else:
            matches.append(ElementMatch(
                label=elem.label, found=False, quote="", chunk_id="",
                judgment="대응 없음", cited_invention_index=fallback_idx, similarity_reason="",
                directness="absent", missing_limitations=[],
            ))
    return matches


def _extract_json_payloads(text: str) -> List[Dict]:
    """Extract top-level JSON arrays or objects from an LLM response.

    Some models return a single object such as {"comparisons": [...]} instead of
    a bare array.  Looking only for arrays can accidentally capture nested
    evidence arrays, so decode either object or array from the earliest JSON
    boundary and let the comparison expander decide what is a real judgment.
    """
    decoder = json.JSONDecoder()
    payloads: List[Dict] = []
    idx = 0
    while idx < len(text):
        starts = [pos for pos in (text.find("[", idx), text.find("{", idx)) if pos != -1]
        if not starts:
            break
        start = min(starts)
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue

        if isinstance(value, list):
            payloads.extend(item for item in value if isinstance(item, dict))
        elif isinstance(value, dict):
            payloads.append(value)
        idx = end
    return payloads


def _copy_first_present(item: Dict, canonical_key: str, aliases: List[str]) -> None:
    if canonical_key in item:
        return
    for alias in aliases:
        if alias in item:
            item[canonical_key] = item[alias]
            return


def _canonicalize_comparison_item(
    item: Dict,
    parent_doc_index: Optional[object] = None,
) -> Dict:
    normalized = dict(item)
    if parent_doc_index is not None and "doc_index" not in normalized:
        normalized["doc_index"] = parent_doc_index
    _copy_first_present(
        normalized,
        "doc_index",
        ["document_index", "cited_invention_index", "prior_art_index", "인용발명_index", "문헌_index"],
    )
    _copy_first_present(
        normalized,
        "label",
        ["claim_element", "element", "element_label", "구성요소", "구성요소_label"],
    )
    _copy_first_present(
        normalized,
        "quote",
        ["citation", "excerpt", "quoted_text", "근거", "인용문", "원문"],
    )
    _copy_first_present(
        normalized,
        "chunk_id",
        ["chunk", "chunkId", "paragraph", "paragraph_id", "location", "문단", "위치"],
    )
    _copy_first_present(
        normalized,
        "judgment",
        ["판정", "판단", "comparison_judgment"],
    )
    _copy_first_present(
        normalized,
        "판단_이유",
        ["판단 이유", "판단이유", "reason", "judgment_reason", "comparison_reason"],
    )
    _copy_first_present(
        normalized,
        "found",
        ["matched", "is_found", "disclosed", "대응여부"],
    )
    return normalized


def _label_from_mapping_key(key: object) -> str:
    label = normalize_label(str(key or ""))
    return label if re.fullmatch(r"(?:P|[A-J](?:-\d+)?)", label) else ""


def _expand_comparison_items(
    items: List[Dict],
    parent_doc_index: Optional[object] = None,
) -> List[Dict]:
    expanded: List[Dict] = []
    nested_keys = (
        "results", "comparisons", "items", "matches", "elements",
        "documents", "docs", "document_results", "judgments",
        "판정", "구성대비", "비교결과", "문헌별_결과", "인용발명별_결과",
        "구성요소별_판정", "구성요소별결과",
    )

    for item in items:
        normalized = _canonicalize_comparison_item(item, parent_doc_index)
        child_doc_index = normalized.get("doc_index", parent_doc_index)
        child_items: List[Dict] = []
        for key in nested_keys:
            nested = item.get(key)
            if isinstance(nested, list):
                child_items.extend(value for value in nested if isinstance(value, dict))
            elif isinstance(nested, dict):
                child_items.append(nested)

        for key, value in item.items():
            label = _label_from_mapping_key(key)
            if not label:
                continue
            if isinstance(value, dict):
                child = dict(value)
                child.setdefault("label", label)
                child_items.append(child)
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, dict):
                        child = dict(child)
                        child.setdefault("label", label)
                        child_items.append(child)

        if child_items:
            expanded.extend(_expand_comparison_items(child_items, child_doc_index))

        has_comparison_fields = {
            "label", "claim_element", "found", "quote", "judgment", "판단_이유", "similarity_reason"
        }.intersection(normalized)
        if has_comparison_fields:
            expanded.append(normalized)

    return expanded


def _parse_json_array(
    response: str,
    elements: List[ClaimElement],
    expected_doc_indices: Optional[List[int]] = None,
) -> List[Dict]:
    text = re.sub(r"```(?:json)?", "", response.strip()).replace("```", "").strip()
    parsed = _extract_json_payloads(text)
    if not parsed:
        raise CompareFailed(
            f"구성대비 응답에서 JSON 배열 또는 객체를 찾지 못했습니다. 응답 길이: {len(response)}자"
        )
    parsed = _expand_comparison_items(parsed)

    expected_labels = {normalize_label(element.label) for element in elements}
    expected_docs = set(expected_doc_indices or [])
    normalized: List[Dict] = []
    invalid_reasons: List[str] = []
    required_fields = {"label", "found", "quote", "chunk_id", "judgment"}
    judgment_aliases = {
        "부분 차이": "일부 차이",
        "부분 유사": "일부 유사",
        "유사": "일부 유사",
        "없음": "대응 없음",
    }

    for item in parsed:
        schema_markers = {
            "label", "claim_element", "found", "judgment", "doc_index", "판단_이유", "similarity_reason"
        }
        if not schema_markers.intersection(item):
            continue

        missing_fields = required_fields.difference(item)
        if "판단_이유" not in item and "similarity_reason" not in item:
            missing_fields.add("판단_이유")
        if expected_doc_indices is not None and "doc_index" not in item:
            missing_fields.add("doc_index")
        if missing_fields:
            invalid_reasons.append(
                "필수 필드 누락: " + ", ".join(sorted(missing_fields))
            )
            continue

        label = normalize_label(str(item.get("label", "")))
        if not label or (expected_labels and label not in expected_labels):
            if "claim_element" in item and "label" not in item:
                invalid_reasons.append("claim_element 대신 label 필드를 사용해야 함")
            else:
                invalid_reasons.append(f"알 수 없는 label: {item.get('label', '')!r}")
            continue

        doc_idx: Optional[int] = None
        if expected_doc_indices is not None:
            try:
                doc_idx = int(item.get("doc_index"))
            except (TypeError, ValueError):
                invalid_reasons.append(f"{label}의 doc_index가 정수가 아님")
                continue
            if doc_idx not in expected_docs:
                invalid_reasons.append(f"{label}의 doc_index가 범위를 벗어남: {doc_idx}")
                continue

        llm_judgment = str(item.get("judgment", "대응 없음")).strip()
        llm_judgment = judgment_aliases.get(llm_judgment, llm_judgment)
        if llm_judgment not in _JUDGMENT_RANK:
            invalid_reasons.append(f"{label}의 허용되지 않은 judgment: {llm_judgment!r}")
            continue

        quote = _shorten_quote(str(item.get("quote", "") or ""))
        quote_translation = _shorten_quote(str(item.get("quote_translation", "") or ""))
        evidence = _normalize_evidence(item.get("evidence", []), quote, item.get("chunk_id", ""))
        directness = str(item.get("directness", "") or "").strip().lower()
        if directness not in {"direct", "inferred", "absent"}:
            directness = "direct" if quote else "absent"
        raw_missing = item.get("missing_limitations", [])
        if isinstance(raw_missing, str):
            missing_limitations = [raw_missing.strip()] if raw_missing.strip() else []
        elif isinstance(raw_missing, list):
            missing_limitations = [
                str(value).strip() for value in raw_missing if str(value).strip()
            ][:5]
        else:
            missing_limitations = []
        reason = str(item.get("판단_이유", item.get("similarity_reason", "")) or "")
        purpose_effect_similarity = re.sub(
            r"\s+",
            " ",
            str(item.get("purpose_effect_similarity", "") or "").strip(),
        )[:240]
        motivation_quote = _shorten_quote(str(item.get("motivation_quote", "") or ""))
        combination_risk = str(item.get("combination_risk", "uncertain") or "uncertain").strip().lower()
        if combination_risk not in {"none_explicit", "uncertain", "contrary_teaching", "principle_change"}:
            combination_risk = "uncertain"
        combination_risk_reason = str(item.get("combination_risk_reason", "") or "").strip()
        terminology_only = bool(re.search(
            r"(?:용어|표현|명칭).{0,20}차이(?:만|에\s*불과)",
            reason,
        ))
        implicit_difference = not terminology_only and bool(re.search(
            r"(?:확인되지|명시되지|개시되지|부재|차이|불충분|추론)",
            reason,
        ))
        # 객관적 차이, 추론 의존, 핵심 하위 제한 누락이 있으면 과도한 판정을 상한 처리한다.
        judgment = _cap_judgment_for_coverage(
            llm_judgment,
            directness,
            missing_limitations,
            reason,
        )
        judgment_adjustment_reason = _judgment_adjustment_reason(
            llm_judgment,
            judgment,
            directness,
            missing_limitations,
            reason,
        )
        found_value = item.get("found", False)
        if isinstance(found_value, str):
            found = found_value.strip().lower() in {"true", "1", "yes"}
        else:
            found = bool(found_value)
        if found and not quote:
            invalid_reasons.append(f"{label}의 found=true 항목에 quote가 없음")
            continue
        if not found and (quote or judgment != "대응 없음"):
            invalid_reasons.append(
                f"{label}의 found=false 항목은 빈 quote와 대응 없음 판정이어야 함"
            )
            continue

        normalized_item = dict(item)
        normalized_item.update({
            "label": label,
            "found": found,
            "quote": quote,
            "quote_translation": quote_translation if quote else "",
            "chunk_id": str(item.get("chunk_id", "") or ""),
            "judgment": judgment,
            "llm_judgment": llm_judgment,
            "judgment_adjusted": judgment != llm_judgment,
            "judgment_adjustment_reason": judgment_adjustment_reason,
            "판단_이유": reason,
            "purpose_effect_similarity": purpose_effect_similarity,
            "evidence": evidence if found else [],
            "directness": directness,
            "missing_limitations": missing_limitations,
            "motivation_quote": motivation_quote,
            "combination_risk": combination_risk,
            "combination_risk_reason": combination_risk_reason,
        })
        if doc_idx is not None:
            normalized_item["doc_index"] = doc_idx
        normalized.append(normalized_item)

    if not normalized:
        detail = f" ({'; '.join(invalid_reasons[:3])})" if invalid_reasons else ""
        raise CompareFailed(f"구성대비 응답에 유효한 구성요소 판정이 없습니다.{detail}")

    deduped: Dict[tuple, Dict] = {}
    for item in normalized:
        key = (
            item.get("doc_index"),
            item["label"],
        ) if expected_doc_indices is not None else item["label"]
        current = deduped.get(key)
        if current is None:
            deduped[key] = item
            continue

        current_rank = _JUDGMENT_RANK.get(current.get("judgment", "대응 없음"), 0)
        item_rank = _JUDGMENT_RANK.get(item.get("judgment", "대응 없음"), 0)
        current_quote_len = len(current.get("quote", "") or "")
        item_quote_len = len(item.get("quote", "") or "")
        if (item_rank, item_quote_len) > (current_rank, current_quote_len):
            deduped[key] = item

    normalized = list(deduped.values())

    if expected_doc_indices is None:
        expected_keys = expected_labels
        actual_keys = {item["label"] for item in normalized}
        missing_keys = sorted(expected_keys - actual_keys)
        missing_text = ", ".join(missing_keys)
    else:
        expected_keys = {
            (doc_idx, label)
            for doc_idx in expected_doc_indices
            for label in expected_labels
        }
        actual_keys = {(item["doc_index"], item["label"]) for item in normalized}
        missing_pairs = sorted(expected_keys - actual_keys)
        missing_text = ", ".join(
            f"doc_index={doc}/label={label}" for doc, label in missing_pairs
        )

    if missing_text or invalid_reasons:
        details = []
        if missing_text:
            details.append(f"누락: {missing_text}")
        if invalid_reasons:
            details.append("형식 오류: " + "; ".join(invalid_reasons[:3]))
        raise CompareFailed("구성대비 응답이 완전하지 않습니다. " + " / ".join(details))
    return normalized
