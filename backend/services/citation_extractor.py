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
_CACHE_SCHEMA_VERSION = 26
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
_NON_PATENT_RELEVANT_TEXT_MAX_CHARS = 30_000
_MIN_SUBSTANTIVE_NON_PATENT_QUOTE_CHARS = 40
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
    "three_dimensional_model": re.compile(
        r"3\s*D|3차원|삼차원|three[- ]?dimensional|3-dimensional|"
        r"\bmesh(?:es)?\b|메시|point\s+cloud|점군",
        re.IGNORECASE,
    ),
    "cad_design": re.compile(
        r"\bCAD\b|computer[- ]aided\s+design|캐드|설계\s*표현|design\s+representation",
        re.IGNORECASE,
    ),
    "visual_extraction": re.compile(
        r"이미지|영상|비전|시각|윤곽|형상|image|vision|visual|contour|shape|"
        r"extract(?:ion|ing|ed)?|추출",
        re.IGNORECASE,
    ),
    "multimodal_fusion": re.compile(
        r"종합|융합|결합|통합|fuse|fused|fusion|combin(?:e|ed|ing|ation)|"
        r"integrat(?:e|ed|ing|ion)|multimodal|multi-modal",
        re.IGNORECASE,
    ),
    "structured_instruction": re.compile(
        r"구조화된\s*(?:명령|지시)|기술\s*지침|작업\s*계획|"
        r"structured\s+(?:command|instruction)|technical\s+instruction|"
        r"working\s+plan|spec(?:ification)?",
        re.IGNORECASE,
    ),
    "conditional_processing": re.compile(
        r"필요한\s*경우|필요\s*여부|조건|판단|분기|"
        r"if\s+(?:needed|required|necessary)|determin(?:e|es|ed|ing)|"
        r"condition|branch|when\s+required",
        re.IGNORECASE,
    ),
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
    "coordinate_transform": re.compile(
        r"좌표(?:계|축)?|좌표\s*변환|coordinate(?:\s+system|\s+axis)?|"
        r"transform(?:ation)?|target\s+coordinate",
        re.IGNORECASE,
    ),
    "camera_control": re.compile(
        r"카메라.*(?:구동|제어|방향|팬|틸트)|(?:팬|틸트).*카메라|"
        r"camera.*(?:control|direction|pan|tilt)|(?:pan|tilt).*camera",
        re.IGNORECASE,
    ),
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


_CORE_RELATION_RE = re.compile(
    r"(?:동작\s*주파수|작동\s*주파수|운용\s*주파수|주파수.*(?:조정|조절|변경)|"
    r"(?:조정|조절|변경).*주파수|파라미터.*(?:되도록|하도록)|"
    r"(?:통해|따라|기초하여|기반하여|대응하여).*(?:조정|조절|변경|생성|제어)|"
    r"(?:조정|조절|변경|생성|제어).*(?:통해|따라|기초하여|기반하여|대응하여)|"
    r"operat(?:ing|ion)\s+frequency|frequency.*(?:adjust|vary|control)|"
    r"(?:adjust|vary|control).*(?:frequency|parameter))",
    re.IGNORECASE,
)


def _core_focus_text(elements: List[ClaimElement]) -> str:
    labels = [
        str(element.label)
        for element in elements
        if _CORE_RELATION_RE.search(" ".join((element.text or "").split()))
    ]
    if not labels:
        return "별도 고정 관계 없음. 각 구성의 구조·입력·처리·출력 관계를 기준으로 판단."
    return (
        "중심 쟁점 구성: "
        + ", ".join(labels)
        + ". 이 구성들의 원인·제어변수·결과 관계를 분리하거나 일반 구성의 명칭만으로 대체하지 말 것."
    )


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
    """Measure whether a short source-paragraph chain carries the technical axes.

    The score is intentionally only a *review trigger*.  It does not produce a
    judgment and never substitutes for a model-provided quotation.
    """
    expected = _technical_concepts(element.text)
    expected.discard("signal_path")  # input/output alone is not a differentiating clue.
    if len(expected) < 2:
        return 0.0

    chunks = _doc_chunks(doc)
    best = 0.0
    for index in range(len(chunks)):
        # 변환 관계가 한 단락, 좌표 계산이 다음 단락, 보상·제어가 그 다음
        # 단락에 이어지는 특허 서술을 하나의 기능사슬로 검토한다.
        window_text = "\n".join(
            chunks[position][1]
            for position in range(index, min(len(chunks), index + 3))
        )
        found = _technical_concepts(window_text)
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
    relevant passage in one document. We retry a document where up to three
    consecutive source paragraphs carry the compound element's technical axes
    and the integrated response either missed it or left a substantive gap.
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
            if (
                item.get("judgment") in {"동일", "실질적 동일"}
                and not item.get("missing_limitations")
            ):
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
    """Keep a recheck when it improves support quality or the judgment."""
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
        old_issues = len(previous.get("quality_issues") or [])
        new_issues = len(item.get("quality_issues") or [])
        quality_improved = old_issues > 0 and new_issues < old_issues
        if item.get("quote") and (new_rank > old_rank or quality_improved):
            upgraded = dict(item)
            upgraded["precision_review"] = True
            existing[index] = upgraded


_NON_PATENT_ACTION_RE = re.compile(
    r"(?:generate|generated|generates|extract|extracted|extracts|process|processed|"
    r"encode|encoded|fuse|fused|combine|combined|transform|transformed|create|created|"
    r"receive|received|use|used|based\s+on|input|output|생성|추출|처리|인코딩|"
    r"융합|결합|변환|수신|입력|출력|기초하여|기반으로)",
    re.IGNORECASE,
)


def _apply_non_patent_evidence_quality(results: List[Dict], doc: ExtractedDocument) -> List[Dict]:
    """Annotate non-patent evidence quality without making another model call."""
    if doc.document_type != "non_patent":
        return results
    for item in results:
        issues: List[str] = []
        if item.get("found"):
            quote = _normalize_verbatim_text(item.get("quote", ""))
            evidence = item.get("evidence") or []
            structured_roles = {
                role
                for span in evidence
                for role in ("subject", "input", "process", "output", "condition", "relationship")
                if str(span.get(role, "") or "").strip()
            }
            if len(quote) < _MIN_SUBSTANTIVE_NON_PATENT_QUOTE_CHARS:
                issues.append("non_patent_quote_too_short")
            if not _NON_PATENT_ACTION_RE.search(quote):
                issues.append("non_patent_quote_lacks_action")
            if (
                item.get("judgment") in {"동일", "실질적 동일", "일부 차이"}
                and len(structured_roles & {"input", "process", "output", "condition", "relationship"}) < 2
            ):
                issues.append("non_patent_evidence_chain_incomplete")
        item["quality_issues"] = issues
        item["evidence_quality"] = "needs_review" if issues else "verified"
        item["analysis_status"] = "needs_review" if issues else "evaluated"
    return results


def _non_patent_quality_review_candidates(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    doc_results: List[List[Dict]],
) -> List[tuple[int, List[ClaimElement]]]:
    candidates: List[tuple[int, List[ClaimElement]]] = []
    by_label = {normalize_label(element.label): element for element in elements}
    for doc_idx, doc in enumerate(prior_docs):
        if doc.document_type != "non_patent" or doc_idx >= len(doc_results):
            continue
        review = [
            by_label[normalize_label(item.get("label", ""))]
            for item in doc_results[doc_idx]
            if item.get("quality_issues")
            and normalize_label(item.get("label", "")) in by_label
        ]
        if review:
            candidates.append((doc_idx, review))
    return candidates


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

# 한국어 청구항과 외국어 인용발명을 혼합 비교할 때, 한국어 토큰만으로 문헌을
# 압축하면 직접 대응하는 실시예가 입력에서 통째로 빠질 수 있다. 자주 쓰이는
# 기능 축을 영어·일본어·중국어 검색어로 확장하되, 최종 대응 판단은 LLM이 원문
# 전체 문맥에서 한다. 일본어의 신자체/표기 변형과 중국어의 간체/번역 변형을 함께
# 넣어 JP/CN 공보의 기계 번역 여부와 무관하게 관련 청크를 회수한다.
_KO_MULTILINGUAL_CLAIM_KEYWORD_GROUPS = (
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
    (("교정", "보정", "조정", "튜닝"),
     ("calibrat", "adjust", "tune", "較正", "校正", "補正", "調整",
      "校准", "标定", "调节")),
    (("작동 주파수", "동작 주파수", "운용 주파수"),
     ("operating frequency", "operation frequency", "動作周波数", "作動周波数",
      "工作频率", "操作频率", "运行频率")),
    (("공기 펄스", "에어 펄스"),
     ("air pulse", "air-pulse", "空気パルス", "空気圧パルス",
      "空气脉冲", "气流脉冲", "气压脉冲")),
    (("음압", "음압 레벨"),
     ("sound pressure", "sound pressure level", "spl", "音圧", "音圧レベル",
      "声压", "声压级")),
    (("메모리", "저장", "기억"),
     ("memory", "storage", "store", "メモリ", "記憶", "保存", "格納",
      "存储器", "存储", "内存", "保存")),
    (("사운드", "소리", "음향", "오디오"),
     ("sound", "audio", "acoustic", "サウンド", "音響", "音声",
      "声音", "音频", "声学")),
    (("생성", "발생"),
     ("generat", "produc", "生成", "発生", "产生")),
    (("일치", "동일", "같은"),
     ("match", "same", "equal", "一致", "同一", "等しい", "相同", "等于")),
    (("다른", "상이", "별개", "독립"),
     ("different", "distinct", "separate", "異なる", "相違", "別個", "独立",
      "不同", "相异", "单独", "独立")),
    (("모듈", "디바이스", "장치"),
     ("module", "device", "モジュール", "デバイス", "装置", "模块", "设备", "装置")),
)


def _budgets(engine: str) -> tuple[int, int, int, int]:
    return _ENGINE_BUDGETS.get((engine or "").lower(), _DEFAULT_BUDGET)


def _full_doc_text(doc: ExtractedDocument) -> str:
    chunks = _doc_chunks(doc)
    return "\n".join(f"{cid} {text}" for cid, text in chunks)


_NON_DESCRIPTION_PAGE_RE = re.compile(
    r"(?im)^\s*(?:초록|요약|abstract|summary|what\s+is\s+claimed\s*:?)\s*$"
    r"|^\s*(?:특허)?청구(?:의)?\s*범위\s*$"
)


def _page_is_non_description(page_text: str) -> bool:
    """문단번호가 없는 fallback 페이지의 초록/요약/청구항 여부를 판정한다."""
    text = page_text or ""
    if _NON_DESCRIPTION_PAGE_RE.search(text):
        return True
    return bool(re.search(r"(?im)^\s*claims\s*:?\s*$", text))


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
_MINIMUM_LIMIT_RE = re.compile(r"(?:최소|minimum|at\s+least|lower\s+bound)", re.IGNORECASE)
_AVERAGE_ONLY_RE = re.compile(r"(?:평균|average|mean)", re.IGNORECASE)
_MINIMUM_EVIDENCE_RE = re.compile(
    r"(?:최소|minimum|at\s+least|not\s+less\s+than|lower\s+bound)",
    re.IGNORECASE,
)
_EXECUTED_ADJUSTMENT_RE = re.compile(
    r"(?:조정하여|수정하여|변경하여|adjust(?:ing|s|ed)?|modif(?:y|ies|ied)|"
    r"chang(?:e|es|ed|ing)|apply|applies|applied|perform|execute)",
    re.IGNORECASE,
)
_RECOMMENDATION_ONLY_RE = re.compile(
    r"(?:권고|추천|제안|recommend|suggest|propose)",
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
    terminology_only = bool(_TERMINOLOGY_ONLY_RE.search(reason or ""))
    coverage_problem = bool(missing_limitations) or (
        not terminology_only and bool(_NON_DISCLOSURE_RE.search(reason or ""))
    )

    if directness == "absent":
        if judgment in {"동일", "실질적 동일", "일부 차이"}:
            return "일부 유사"
        return judgment

    if not coverage_problem:
        return judgment

    if directness == "inferred" and judgment in _HIGH_JUDGMENTS:
        return "일부 차이"

    if judgment in _HIGH_JUDGMENTS:
        if len(missing_limitations or []) >= 2:
            return "일부 유사"
        return "일부 차이"

    if judgment == "일부 차이" and len(missing_limitations or []) >= 2:
        return "일부 유사"

    return judgment


def _reconcile_judgment_with_reason(
    judgment: str,
    directness: str,
    missing_limitations: list[str],
    reason: str,
    quote: str,
) -> str:
    """좁은 범위의 판정 라벨·판단 이유 모순을 해소한다."""
    if (
        judgment == "일부 차이"
        and (directness or "").strip().lower() == "direct"
        and not missing_limitations
        and bool((quote or "").strip())
        and (
            "실질적으로 동일" in (reason or "")
            or _TERMINOLOGY_ONLY_RE.search(reason or "")
        )
    ):
        return "실질적 동일"
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
    if (
        llm_judgment == "일부 차이"
        and final_judgment == "실질적 동일"
        and not missing_limitations
    ):
        return "reason_label_reconciliation"
    normalized_directness = (directness or "").strip().lower()
    if normalized_directness == "absent":
        return "directness_absent"
    if normalized_directness == "inferred" and llm_judgment in _HIGH_JUDGMENTS:
        return "directness_inferred"
    if len(missing_limitations or []) >= 2:
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


_EMPTY_MISSING_LIMITATION_MARKERS = {
    "",
    "-",
    "n/a",
    "na",
    "none",
    "null",
    "없음",
    "해당 없음",
    "차이 없음",
    "누락 없음",
}


def _normalize_missing_limitations(raw_missing) -> list[str]:
    """모델이 빈 누락 목록을 자연어 표지로 반환한 경우 실제 공백으로 정규화한다."""
    if isinstance(raw_missing, str):
        values = [raw_missing]
    elif isinstance(raw_missing, list):
        values = raw_missing
    else:
        return []

    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        marker = re.sub(r"[.\s]+$", "", text).strip().lower()
        if marker in _EMPTY_MISSING_LIMITATION_MARKERS:
            continue
        if text:
            normalized.append(text)
    return normalized[:5]


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
                "subject": str(item.get("subject", "") or "").strip(),
                "input": str(item.get("input", "") or "").strip(),
                "process": str(item.get("process", "") or "").strip(),
                "output": str(item.get("output", "") or "").strip(),
                "condition": str(item.get("condition", "") or "").strip(),
                "relationship": str(item.get("relationship", "") or "").strip(),
                "quote": quote,
                "quote_translation": _shorten_quote(str(item.get("quote_translation", "") or "")),
                "chunk_id": str(item.get("chunk_id", "") or "").strip(),
                "page": item.get("page"),
                "section": str(item.get("section", "") or "").strip(),
            })
            if len(evidence) >= 5:
                break

    if not evidence and fallback_quote:
        evidence.append({
            "limitation": "대표 근거",
            "subject": "",
            "input": "",
            "process": "",
            "output": "",
            "condition": "",
            "relationship": "",
            "quote": fallback_quote,
            "quote_translation": "",
            "chunk_id": str(fallback_chunk_id or "").strip(),
            "page": None,
            "section": "",
        })
    return evidence


def _normalize_verbatim_text(value: str) -> str:
    """Normalize extraction-only whitespace without changing quoted wording."""
    text = str(value or "").replace("\u00a0", " ")
    text = re.sub(r"(?<=\w)-\s*\r?\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _quote_is_verbatim(quote: str, corpus: str, min_segment_len: int = 12) -> bool:
    """Return True only when every quoted segment occurs verbatim in the source.

    A loose word-overlap check is useful for diagnostics, but it must not authorize
    a direct-disclosure or novelty finding.  In particular, a model-written sentence
    assembled from claim language and scattered source terms is not a quotation.
    """
    normalized_corpus = _normalize_verbatim_text(corpus)
    if not normalized_corpus:
        return False
    segments = [
        re.sub(r"^\s*\[[^\]]+\]\s*", "", segment).strip()
        for segment in re.split(r"\s*(?:…|\.{3,})\s*", str(quote or ""))
    ]
    normalized_segments = [
        _normalize_verbatim_text(segment)
        for segment in segments
        if len(_normalize_verbatim_text(segment)) >= min_segment_len
    ]
    return bool(normalized_segments) and all(
        segment in normalized_corpus for segment in normalized_segments
    )


_QUOTE_RECOVERY_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with",
    "is", "are", "be", "by", "that", "this", "상기", "및", "또는", "따라",
    "기초", "하는", "한다", "위한", "으로", "에서", "대한", "포함",
}


def _quote_recovery_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9가-힣]{2,}", str(value or "").lower())
        if token not in _QUOTE_RECOVERY_STOPWORDS
    }


def _source_chunk_text(doc: ExtractedDocument, chunk_id: str) -> str:
    """Return only the source text attached to an exact model-cited chunk id."""
    target = str(chunk_id or "").strip()
    if not target:
        return ""
    for candidate_id, text in _doc_chunks(doc):
        if str(candidate_id or "").strip() == target:
            return str(text or "").strip()
    return ""


def _recover_verbatim_quote(
    model_quote: str,
    source_doc: ExtractedDocument,
    chunk_id: str,
) -> str:
    """Recover a conservative exact excerpt from the model-cited source chunk.

    The recovery never searches another document or an unrelated paragraph.  It
    merely replaces a paraphrased representative quote with exact wording from the
    cited chunk.  Because the excerpt may support only part of a compound element,
    callers must keep it as inferred evidence rather than direct disclosure.
    """
    source_text = _source_chunk_text(source_doc, chunk_id)
    probe_tokens = _quote_recovery_tokens(model_quote)
    source_tokens = _quote_recovery_tokens(source_text)
    overlap = probe_tokens & source_tokens
    if not source_text or len(overlap) < 2:
        return ""

    coverage = len(overlap) / max(1, len(probe_tokens))
    if coverage < 0.20:
        return ""
    if len(source_text) <= _QUOTE_MAX_CHARS:
        return source_text

    units = [
        unit.strip()
        for unit in re.split(r"(?<=[.!?。！？])\s+|\r?\n+", source_text)
        if unit.strip()
    ]
    candidates: list[tuple[float, int, str]] = []
    for start in range(len(units)):
        excerpt = ""
        for end in range(start, min(len(units), start + 4)):
            combined = " ".join(units[start:end + 1]).strip()
            if len(combined) > _QUOTE_MAX_CHARS:
                break
            excerpt = combined
            excerpt_tokens = _quote_recovery_tokens(excerpt)
            matched = probe_tokens & excerpt_tokens
            score = len(matched) / max(1, len(probe_tokens))
            candidates.append((score, len(matched), excerpt))
        if not excerpt and len(units[start]) > _QUOTE_MAX_CHARS:
            shortened = _shorten_quote(units[start])
            excerpt_tokens = _quote_recovery_tokens(shortened)
            matched = probe_tokens & excerpt_tokens
            candidates.append((
                len(matched) / max(1, len(probe_tokens)),
                len(matched),
                shortened,
            ))

    if not candidates:
        return ""
    score, matched_count, recovered = max(
        candidates,
        key=lambda value: (value[0], value[1], len(value[2])),
    )
    return recovered if matched_count >= 2 and score >= 0.20 else ""


def _append_missing_limitation(missing: list[str], limitation: str) -> None:
    if limitation and limitation not in missing and len(missing) < 5:
        missing.append(limitation)


def _apply_korean_compound_relationship_guard(
    claim_text: str,
    evidence_text: str,
    directness: str,
    missing_limitations: list[str],
) -> tuple[str, list[str]]:
    """Conservatively audit Korean compound limitations against verbatim evidence.

    This guard covers relationship errors that keyword similarity cannot resolve:
    conversion plus calibration, state-conditioned decisions, and controls driven
    by multiple named upstream results.  It does not attempt a general semantic
    comparison and therefore activates only for explicit Korean claim patterns.
    """
    claim = re.sub(r"\s+", " ", str(claim_text or "")).strip()
    evidence = re.sub(r"\s+", " ", str(evidence_text or "")).strip()
    if not re.search(r"[가-힣]", claim):
        return directness, missing_limitations

    guarded = False
    if "변환" in claim and re.search(r"(?:좌표|제어\s*정보)", claim):
        if not re.search(
            r"(?:변환|환산|매핑|좌표계\s*(?:변경|변환)|"
            r"convert(?:s|ed|ing)?|transform(?:s|ed|ing|ation)?|"
            r"coordinate\s+(?:system|axis).{0,80}(?:to|into)|mapping)",
            evidence,
            re.IGNORECASE,
        ):
            _append_missing_limitation(
                missing_limitations,
                "입력 정보를 카메라 제어 좌표 정보로 변환하는 명시적 처리 관계",
            )
            guarded = True

    if re.search(r"설치\s*환경", claim) and re.search(r"오차.{0,12}보정|보정.{0,12}오차", claim):
        installation_context = re.search(
            r"(?:설치\s*(?:환경|상태|조건|위치)|설치된|배치된|현장\s*환경|"
            r"환경\s*오차|상대\s*(?:위치|배치)|"
            r"install(?:ation|ed)?|mount(?:ing|ed)?|position(?:ing|ed)?|"
            r"relative\s+(?:position|location|orientation)|independently\s+positioned)",
            evidence,
            re.IGNORECASE,
        )
        calibration_action = re.search(
            r"(?:보정|교정|보상|오차|편차|왜곡|"
            r"compensat(?:e|es|ed|ing|ion)|calibrat(?:e|es|ed|ing|ion)|"
            r"correct(?:s|ed|ing|ion)|alignment)",
            evidence,
            re.IGNORECASE,
        )
        coordinate_context = re.search(
            r"(?:좌표|좌표계|좌표축|coordinate|axis|pan\s+angle|tilt\s+angle)",
            evidence,
            re.IGNORECASE,
        )
        if not (installation_context and calibration_action and coordinate_context):
            _append_missing_limitation(
                missing_limitations,
                "설치 환경에 따른 좌표 오차를 보정하는 명시적 처리 관계",
            )
            guarded = True

    if (
        re.search(r"현재\s*동작\s*상태", claim)
        and re.search(r"(?:여부를?\s*결정|여부\s*판단)", claim)
    ):
        state_decision_link = re.search(
            r"(?:동작\s*상태|현재\s*상태|operating\s+state|current\s+(?:state|mode)|"
            r"idle|tracking\s+state).{0,140}"
            r"(?:기초|따라|고려|입력|based\s+on|according\s+to|consider|depending\s+on)"
            r".{0,140}(?:방향\s*전환|여부|결정|판단|turn|switch|direction|determin|decid)",
            evidence,
            re.IGNORECASE,
        )
        if not state_decision_link:
            _append_missing_limitation(
                missing_limitations,
                "카메라의 현재 동작 상태를 판단 입력으로 사용하여 방향 전환 여부를 결정하는 관계",
            )
            guarded = True

    if re.search(r"결정\s*결과.{0,80}및.{0,80}출력값.{0,80}따라", claim):
        combined_input_link = (
            re.search(r"(?:결정\s*결과|decision\s+(?:result|output))", evidence, re.IGNORECASE)
            and re.search(r"(?:출력값|output\s+(?:value|result))", evidence, re.IGNORECASE)
            and re.search(
                r"(?:함께|동시에|및|both|together|and).{0,120}"
                r"(?:따라|기초|입력|based\s+on|according\s+to|input)",
                evidence,
                re.IGNORECASE,
            )
        )
        if not combined_input_link:
            _append_missing_limitation(
                missing_limitations,
                "의사결정 결과와 좌표 보정부 출력값을 함께 입력하여 카메라를 제어하는 결합관계",
            )
            guarded = True

    if guarded and directness == "direct":
        directness = "inferred"
    return directness, missing_limitations


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
        r"^[\s(\[{]*([A-Za-z])\s*(?:-\s*(\d+))?[\s)\]}]*(?=$|\s|[:：._-])",
        raw,
    )
    if not m:
        return raw.upper()
    base = m.group(1).upper()
    return f"{base}-{m.group(2)}" if m.group(2) else base


_COMPARISON_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _comparison_safe_elements(elements: List[ClaimElement]) -> List[ClaimElement]:
    """Return elements with unique labels suitable for comparison prompts."""
    safe_elements: List[ClaimElement] = []
    used: set[str] = set()
    auto_idx = 0

    for elem in elements:
        label = normalize_label(elem.label)
        if not re.fullmatch(r"(?:P|[A-Z](?:-\d+)?)", label or "") or label in used:
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
    force_non_patent_full_text: bool = False,
) -> str:
    """대응관계 텍스트를 LLM 입력용으로 최적화해 반환.

    문헌 전문이 입력 예산 안에 들어오면 그대로 쓰고, 넘칠 때만 청구항 키워드가
    적중한 청크와 그 앞뒤 청크를 예산까지 모아 압축한다(벡터 검색은 쓰지 않는다).

    max_chars: 이 함수에서 잘라낼 최대 길이. 호출자가 직접 지정 시 사용.
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
    if doc.document_type == "non_patent":
        relevant_limit = min(relevant_limit, _NON_PATENT_RELEVANT_TEXT_MAX_CHARS)
        hard_limit = min(hard_limit, _NON_PATENT_RELEVANT_TEXT_MAX_CHARS)
    if not elements:
        return full_text[:hard_limit]

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

    selected_orders = set() if doc.document_type == "non_patent" else {0}
    total = sum(len(chunks[order][0]) + len(chunks[order][1]) + 2 for order in selected_orders)
    for score, order, _chunk_id, text in sorted(scored, key=lambda x: (-x[0], x[1])):
        # 적중 청크와 앞뒤 청크를 하나의 번들로 추가하되 전체 예산을 넘기지 않는다.
        bundle = [
            neighbor
            for neighbor in (max(0, order - 1), order, min(len(chunks) - 1, order + 1))
            if neighbor not in selected_orders
        ]
        bundle_len = sum(len(chunks[neighbor][0]) + len(chunks[neighbor][1]) + 2 for neighbor in bundle)
        if total + bundle_len > relevant_limit:
            if order not in selected_orders:
                item_len = len(chunks[order][0]) + len(text) + 2
                if total + item_len <= relevant_limit:
                    selected_orders.add(order)
                    total += item_len
            continue
        selected_orders.update(bundle)
        total += bundle_len

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
    if doc.paragraph_chunks:
        # 비특허문헌의 초록·참고문헌·감사의 글 제외는 추출 단계
        # (pdf_extractor._build_non_patent_records_and_chunks)에서 이미 끝난다.
        # 여기서 섹션명으로 한 번 더 거르면, 섹션 검출이 실패해 본문이 초록으로
        # 라벨링된 문헌이 통째로 비교 대상에서 사라진다. 특허문헌의 초록/청구항
        # 섹션만 방어적으로 제외한다.
        return [
            (chunk.chunk_id or chunk.paragraph_no or "", chunk.original_text.strip())
            for chunk in doc.paragraph_chunks
            if chunk.original_text
            and chunk.original_text.strip()
            and (
                doc.document_type == "non_patent"
                or not re.fullmatch(
                    r"\s*(?:초록|요약|abstract|summary|claims?|(?:특허)?청구(?:의)?\s*범위)\s*",
                    chunk.section or "",
                    re.IGNORECASE,
                )
            )
        ]

    if doc.paragraphs:
        if doc.paragraph_records:
            records = [record for record in doc.paragraph_records if not record.chunk_excluded]
            return [
                (
                    f"[{record.paragraph_no}]" if record.paragraph_no else "",
                    record.original_text.strip() or record.normalized_text.strip(),
                )
                for record in records
                if record.original_text.strip() or record.normalized_text.strip()
            ]
        return [
            (para_id, text.strip())
            for para_id, text in doc.paragraphs.items()
            if text and text.strip()
        ]

    if doc.pages:
        chunks = []
        for page_num, page_text in doc.pages.items():
            text = (page_text or "").strip()
            if not text or _page_is_non_description(text):
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
    # documents and compact each one independently with claim-keyword selection.
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
    tokens = re.findall(
        r"[A-Za-z0-9가-힣]{2,}|[ぁ-んァ-ヶー一-龯]{2,}",
        text.lower(),
    )
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

    for triggers, expansions in _KO_MULTILINGUAL_CLAIM_KEYWORD_GROUPS:
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

        # 검색·검증 대상은 초록·요약·청구항을 제외한 발명의 상세한 설명 청크다.
        if doc_idx not in corpus_cache:
            corpus_cache[doc_idx] = _full_doc_text(doc).lower()
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
        staged_results = None
        hybrid_results = None
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
        review_by_doc: Dict[int, Dict[str, ClaimElement]] = {}
        for doc_idx, review_elements in [
            *_false_negative_review_candidates(elements, prior_docs, doc_results),
            *_non_patent_quality_review_candidates(elements, prior_docs, doc_results),
        ]:
            bucket = review_by_doc.setdefault(doc_idx, {})
            for element in review_elements:
                bucket[normalize_label(element.label)] = element

        for doc_idx, review_map in review_by_doc.items():
            review_elements = list(review_map.values())
            try:
                reviewed = await _batch_judge_for_doc(
                    review_elements,
                    prior_docs[doc_idx],
                    doc_idx,
                    settings,
                    precision_review=True,
                    force_non_patent_full_text=False,
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
            for item in doc_results[doc_idx]:
                if (
                    normalize_label(item.get("label", "")) in review_map
                    and item.get("quality_issues")
                ):
                    item["analysis_status"] = "manual_review_required"
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
    force_non_patent_full_text: bool = False,
) -> List[Dict]:
    full_text = _build_doc_text(
        doc,
        elements,
        engine=settings.engine,
        settings=settings,
        force_non_patent_full_text=force_non_patent_full_text,
    )

    elements_text = "\n".join(f"({e.label}) {e.text}" for e in elements)

    prompt = render_prompt(
        "prompt_compare_single.txt",
        doc_filename=doc.filename,
        elements_text=elements_text,
        core_focus=_core_focus_text(elements),
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

    results = await _call_and_parse_comparison(
        prompt,
        elements,
        settings,
        source_docs=[doc],
        context=f"인용발명 {doc_idx + 1} 구성대비",
    )
    return _apply_non_patent_evidence_quality(results, doc)


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
        core_focus=_core_focus_text(elements),
        docs_block=docs_block,
    )

    return await _call_and_parse_comparison(
        prompt,
        elements,
        settings,
        expected_doc_indices=list(range(len(prior_docs))),
        source_docs=prior_docs,
        context="하이브리드 구성대비",
    )


async def _call_and_parse_comparison(
    prompt: str,
    elements: List[ClaimElement],
    settings: Settings,
    *,
    expected_doc_indices: Optional[List[int]] = None,
    source_docs: Optional[List[ExtractedDocument]] = None,
    context: str,
) -> List[Dict]:
    """Call the comparison model once and validate its recovered response locally."""
    system = load_prompt("system_compare.txt", _SYSTEM_BATCH)
    try:
        response = await call_ai(prompt, system, settings, agent="compare")
    except Exception as exc:
        raise CompareFailed(f"{context} LLM 호출 실패: {exc}") from exc

    try:
        return _parse_json_array(
            response,
            elements,
            expected_doc_indices,
            source_docs=source_docs,
        )
    except CompareFailed as exc:
        if "quote_translation" not in str(exc):
            raise CompareFailed(f"{context} 응답 형식 검증 실패: {exc}") from exc
        repair_prompt = (
            prompt
            + "\n\n[응답 보정]\n"
            + "직전 응답에서 외국어 quote의 한국어 quote_translation이 누락되었습니다. "
            + "각 found=true 외국어 인용에 원문에 충실한 한국어 번역을 반드시 채우고, "
            + "동일한 JSON 스키마 전체를 다시 출력하십시오."
        )
        try:
            repaired_response = await call_ai(
                repair_prompt,
                system,
                settings,
                agent="compare",
            )
            return _parse_json_array(
                repaired_response,
                elements,
                expected_doc_indices,
                source_docs=source_docs,
            )
        except Exception as repair_exc:
            raise CompareFailed(
                f"{context} 번역 누락 보정 실패: {repair_exc}"
            ) from repair_exc


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
                technical_judgment=best_match.get(
                    "technical_judgment",
                    best_match.get("llm_judgment", best_match.get("judgment", "")),
                ),
                evidence_status=best_match.get(
                    "evidence_status",
                    "verified" if best_match.get("quote") else "absent",
                ),
                unverified_quote=_shorten_quote(best_match.get("unverified_quote", "")),
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
    return label if re.fullmatch(r"(?:P|[A-Z](?:-\d+)?)", label) else ""


def _label_from_element_text(
    value: object,
    elements: List[ClaimElement],
) -> str:
    """Recover a label only when the model returned one exact element body."""
    text_key = re.sub(r"\s+", " ", str(value or "")).strip(" \t\r\n,;:：.")
    if not text_key:
        return ""
    matches = [
        normalize_label(element.label)
        for element in elements
        if re.sub(r"\s+", " ", str(element.text or "")).strip(
            " \t\r\n,;:：."
        ) == text_key
    ]
    return matches[0] if len(set(matches)) == 1 else ""


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
    *,
    source_docs: Optional[List[ExtractedDocument]] = None,
) -> List[Dict]:
    text = re.sub(r"```(?:json)?", "", response.strip()).replace("```", "").strip()
    parsed = _extract_json_payloads(text)
    if not parsed:
        raise CompareFailed(
            f"구성대비 응답에서 JSON 배열 또는 객체를 찾지 못했습니다. 응답 길이: {len(response)}자"
        )
    parsed = _expand_comparison_items(parsed)

    expected_labels = {normalize_label(element.label) for element in elements}
    elements_by_label = {
        normalize_label(element.label): element
        for element in elements
    }
    expected_docs = set(expected_doc_indices or [])
    normalized: List[Dict] = []
    invalid_reasons: List[str] = []
    required_fields = {"label", "found", "quote", "chunk_id", "judgment"}
    judgment_aliases = {
        "동일 95~100": "동일",
        "실질적 동일 90~94": "실질적 동일",
        "일부 차이 85~89": "일부 차이",
        "일부 유사 80~84": "일부 유사",
        "차이 1~79": "차이",
        "대응 없음 0": "대응 없음",
        "동일 개시": "동일",
        "부분 대응": "일부 차이",
        "부분 차이": "일부 차이",
        "기술적 관련성": "일부 유사",
        "부분 유사": "일부 유사",
        "유사": "일부 유사",
        "관련성": "일부 유사",
        "약한 관련성": "차이",
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

        raw_label = item.get("label", "")
        label = normalize_label(str(raw_label))
        if expected_labels and label not in expected_labels:
            recovered_label = _label_from_element_text(raw_label, elements)
            if recovered_label in expected_labels:
                label = recovered_label
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
        original_quote_translation = quote_translation
        evidence = _normalize_evidence(item.get("evidence", []), quote, item.get("chunk_id", ""))
        directness = str(item.get("directness", "") or "").strip().lower()
        if directness not in {"direct", "inferred", "absent"}:
            directness = "direct" if quote else "absent"
        raw_missing = item.get("missing_limitations", [])
        missing_limitations = _normalize_missing_limitations(raw_missing)
        reason = str(item.get("판단_이유", item.get("similarity_reason", "")) or "")
        claim_element_text = str(
            getattr(elements_by_label.get(label), "text", "") or ""
        )
        quote_fidelity_issue = False
        quote_recovered = False
        unverified_quote = ""
        evidence_status = "verified" if quote else "absent"
        if source_docs:
            source_idx = doc_idx if doc_idx is not None else 0
            source_doc = (
                source_docs[source_idx]
                if 0 <= source_idx < len(source_docs)
                else None
            )
            if source_doc is not None:
                source_corpus = _full_doc_text(source_doc)
                valid_evidence = [
                    span for span in evidence
                    if _quote_is_verbatim(str(span.get("quote", "") or ""), source_corpus)
                ]
                if quote and not _quote_is_verbatim(quote, source_corpus):
                    quote_fidelity_issue = True
                    unverified_quote = quote
                    if valid_evidence:
                        quote = str(valid_evidence[0].get("quote", "") or "")
                        quote_translation = str(
                            valid_evidence[0].get("quote_translation", "") or ""
                        )
                        item["chunk_id"] = str(
                            valid_evidence[0].get("chunk_id", "") or ""
                        )
                        evidence_status = "verified_from_evidence"
                    else:
                        recovered_quote = _recover_verbatim_quote(
                            quote,
                            source_doc,
                            str(item.get("chunk_id", "") or ""),
                        )
                        if recovered_quote:
                            quote = recovered_quote
                            # 원문은 지정 문단에서 정확한 문장으로 교체하되, 모델이
                            # 제공한 한국어 번역은 표시용으로 보존한다. 번역 자체가
                            # 없으면 아래 완전성 검증에서 재시도를 요구한다.
                            quote_translation = original_quote_translation
                            valid_evidence = [{
                                "limitation": "지정 문단에서 복구된 부분 근거",
                                "quote": recovered_quote,
                                "quote_translation": quote_translation,
                                "chunk_id": str(item.get("chunk_id", "") or ""),
                            }]
                            quote_recovered = True
                            evidence_status = "recovered_from_cited_chunk"
                            if directness == "direct":
                                directness = "inferred"
                        else:
                            quote = ""
                            quote_translation = ""
                            evidence_status = "unverified"
                evidence = valid_evidence
                if quote_fidelity_issue and not quote_recovered and not quote:
                    _append_missing_limitation(
                        missing_limitations,
                        "대표 발췌가 인용발명 원문과 일치하지 않아 직접 근거로 사용할 수 없음",
                    )
                    if directness == "direct":
                        directness = "inferred"
        if (
            source_docs
            and
            quote
            and not re.search(r"[\uac00-\ud7a3]", quote)
            and not re.search(r"[\uac00-\ud7a3]", quote_translation)
        ):
            invalid_reasons.append(f"{label}의 외국어 quote에 한국어 quote_translation이 없음")
            continue
        evidence_text = " ".join([
            quote,
            quote_translation,
            *[
                " ".join([
                    str(span.get("quote", "") or ""),
                    str(span.get("quote_translation", "") or ""),
                ])
                for span in evidence
            ],
        ])
        directness, missing_limitations = _apply_korean_compound_relationship_guard(
            claim_element_text,
            evidence_text,
            directness,
            missing_limitations,
        )
        # 평균값 제어는 최소값/하한 조건의 직접 개시가 아니다. 특히 살두께처럼
        # 평균과 국소 최소값의 기술적 의미가 다른 경우 LLM이 둘을 같은 조건으로
        # 승격하지 못하도록 원문 표현을 기준으로 하위 제한을 보존한다.
        if (
            _MINIMUM_LIMIT_RE.search(claim_element_text)
            and _AVERAGE_ONLY_RE.search(evidence_text)
            and not _MINIMUM_EVIDENCE_RE.search(evidence_text)
        ):
            minimum_limitation = next(
                (
                    part.strip()
                    for part in re.split(r"(?:및|그리고|,)", claim_element_text)
                    if _MINIMUM_LIMIT_RE.search(part)
                ),
                "최소값 또는 하한 조건",
            )
            if minimum_limitation not in missing_limitations:
                missing_limitations.append(minimum_limitation)
            if directness == "direct":
                directness = "inferred"
        if (
            _EXECUTED_ADJUSTMENT_RE.search(claim_element_text)
            and _RECOMMENDATION_ONLY_RE.search(quote)
            and not _EXECUTED_ADJUSTMENT_RE.search(quote)
        ):
            execution_limitation = "형상 파라미터를 실제로 조정하여 모델을 생성하는 실행 단계"
            if execution_limitation not in missing_limitations:
                missing_limitations.append(execution_limitation)
            if directness == "direct":
                directness = "inferred"
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
        reconciled_judgment = _reconcile_judgment_with_reason(
            llm_judgment,
            directness,
            missing_limitations,
            reason,
            quote,
        )
        judgment = _cap_judgment_for_coverage(
            reconciled_judgment,
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
        if found and not quote and quote_fidelity_issue:
            found = False
            judgment = "대응 없음"
            judgment_adjustment_reason = "quote_not_verbatim"
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
            "technical_judgment": llm_judgment,
            "llm_judgment": llm_judgment,
            "judgment_adjusted": judgment != llm_judgment,
            "judgment_adjustment_reason": judgment_adjustment_reason,
            "evidence_status": evidence_status,
            "unverified_quote": unverified_quote,
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
