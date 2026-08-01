"""
인용 체인 그래프 — 비교 캐시 기반 주인용발명 선정 (다국어 특허 완전 지원)

[버전 이력]
- v1: 키워드 매칭 (한국어 키워드 ↔ 영어 본문 매칭 실패 문제)
- v2: LLM 비교 캐시(comparisons_{i}.json) 기반 → 언어 무관하게 정확한 선정
- v3: 보조인용발명 선정을 "보완성(complementarity)" 기준으로 개선
       주인용발명이 커버하지 못하는 구성요소를 가장 잘 채우는 문헌을 2차로 선정
       종속항 체인 build가 실제 호출되도록 수정
- v4: 독립항은 원칙적으로 2개 문헌을 유지하되, 단순 주지관용 구성의
       명시 근거에 한해 제3문헌을 conventional_support 역할로 예외 허용
- v5: 종속항은 부모 체인을 그대로 상속하고, 해당 종속항의 남은 구성을
       하나의 문헌이 모두 보완할 때에만 새 인용발명 1개를 추가
- v6: 종속항 결론·프롬프트 계약 변경에 맞춰 기존 보고서 캐시를 한 번 갱신
- v7: 판정 퍼센트 밴드와 동일/실질적 동일 구별 기준 정리에 맞춰 캐시 갱신
- v8: 조건 기반 선택식 구성은 상위개념·복수대안·선택구조 중심으로 검색/주인용 선정
- v9: 범용 골격의 넓은 커버리지보다 차별적 핵심 구성·효과의 직접 개시를
      주인용 선정의 우선 기준으로 사용하고, 잔여 핵심 제한만 보조문헌으로 보완
- v10: 보고서 문헌명 재매핑·중복 발췌·복수 문헌의 단일문헌 신규성 오판을 수정
"""
from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from backend.models.schemas import ExtractedDocument, ParsedClaim
from backend.services.citation_extractor import normalize_label
from backend.services.quantitative_assessment import assess_claims

logger = logging.getLogger(__name__)

MAX_INDEPENDENT_REFS = 2   # 독립항에 사용할 최대 인용발명 수
MAX_INDEPENDENT_REFS_WITH_CONVENTIONAL_SUPPORT = 3
MAX_DEPTH_INCREMENT = 1    # 종속항 1단계당 추가 허용 인용발명 수
CITATION_CHAIN_POLICY_VERSION = 28

# 패밀리별 주인용발명 선정에서 종속항 커버리지는 독립항 적합도를 뒤집는
# 주점수가 아니라, 독립항 점수가 근접한 후보들 사이의 타이브레이커로만 쓴다.
PRIMARY_NEAR_TIE_MARGIN = 5.0
DISTINCTIVE_CORE_NEAR_TIE_MARGIN = 0.12
DEPENDENT_REUSE_TIEBREAK_MAX = 2.0

# ── 주인용 자격 게이트 ───────────────────────────────────────────────────
# 순서쌍 전수 탐색은 유지하되, 보조문헌이 좋다는 이유만으로 명백히 열등한
# 문헌이 주인용 자리를 차지하는 역전을 막는다. 차별적 핵심의 직접 개시량이
# 최고 문헌에 크게 못 미치는 문헌은 애초에 주인용 후보에서 뺀다.
# 두 기준(비율/절대차) 중 하나만 만족해도 통과시키는 이유는, 최고값이 0에
# 가까운 사건에서 비율 기준만 쓰면 후보가 비합리적으로 좁아지기 때문이다.
PRIMARY_ELIGIBILITY_RATIO = 0.72
PRIMARY_ELIGIBILITY_ABS_MARGIN = 0.10
# LLM 최종 판정에 넘길 후보 수. 고정 상수를 피하기 위해 점수 마진으로 정하고
# 범위만 제한한다.
PRIMARY_SHORTLIST_MIN = 2
PRIMARY_SHORTLIST_MAX = 5
SECONDARY_SHORTLIST_MAX = 4
# 전체 점수가 낮아도 차별적 핵심 구성을 원문으로 직접 개시한 문헌은 후보에서
# 누락되면 안 된다. 알고리즘 정렬이 놓친 문헌을 LLM이 볼 기회를 보장한다.
CORE_DISCLOSURE_FORCE_INCLUDE_SIMILARITY = 0.85

# 판정 점수표 (높을수록 유사)
_JUDGMENT_SCORE = {
    "동일": 5,
    "실질적 동일": 4,
    "일부 차이": 3,
    "일부 유사": 2,
    "대응 없음": 0,
    "차이": 0,
}

# 주인용발명의 "커버" 기준: 이 점수 이상이면 해당 구성요소를 커버한 것으로 봄
_PRIMARY_COVER_THRESHOLD = 3   # "일부 차이" 이상

# 보조인용발명이 공백을 채우는 최소 기준: 이 점수 이상이면 보완 인정
# "일부 유사"는 핵심 특징 미개시 상태이므로 결합 공백 보완 근거로 쓰지 않는다.
_SECONDARY_FILL_THRESHOLD = 3  # "일부 차이" 이상

# 소프트 공백(약점): "일부 차이" — 커버는 됐지만 극복해야 할 차이가 남은 구성요소
_SOFT_GAP_SCORE = 3
# 소프트 공백 보강 인정 기준: 보조 문헌이 주인용발명보다 명확히 좋은 판정일 때만
_SECONDARY_IMPROVE_THRESHOLD = 4  # "실질적 동일" 이상
# 보조 문헌의 불완전한 명시 근거 기준.
# `차이`라도 실제 발췌가 있으면 청구항 전체에는 못 미치지만 주인용발명의 빠진
# 하위 제한을 설명하는 문헌일 수 있으므로 마지막 단계의 보조 후보로 남긴다.
_SECONDARY_SUPPORT_THRESHOLD = 0
# Dependent claims are often narrower implementation choices. If a newly added
# document expressly teaches the main implementation axis but leaves a limitation
# different, keep it in the chain as partial support so the report can explain
# both the usable teaching and the remaining difference. A mere `차이` quote is
# audit evidence only and must not become an actual dependent-claim reference.
_DEPENDENT_PARTIAL_SUPPORT_THRESHOLD = 2  # "일부 유사" 이상
# 주인용발명 단독 가중 유사도가 이 이상이면 소프트 공백이 있어도 결합 불필요 (단독 충분)
SINGLE_SUFFICIENT_SIMILARITY = 95.0

# 판정 라벨 → 내부 결손 민감도 앵커. 보고서에 표시되는 95~100 등의 법적·표현상
# 유사도 퍼센트가 아니며, 평균점수가 필수구성 결손을 덮지 않도록 보수적으로 벌린다.
# 신뢰도·문헌 선정 감사용으로만 사용하고 보고서에는 출력하지 않는다.
_LABEL_PERCENT = {
    "동일": 95,
    "실질적 동일": 80,
    "일부 차이": 60,
    "일부 유사": 35,
    "차이": 15,
    "대응 없음": 0,
}

# Normalized judgment values used by MainScore/SubScore. These mirror the
# product policy: identical=1.00, substantially identical=0.85,
# partial difference=0.55, partial similarity=0.35, difference=0.15,
# no correspondence=0.00.
_JUDGMENT_SIMILARITY = {
    "동일": 1.00,
    "실질적 동일": 0.85,
    "일부 차이": 0.55,
    "일부 유사": 0.35,
    "차이": 0.15,
    "대응 없음": 0.00,
}

# 신뢰도 경고 기준 (경고 부착용 — 보고서 결론을 차단하지 않음)
PRIMARY_SIMILARITY_FLOOR = 40   # 주인용발명 단독 유사도 하한
CONFIDENT_SIMILARITY_FLOOR = 80  # 결합 후 유사도 확신 기준
UNCOVERED_PERCENT_THRESHOLD = 35  # 이 값 이하이면 해당 구성요소 '미커버'로 간주

# 주/보조 인용발명 선정 점수:
# - 주인용발명은 청구항의 차별적 핵심 구조·작동관계·효과에 관한 직접 근거를 우선한다.
# - 범용 입출력·처리 골격의 폭은 핵심 직접개시가 비슷할 때의 보조 지표로만 쓴다.
# - 보조인용발명은 주인용발명의 차이점, 특히 중요 구성의 공백을 메우는 문헌을 고른다.
_CORE_IMPORTANCE_THRESHOLD = 4
_CORE_STRONG_PERCENT = 75
_CORE_WEAK_PERCENT = 35
_PRIMARY_COVERAGE_BONUS = 40
_CORE_STRONG_BONUS = 8
_CORE_MISS_PENALTY = 15
_CORE_WEAK_PENALTY = 6

_COMBINATION_RATIONALES = {
    "gap_filling": {
        "label": "공백 보완형",
        "description": "문헌 1이 가장 가까운 출발점이고, 문헌 2가 문헌 1의 빠진 구성요소를 보완하는 유형",
        "writing_guidance": "문헌 1의 개시, 차이점, 문헌 2의 보완 근거, 결합/적용 이유, 예측 가능한 효과를 구분해 작성한다.",
    },
    "substitution": {
        "label": "단순 치환형",
        "description": "문헌 1의 특정 구성을 문헌 2의 알려진 대체수단으로 바꾸면 청구항에 이르는 유형",
        "writing_guidance": "치환 대상 구성과 문헌 2의 대체수단을 대응시키고, 치환 후에도 문헌 1의 기본 원리가 유지되는지 설명한다.",
    },
    "known_tech_application": {
        "label": "공지기술 적용형",
        "description": "문헌 1의 장치/방법에 문헌 2의 알려진 개선 기술을 적용하는 유형",
        "writing_guidance": "문헌 1의 약점과 문헌 2의 명시적 개선수단을 연결하고, 적용 결과가 통상의 기술자에게 예측 가능한지 설명한다.",
    },
    "problem_solution": {
        "label": "문제-해결 동기형",
        "description": "문헌 1과 청구항의 차이를 객관적 기술문제로 정리하고, 문헌 2가 그 해결수단을 제시하는 유형",
        "writing_guidance": "차이점에서 객관적 기술문제를 도출한 뒤, 문헌 2의 해결수단을 적용할 이유를 간결하게 제시한다.",
    },
    "obvious_to_try": {
        "label": "obvious to try형",
        "description": "문헌 1의 문제에 대해 문헌 2가 제한된 수의 예측 가능한 선택지 중 하나를 제시하는 유형",
        "writing_guidance": "선택지가 제한적이고 적용 결과가 예측 가능하다는 점을 중심으로 작성한다.",
    },
    "design_variation": {
        "label": "설계변경 보강형",
        "description": "문헌 2가 동일 또는 인접 분야에서 해당 변경이 통상적으로 사용되었음을 뒷받침하는 유형",
        "writing_guidance": "주지관용기술처럼 단정하지 말고, 문헌 2의 명시 근거가 설계변경의 보강 근거임을 구분해 작성한다.",
    },
    "aggregation": {
        "label": "기능 중복/병렬 결합형",
        "description": "문헌 1과 문헌 2의 기능이 독립적으로 결합될 뿐, 새로운 상호작용 효과가 없는 유형",
        "writing_guidance": "각 기능이 독립적으로 작동하고 결합에 따른 예측 곤란한 상호작용 효과가 없다는 점을 설명한다.",
    },
    "specific_selection": {
        "label": "상위개념-하위개념 보강형",
        "description": "문헌 1이 상위개념을 개시하고, 문헌 2가 그중 특정 하위 구현을 구체적으로 보여주는 유형",
        "writing_guidance": "문헌 1의 상위개념과 문헌 2의 하위 구현을 연결하고, 선택의 예측 가능성을 설명한다.",
    },
    "supporting_evidence": {
        "label": "명시 근거 보강",
        "description": "완전한 보완까지는 아니지만 보조 인용발명이 주 인용발명의 차이점 또는 약점에 관한 명시 근거를 제공하는 유형",
        "writing_guidance": "문헌 2가 완전한 대응 근거인지 단순 보강 근거인지 구분하고, 남는 차이점은 별도로 표시한다.",
    },
    "conventional_support": {
        "label": "주지관용 구성 문헌 보강형",
        "description": "핵심 기술사상은 앞선 인용발명으로 판단하고, 별도 문헌은 CPU·바퀴·일반 제어부와 같은 통상적 구성의 명시 근거로만 사용하는 유형",
        "writing_guidance": "주지관용 구성의 통상적 기능과 단순 결합 가능성을 설명하되, 이 문헌을 핵심 차이점이나 새로운 상호작용의 보완 근거로 확대하지 않는다.",
    },
    "common_general_knowledge": {
        "label": "인용발명 + 주지관용 검토형",
        "description": "주 인용발명과의 차이 중 단순·통상적 구성만 주지관용 여부를 검토하고, 다른 인용발명은 결합 근거로 사용하지 않는 유형",
        "writing_guidance": "주 인용발명의 직접 개시와 차이점을 먼저 확정하고, 표시된 구성에 한해서만 주지관용의 근거와 단순 채용 가능성을 검토한다. 다른 인용발명을 보완 근거로 기재하지 않는다.",
    },
    "single_reference": {
        "label": "단일 문헌 충분",
        "description": "주 인용발명만으로 청구항 대비가 충분하여 보조 인용발명을 채택하지 않는 유형",
        "writing_guidance": "문헌 1 단독 대비 결과를 중심으로 작성하고, 다른 문헌은 보조 근거로 확장하지 않는다.",
    },
    "insufficient_support": {
        "label": "보완 인용발명 없음",
        "description": "주 인용발명에 남은 차이점을 보완하는 인용발명이 없어 현재 문헌만으로 거절 근거가 완성되지 않는 유형",
        "writing_guidance": "남은 차이점과 추가로 필요한 문헌 근거를 명확히 작성하고, 단일 문헌으로 충분하다고 표현하지 않는다.",
    },
}


# ---------------------------------------------------------------------------
# 보완성(Complementarity) 기반 헬퍼
# ---------------------------------------------------------------------------

def _load_cache(job_dir: str, doc_idx: int) -> Optional[Dict]:
    path = Path(job_dir) / f"comparisons_{doc_idx}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _importance_value(value) -> int:
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return 3


_CONVENTIONAL_COMPONENT_RE = re.compile(
    r"(?:"
    r"\bcpu\b|\bprocessor\b|\bmicroprocessor\b|\bcontroller\b|\bmemory\b|"
    r"\bstorage\b|\binterface\b|\btransceiver\b|\bdisplay\b|\bbattery\b|"
    r"\bwheel\b|\bmotor\b|\bsensor\b|\bhousing\b|\bframe\b|\bserver\b|"
    r"중앙\s*처리\s*장치|프로세서|마이크로프로세서|마이크로컨트롤러|제어부|제어기|"
    r"제어\s*유닛|볼륨\s*제어\s*유닛|"
    r"컨트롤러|메모리|저장부|통신부|송수신부|인터페이스|입력부|출력부|표시부|"
    r"디스플레이|전원부|배터리|바퀴|휠|모터|센서|하우징|프레임|서버|단말"
    r")",
    re.IGNORECASE,
)
_SPECIALIZED_CONSTRAINT_RE = re.compile(
    r"(?:\d|피드백|전역|국소|학습|암호|복호|보정|적응|동기|임계|특정\s*조건|"
    r"고조파|왜곡|비선형|트랜지스터|부하|상대|비율|에너지|"
    r"상호\s*작용|연동|based\s+on|in\s+response\s+to|feedback|global|local|"
    r"adaptive|threshold|synchron|encrypt|decrypt|calibrat|\bharmonic\b|\bdistortion\b|"
    r"non[- ]?linear|\btransistor\b|\bload\b|\brelative\b|\bratio\b|\benergy\b)",
    re.IGNORECASE,
)
_TECHNICAL_RELATION_RE = re.compile(
    r"(?:동작\s*주파수|작동\s*주파수|운용\s*주파수|주파수.*(?:조정|조절|변경)|"
    r"(?:조정|조절|변경).*주파수|파라미터.*(?:되도록|하도록)|"
    r"(?:통해|따라|기초하여|기반하여|대응하여).*(?:조정|조절|변경|생성|제어)|"
    r"(?:조정|조절|변경|생성|제어).*(?:통해|따라|기초하여|기반하여|대응하여)|"
    r"operat(?:ing|ion)\s+frequency|frequency.*(?:adjust|vary|control)|"
    r"(?:adjust|vary|control).*(?:frequency|parameter)|"
    r"(?:based\s+on|according\s+to|in\s+response\s+to).*(?:adjust|vary|control))",
    re.IGNORECASE,
)
_GENERIC_INTERFACE_RE = re.compile(
    r"(?:고\s*임피던스\s*입력|저\s*임피던스\s*출력|"
    r"high\s+impedance\s+input|low\s+impedance\s+output|"
    r"입력|출력|input|output)",
    re.IGNORECASE,
)
_FUNCTIONAL_VERB_RE = re.compile(
    r"(?:제어|수신|저장|비교|판단|생성|전송|변환|검출|처리|구동|control|receive|"
    r"store|compare|determin|generat|transmit|convert|detect|process|drive)",
    re.IGNORECASE,
)
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


def _is_conditioned_selection_element(text: str) -> bool:
    """Return True when an OR limitation may really claim conditional selection."""
    source = text or ""
    return bool(_SELECTION_OR_RE.search(source) and _SELECTION_CONDITION_RE.search(source))


def _is_generic_interface_element(element) -> bool:
    """Return whether a short interface-only limitation should not become a core.

    The component order produced by an LLM parser is not a legal measure of
    inventiveness.  A first-listed input/output specification therefore cannot
    outweigh a later compound circuit or signal-relation limitation solely
    because it received importance ``5``.
    """
    text = " ".join((element.text or "").split())
    return bool(
        text
        and len(text) <= 80
        and _GENERIC_INTERFACE_RE.search(text)
        and not _SPECIALIZED_CONSTRAINT_RE.search(text)
        and not bool(element.is_sub)
    )


def _is_distinctive_technical_element(element) -> bool:
    """Return whether a compound relation must remain in the primary-core score."""
    text = " ".join((element.text or "").split())
    if not text:
        return False
    return bool(
        len(text) > 100
        or _SPECIALIZED_CONSTRAINT_RE.search(text)
        or _TECHNICAL_RELATION_RE.search(text)
        or _is_conditioned_selection_element(text)
    )


def _is_simple_conventional_component(element) -> bool:
    """Cap a bare component recital even when positional parsing assigned 5."""
    text = " ".join((element.text or "").split())
    return bool(
        text
        and len(text) <= 80
        and _CONVENTIONAL_COMPONENT_RE.search(text)
        and not _SPECIALIZED_CONSTRAINT_RE.search(text)
        and not _TECHNICAL_RELATION_RE.search(text)
        and len(_FUNCTIONAL_VERB_RE.findall(text)) <= 1
        and not bool(element.is_sub)
    )


def _conventionality_basis(element) -> Optional[str]:
    """Return a conservative basis when an element is a simple conventional part.

    This is intentionally narrower than a legal conclusion that a feature is
    common general knowledge. It only routes short, low-importance, independent
    component limitations to the conventional-support reporting policy.
    """
    text = " ".join((element.text or "").split())
    if not text or len(text) > 120:
        return None
    if _importance_value(element.importance) > 3 or bool(element.is_sub):
        return None
    if not _CONVENTIONAL_COMPONENT_RE.search(text):
        return None
    if _SPECIALIZED_CONSTRAINT_RE.search(text):
        return None
    if len(_FUNCTIONAL_VERB_RE.findall(text)) > 1:
        return None
    if len(re.findall(r"(?:및|또는|그리고|\band\b|\bor\b)", text, re.IGNORECASE)) > 1:
        return None
    return "낮은 중요도의 짧고 독립적인 일반 구성으로서 특수한 수치·조건·상호작용 제한이 없음"


def _element_weight_map(claims: List[ParsedClaim]) -> Dict[tuple[str, str], int]:
    weights: Dict[tuple[str, str], int] = {}
    for claim in claims:
        claim_key = str(claim.claim_number)
        for element in claim.elements:
            weight = _importance_value(element.importance)
            if _is_conditioned_selection_element(element.text):
                weight = max(weight, _CORE_IMPORTANCE_THRESHOLD)
            weights[(claim_key, normalize_label(element.label))] = weight
    return weights


def _items_by_label(items: list) -> Dict[str, Dict]:
    by_label: Dict[str, Dict] = {}
    for item in items:
        if isinstance(item, dict):
            by_label[normalize_label(item.get("label", ""))] = item
    return by_label


def _similarity_for_judgment(judgment: str) -> float:
    if judgment in _JUDGMENT_SIMILARITY:
        return _JUDGMENT_SIMILARITY[judgment]
    return _LABEL_PERCENT.get(judgment, 0) / 100.0


def _atomic_limitation_coverage(item: Optional[Dict]) -> Optional[float]:
    """하위 제한 evidence/missing 목록이 있을 때 원자적 커버 비율을 반환한다.

    기존 캐시는 하위 제한 목록이 없으므로 None을 반환하여 과거 판정 강도를 그대로
    유지한다. 새 캐시에서는 같은 `일부 차이` 안에서도 실제로 확인된 하위 제한 수와
    누락된 하위 제한 수가 문헌 선정 점수에 반영된다.
    """
    if not item:
        return None
    evidence = item.get("evidence") or []
    missing = item.get("missing_limitations") or []
    covered_count = len([value for value in evidence if isinstance(value, dict) and value.get("quote")])
    missing_count = len([value for value in missing if str(value).strip()])
    denominator = covered_count + missing_count
    if denominator == 0:
        return None
    return covered_count / denominator


def _item_similarity(item: Optional[Dict]) -> float:
    """판정 라벨과 원자적 하위 제한 커버리지를 함께 반영한 내부 강도값."""
    if not item:
        return 0.0
    base = _similarity_for_judgment(item.get("judgment", "대응 없음"))
    atomic = _atomic_limitation_coverage(item)
    if atomic is None:
        return base
    # 라벨 판정을 대체하지 않고, 동일 라벨 후보 사이에서 하위 제한 누락이 많은
    # 문헌을 보수적으로 낮추는 범위에서만 조정한다.
    directness = str(item.get("directness") or "").strip().lower()
    directness_factor = 1.0 if directness in {"", "direct"} else 0.9 if directness == "inferred" else 0.8
    return base * (0.70 + 0.30 * atomic) * directness_factor


def _weighted_average(rows: list[Dict], value_key: str = "similarity") -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        weight = float(row.get("importance", 3))
        numerator += weight * float(row.get(value_key, 0.0))
        denominator += weight
    return numerator / denominator if denominator else 0.0


def _direct_evidence_similarity(row: Dict) -> float:
    """직접 발췌로 확인되는 내부 강도값.

    넓은 기능적 유사성만 있는 문헌이 핵심 구성의 직접 원문을 가진 문헌을
    주인용 후보에서 밀어내지 않도록, 원문과 directness를 별도 축으로 둔다.
    과거 캐시처럼 directness가 비어 있으면 발췌가 있는 경우 직접 근거로 본다.
    """
    item = row.get("item") or {}
    similarity = float(row.get("similarity", 0.0))
    if similarity < 0.35:
        return 0.0
    if not item.get("quote"):
        return similarity * 0.35
    directness = str(item.get("directness") or "direct").strip().lower()
    if directness == "absent":
        return 0.0
    if directness == "inferred":
        return similarity * 0.65
    return similarity


def _comparison_rows(
    cache: Optional[Dict],
    claims: List[ParsedClaim],
    dynamic_weights: Optional[Dict[tuple[str, str], float]] = None
) -> list[Dict]:
    rows: list[Dict] = []
    if not cache:
        return rows

    for claim in claims:
        claim_results = cache.get(str(claim.claim_number), [])
        if not isinstance(claim_results, list):
            claim_results = []
        by_label = _items_by_label(claim_results)

        for idx, element in enumerate(claim.elements):
            label = normalize_label(element.label)
            item = by_label.get(label, {})
            judgment = item.get("judgment", "대응 없음")
            importance = _importance_value(element.importance)
            if dynamic_weights:
                importance = dynamic_weights.get((str(claim.claim_number), label), importance)
            selection_structure = _is_conditioned_selection_element(element.text)
            if selection_structure:
                importance = max(importance, _CORE_IMPORTANCE_THRESHOLD)
            rows.append({
                "claim_key": str(claim.claim_number),
                "label": label,
                "element_index": idx,
                "importance": importance,
                "selection_structure": selection_structure,
                "item": item,
                "judgment": judgment,
                "rank": _JUDGMENT_SCORE.get(judgment, 0),
                "similarity": _item_similarity(item),
                "atomic_coverage": _atomic_limitation_coverage(item),
                "has_quote": bool(item.get("quote")),
            })
    return rows


def _score_prior_cache(
    cache: Optional[Dict],
    claims: List[ParsedClaim],
    dynamic_weights: Optional[Dict[tuple[str, str], float]] = None
) -> tuple[float, int, Dict]:
    """기술분야 어휘에 의존하지 않고 주 문헌 후보를 평가한다."""
    if not cache:
        return 0.0, 0, {}

    rows = _comparison_rows(cache, claims, dynamic_weights)
    if not rows:
        return 0.0, 0, {}

    element_coverage = _weighted_average(rows)
    total_weight = sum(r["importance"] for r in rows) or 1
    evidence_adjusted = sum(
        r["importance"] * r["similarity"] * (
            0.55
            + 0.25 * bool(r["item"].get("quote"))
            + 0.10 * bool(r["item"].get("chunk_id") or r["item"].get("paragraph_no"))
            + 0.10 * bool(r["item"].get("판단_이유") or r["item"].get("similarity_reason"))
        )
        for r in rows
    ) / total_weight
    strong_breadth = sum(
        r["importance"] for r in rows if r["similarity"] >= 0.85
    ) / total_weight
    core_rows = [r for r in rows if r["importance"] >= _CORE_IMPORTANCE_THRESHOLD]
    # 중요도·문헌 희소성으로 핵심 구성이 식별되지 않는 예외에서는 전체 구성을
    # 사용하되, 정상 사건에서는 핵심 구성만 주 인용발명의 주점수로 삼는다.
    distinctive_rows = core_rows or rows
    distinctive_weight = sum(r["importance"] for r in distinctive_rows) or 1
    raw_core_coverage = _weighted_average(distinctive_rows)
    # `차이` 수준의 인용문은 감사·보조 검토에는 남기되, 주인용발명의 핵심
    # 직접개시 점수로 승격하지 않는다.
    core_coverage = sum(
        r["importance"] * (r["similarity"] if r["similarity"] >= 0.35 else 0.0)
        for r in distinctive_rows
    ) / distinctive_weight
    distinctive_direct_coverage = sum(
        r["importance"] * _direct_evidence_similarity(r)
        for r in distinctive_rows
    ) / distinctive_weight
    distinctive_strong_breadth = sum(
        r["importance"]
        for r in distinctive_rows
        if _direct_evidence_similarity(r) >= 0.55
    ) / distinctive_weight
    critical_gap_weight = sum(
        r["importance"] for r in distinctive_rows if r["similarity"] < 0.35
    ) / distinctive_weight
    direct_disclosure_breadth = (
        sum(1 for r in rows if _direct_evidence_similarity(r) >= 0.55) / len(rows)
    )
    # 기술분야 적합성은 별도 LLM 호출 없이, 핵심 관계를 둘러싼 나머지 구성의
    # 직접 개시율을 대용 지표로 사용한다. 핵심 점수가 모두 낮을 때 B/C와 같은
    # 기반 구성을 많이 직접 개시한 문헌이 주 문헌 동률을 우선 해소한다.
    contextual_rows = [r for r in rows if r not in distinctive_rows] or rows
    contextual_direct_count = sum(
        1 for r in contextual_rows if _direct_evidence_similarity(r) >= 0.55
    )
    field_alignment_coverage = (
        sum(_direct_evidence_similarity(r) for r in contextual_rows) / len(contextual_rows)
        if contextual_direct_count >= 2
        else 0.0
    )
    main_score = max(
        0.0,
        0.40 * distinctive_direct_coverage
        + 0.25 * core_coverage
        + 0.15 * distinctive_strong_breadth
        + 0.12 * element_coverage
        + 0.08 * evidence_adjusted
        - 0.20 * critical_gap_weight,
    )
    match_count = sum(1 for r in rows if r["rank"] >= _SECONDARY_FILL_THRESHOLD)
    detail = {
        "main_score": round(main_score, 4),
        "element_coverage": round(element_coverage, 4),
        "evidence_adjusted_coverage": round(evidence_adjusted, 4),
        "strong_breadth": round(strong_breadth, 4),
        "critical_gap_weight": round(critical_gap_weight, 4),
        "core_coverage": round(core_coverage, 4),
        "raw_core_coverage": round(raw_core_coverage, 4),
        "distinctive_direct_coverage": round(distinctive_direct_coverage, 4),
        "distinctive_strong_breadth": round(distinctive_strong_breadth, 4),
        "direct_disclosure_breadth": round(direct_disclosure_breadth, 4),
        "field_alignment_coverage": round(field_alignment_coverage, 4),
        "distinctive_core_labels": [r["label"] for r in distinctive_rows],
        "generic_breadth_is_tiebreak_only": True,
        "formula": (
            "0.40*distinctive_direct + 0.25*distinctive_core + "
            "0.15*distinctive_breadth + 0.12*overall + 0.08*evidence - 0.20*core_gap"
        ),
    }

    return round(main_score * 100, 2), match_count, detail


def _compute_primary_gaps(cache: Optional[Dict], claim_keys: Optional[set] = None) -> set:
    """
    주인용발명이 커버하지 못하는 (청구항번호, 구성요소라벨) 쌍 집합 반환.
    커버 기준: _PRIMARY_COVER_THRESHOLD 이상 판정 → 커버됨
    claim_keys가 주어지면 해당 청구항(독립항)만 본다 — 캐시에 종속항 키도
    저장되므로, 독립항 결합 판단에 종속항 공백이 섞이지 않게 한다.
    """
    if not cache:
        return set()

    gaps = set()
    for claim_key, items in cache.items():
        if claim_keys is not None and claim_key not in claim_keys:
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            label = item.get("label", "")
            judgment = item.get("judgment", "대응 없음")
            score = _JUDGMENT_SCORE.get(judgment, 0)
            if score < _PRIMARY_COVER_THRESHOLD:
                gaps.add((claim_key, normalize_label(label)))
    return gaps


def _compute_complementarity_score(
    cache: Optional[Dict],
    gaps: set,
    weights: Optional[Dict[tuple[str, str], int]] = None,
) -> int:
    """
    해당 인용발명이 주인용발명의 공백(gaps)을 얼마나 보완하는지 점수 계산.
    gaps에 있는 (청구항, 구성요소) 중 _SECONDARY_FILL_THRESHOLD 이상 판정 시 보완 인정.
    반환값: 보완 점수 합계 (0이면 공백을 전혀 채우지 못함)
    """
    if not cache:
        return 0

    score = 0
    for (claim_key, label) in gaps:
        items = cache.get(claim_key, [])
        if not isinstance(items, list):
            continue
        # gaps의 label은 이미 정규화돼 있으므로 cache item도 정규화해 비교한다.
        item = next((i for i in items if normalize_label(i.get("label")) == label), None)
        if item is None:
            continue
        judgment = item.get("judgment", "대응 없음")
        j_score = _JUDGMENT_SCORE.get(judgment, 0)
        if j_score >= _SECONDARY_FILL_THRESHOLD:
            score += j_score * (weights or {}).get((claim_key, label), 3)
    return score


def _compute_soft_gaps(cache: Optional[Dict], claim_keys: Optional[set] = None) -> set:
    """주인용발명 판정이 '일부 차이'인 (청구항번호, 구성요소라벨) 쌍 집합 반환.

    커버는 됐지만 차이가 남아 있어, 문헌 근거 없이는 주지관용 논거로
    극복해야 하는 약점 구성요소다.
    claim_keys가 주어지면 해당 청구항(독립항)만 본다.
    """
    if not cache:
        return set()

    soft = set()
    for claim_key, items in cache.items():
        if claim_keys is not None and claim_key not in claim_keys:
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            judgment = item.get("judgment", "대응 없음")
            if _JUDGMENT_SCORE.get(judgment, 0) == _SOFT_GAP_SCORE:
                soft.add((claim_key, normalize_label(item.get("label", ""))))
    return soft


def _compute_soft_improvement_score(
    cache: Optional[Dict],
    soft_gaps: set,
    weights: Optional[Dict[tuple[str, str], int]] = None,
) -> int:
    """소프트 공백 구성요소에 대해 '실질적 동일' 이상 판정을 가진 경우만 보강 인정.

    주인용발명(일부 차이=3)보다 명확히 좋은 판정일 때만 점수에 더해,
    단독으로 충분한 케이스에 불필요한 결합이 붙는 것을 막는다.
    """
    if not cache:
        return 0

    score = 0
    for (claim_key, label) in soft_gaps:
        items = cache.get(claim_key, [])
        if not isinstance(items, list):
            continue
        item = next((i for i in items if normalize_label(i.get("label")) == label), None)
        if item is None:
            continue
        j_score = _JUDGMENT_SCORE.get(item.get("judgment", "대응 없음"), 0)
        if j_score >= _SECONDARY_IMPROVE_THRESHOLD:
            score += j_score * (weights or {}).get((claim_key, label), 3)
    return score


def _compute_supporting_evidence_score(
    cache: Optional[Dict],
    target_gaps: set,
    weights: Optional[Dict[tuple[str, str], int]] = None,
) -> int:
    """주지관용보다 우선 검토할 보조 문헌의 불완전한 명시 근거 점수.

    '일부 차이' 이상이면 앞선 hard/soft 단계에서 이미 강한 보완 후보가 되므로,
    이 함수는 그보다 낮거나 동등하게 불완전하더라도 발췌가 있는 문헌 근거를
    결합 논리 프롬프트에 전달하기 위한 마지막 안전망이다.
    """
    if not cache:
        return 0

    score = 0
    for (claim_key, label) in target_gaps:
        items = cache.get(claim_key, [])
        if not isinstance(items, list):
            continue
        item = next((i for i in items if normalize_label(i.get("label")) == label), None)
        if item is None or not item.get("quote"):
            continue
        j_score = _JUDGMENT_SCORE.get(item.get("judgment", "대응 없음"), 0)
        judgment = item.get("judgment", "대응 없음")
        if judgment != "대응 없음" and j_score >= _SECONDARY_SUPPORT_THRESHOLD:
            # `차이`의 점수 0이 실제 발췌 근거까지 소거하지 않도록 최소 증거점수 1을 준다.
            score += max(1, j_score) * (weights or {}).get((claim_key, label), 3)
    return score


_CONTROL_CLUSTER_RE = re.compile(
    r"(?:mode|control|controller|process|processing|determin|detect|decision|"
    r"switch|select|convert|generate|output|adaptive|luminance|remosaic|"
    r"binning|depth|제어|판단|검출|모드|처리|생성|출력|변환|선택|스위칭|적응)",
    re.IGNORECASE,
)
_SENSOR_CLUSTER_RE = re.compile(
    r"(?:sensor|pixel|array|filter|photodiode|image sensor|센서|픽셀|어레이|필터|포토다이오드)",
    re.IGNORECASE,
)
_SIGNAL_CLUSTER_RE = re.compile(
    r"(?:signal|data|readout|sampling|demultiplex|map|신호|데이터|샘플링|역다중화|맵)",
    re.IGNORECASE,
)
_MECHANICAL_CLUSTER_RE = re.compile(
    r"(?:motor|wheel|gear|shaft|hinge|구동축|기어|휠|모터|힌지)",
    re.IGNORECASE,
)
_IO_CLUSTER_RE = re.compile(
    r"(?:display|transmit|receive|interface|memory|storage|표시|송신|수신|인터페이스|메모리|저장)",
    re.IGNORECASE,
)


def _infer_element_cluster(text: str) -> str:
    compact = " ".join((text or "").split())
    if not compact:
        return "generic"
    if _CONTROL_CLUSTER_RE.search(compact):
        return "control"
    if _SENSOR_CLUSTER_RE.search(compact):
        return "sensor"
    if _SIGNAL_CLUSTER_RE.search(compact):
        return "signal"
    if _MECHANICAL_CLUSTER_RE.search(compact):
        return "mechanical"
    if _IO_CLUSTER_RE.search(compact):
        return "io"
    return "generic"


def _target_cluster_map(claims: List[ParsedClaim], targets: set[tuple[str, str]]) -> Dict[tuple[str, str], str]:
    target_lookup = set(targets)
    result: Dict[tuple[str, str], str] = {}
    for claim in claims:
        claim_key = str(claim.claim_number)
        for element in claim.elements:
            key = (claim_key, normalize_label(element.label))
            if key not in target_lookup:
                continue
            result[key] = _infer_element_cluster(element.text or "")
    return result


def _cache_item(cache: Optional[Dict], claim_key: str, label: str) -> Optional[Dict]:
    items = (cache or {}).get(claim_key, [])
    if not isinstance(items, list):
        return None
    return next((i for i in items if normalize_label(i.get("label")) == label), None)


def _strongly_supported_labels(
    cache: Optional[Dict],
    claim_key: str,
    labels: set[str],
) -> set[str]:
    supported: set[str] = set()
    for label in labels:
        item = _cache_item(cache, claim_key, label)
        if not item or not item.get("quote"):
            continue
        if _JUDGMENT_SCORE.get(item.get("judgment", "대응 없음"), 0) >= _SECONDARY_IMPROVE_THRESHOLD:
            supported.add(label)
    return supported


def _best_conventional_support_doc(
    caches: Dict[int, Optional[Dict]],
    claim_key: str,
    target_labels: set[str],
    element_weights: Dict[tuple[str, str], int],
    excluded: set[int],
    num_docs: int,
) -> tuple[Optional[int], set[str]]:
    best_idx: Optional[int] = None
    best_labels: set[str] = set()
    best_score = 0
    for doc_idx in range(num_docs):
        if doc_idx in excluded:
            continue
        labels = _strongly_supported_labels(caches.get(doc_idx), claim_key, target_labels)
        score = sum(
            _JUDGMENT_SCORE.get(
                (_cache_item(caches.get(doc_idx), claim_key, label) or {}).get("judgment", "대응 없음"),
                0,
            ) * element_weights.get((claim_key, label), 2)
            for label in labels
        )
        if score > best_score:
            best_idx, best_labels, best_score = doc_idx, labels, score
    return best_idx, best_labels


def _apply_conventional_support_policy(
    chains: Dict[str, Dict],
    claims: List[ParsedClaim],
    caches: Dict[int, Optional[Dict]],
    num_docs: int,
    element_weights: Dict[tuple[str, str], int],
) -> List[int]:
    """Apply the exceptional third-reference policy to independent claims.

    Two references remain the normal ceiling. A third reference is admitted
    only when references 1 and 2 are already used for substantive limitations
    and the third strongly documents a residual, simple conventional component.
    """
    conventional_doc_order: List[int] = []

    for claim in (c for c in claims if c.claim_type == "independent"):
        claim_key = str(claim.claim_number)
        chain = chains.get(claim_key)
        if not chain or not chain.get("total"):
            continue
        if chain.get("selection_locked"):
            # 이미 발행된 독립항 보고서의 1·2·예외적 3문헌 구성은 종속항 추가나
            # 후속 정책 재계산으로 변경하지 않는다.
            continue

        elements_by_label = {
            normalize_label(element.label): element
            for element in claim.elements
        }
        primary_idx = chain["total"][0]
        primary_gap_labels = {
            label
            for label in elements_by_label
            if _JUDGMENT_SCORE.get(
                (_cache_item(caches.get(primary_idx), claim_key, label) or {}).get(
                    "judgment", "대응 없음"
                ),
                0,
            ) < _PRIMARY_COVER_THRESHOLD
        }
        conventional = {
            label: _conventionality_basis(elements_by_label[label])
            for label in primary_gap_labels
        }
        conventional = {label: basis for label, basis in conventional.items() if basis}
        if not conventional:
            chain["reference_roles"] = {
                str(doc_idx): "primary" if pos == 0 else "substantive_secondary"
                for pos, doc_idx in enumerate(chain["total"])
            }
            continue

        conventional_labels = set(conventional)
        nonconventional_labels = primary_gap_labels - conventional_labels
        original_secondary = chain["total"][1] if len(chain["total"]) > 1 else None

        # When every primary gap is conventional, use a second document only
        # if it gives strong, explicit evidence. Otherwise use the dedicated
        # primary-reference + common-general-knowledge combination template.
        if not nonconventional_labels:
            support_idx, support_labels = _best_conventional_support_doc(
                caches,
                claim_key,
                conventional_labels,
                element_weights,
                {primary_idx},
                num_docs,
            )
            chain["total"] = [primary_idx] + ([support_idx] if support_idx is not None else [])
            chain["added"] = chain["total"][:]
            chain["reference_roles"] = {str(primary_idx): "primary"}
            if support_idx is not None:
                chain["reference_roles"][str(support_idx)] = "conventional_support"
                chain["conventional_support"] = {
                    "doc_idx": support_idx,
                    "position": 2,
                    "role": "conventional_support",
                    "labels": sorted(support_labels),
                    "basis": {label: conventional[label] for label in sorted(support_labels)},
                }
                conventional_doc_order.append(support_idx)
            unsupported = conventional_labels - support_labels
            if unsupported:
                chain["common_general_knowledge"] = [
                    {
                        "label": label,
                        "text": elements_by_label[label].text,
                        "basis": conventional[label],
                    }
                    for label in sorted(unsupported)
                ]
            rationale_type = (
                "conventional_support" if support_idx is not None
                else "common_general_knowledge"
            )
            chain["combination_rationale"] = _combination_rationale_for(
                None, candidate_types=[rationale_type]
            )
            chain["combination_rationale_type"] = rationale_type
            continue

        selected = [primary_idx]
        secondary_is_substantive = False
        if original_secondary is not None:
            secondary_nonconventional_evidence = {
                label
                for label in nonconventional_labels
                if (
                    (_cache_item(caches.get(original_secondary), claim_key, label) or {}).get("quote")
                    and _JUDGMENT_SCORE.get(
                        (_cache_item(caches.get(original_secondary), claim_key, label) or {}).get(
                            "judgment", "대응 없음"
                        ),
                        0,
                    ) >= _SECONDARY_FILL_THRESHOLD
                )
            }
            if secondary_nonconventional_evidence:
                selected.append(original_secondary)
                secondary_is_substantive = True

        # If no substantive second reference survived, a strong conventional
        # document may be used as reference 2, but never as a substitute for the
        # still-unresolved inventive limitation.
        if not secondary_is_substantive:
            support_idx, support_labels = _best_conventional_support_doc(
                caches,
                claim_key,
                conventional_labels,
                element_weights,
                {primary_idx},
                num_docs,
            )
            if support_idx is not None:
                selected.append(support_idx)
                chain["conventional_support"] = {
                    "doc_idx": support_idx,
                    "position": 2,
                    "role": "conventional_support",
                    "labels": sorted(support_labels),
                    "basis": {label: conventional[label] for label in sorted(support_labels)},
                }
                conventional_doc_order.append(support_idx)
            chain["total"] = selected[:MAX_INDEPENDENT_REFS]
            chain["added"] = chain["total"][:]
            chain["reference_roles"] = {
                str(doc_idx): "primary" if pos == 0 else "conventional_support"
                for pos, doc_idx in enumerate(chain["total"])
            }
            supported = set((chain.get("conventional_support") or {}).get("labels", []))
            unsupported = conventional_labels - supported
            if unsupported:
                chain["common_general_knowledge"] = [
                    {"label": label, "text": elements_by_label[label].text, "basis": conventional[label]}
                    for label in sorted(unsupported)
                ]
            rationale_type = (
                "conventional_support" if chain.get("conventional_support")
                else "common_general_knowledge"
            )
            chain["combination_rationale"] = _combination_rationale_for(
                None, candidate_types=[rationale_type]
            )
            chain["combination_rationale_type"] = rationale_type
            continue

        chain["total"] = selected[:MAX_INDEPENDENT_REFS]
        chain["added"] = chain["total"][:]
        chain["reference_roles"] = {
            str(primary_idx): "primary",
            str(original_secondary): "substantive_secondary",
        }
        residual_conventional = {
            label
            for label in conventional_labels
            if max(
                _JUDGMENT_SCORE.get(
                    (_cache_item(caches.get(doc_idx), claim_key, label) or {}).get(
                        "judgment", "대응 없음"
                    ),
                    0,
                )
                for doc_idx in selected
            ) < _PRIMARY_COVER_THRESHOLD
        }
        third_idx, third_labels = _best_conventional_support_doc(
            caches,
            claim_key,
            residual_conventional,
            element_weights,
            set(selected),
            num_docs,
        )
        if third_idx is not None and len(chain["total"]) == MAX_INDEPENDENT_REFS:
            chain["total"] = (chain["total"] + [third_idx])[:MAX_INDEPENDENT_REFS_WITH_CONVENTIONAL_SUPPORT]
            chain["added"] = chain["total"][:]
            chain["reference_roles"][str(third_idx)] = "conventional_support"
            chain["conventional_support"] = {
                "doc_idx": third_idx,
                "position": 3,
                "role": "conventional_support",
                "labels": sorted(third_labels),
                "basis": {label: conventional[label] for label in sorted(third_labels)},
            }
            conventional_doc_order.append(third_idx)
        unsupported = residual_conventional - third_labels
        if unsupported:
            chain["common_general_knowledge"] = [
                {"label": label, "text": elements_by_label[label].text, "basis": conventional[label]}
                for label in sorted(unsupported)
            ]

    return list(dict.fromkeys(conventional_doc_order))


def _score_secondary_candidate_base(
    cache: Optional[Dict],
    primary_cache: Optional[Dict],
    claims: List[ParsedClaim],
    primary_gaps: set,
    soft_gaps: set,
    weights: Optional[Dict[tuple[str, str], int]] = None,
) -> tuple[float, Dict]:
    targets = primary_gaps | soft_gaps
    if not cache or not targets:
        return 0.0, {}

    weighted_gain = 0.0
    weighted_residual = 0.0
    weighted_evidence = 0.0
    denominator = 0.0
    hard_fill = 0
    soft_fill = 0
    specific_fill = 0
    filled_labels: list[str] = []

    for claim_key, label in targets:
        base_weight = float((weights or {}).get((claim_key, label), 3))
        if (claim_key, label) in soft_gaps and (claim_key, label) not in primary_gaps:
            base_weight *= 0.65
        denominator += base_weight

        item = _cache_item(cache, claim_key, label)
        judgment = item.get("judgment", "대응 없음") if item else "대응 없음"
        sim = _item_similarity(item)
        rank = _JUDGMENT_SCORE.get(judgment, 0)
        primary_item = _cache_item(primary_cache, claim_key, label)
        primary_sim = _item_similarity(primary_item)
        gain = max(0.0, sim - primary_sim)
        weighted_gain += base_weight * gain
        weighted_residual += base_weight * sim
        evidence_factor = (
            0.55
            + 0.25 * bool(item and item.get("quote"))
            + 0.10 * bool(item and (item.get("chunk_id") or item.get("paragraph_no")))
            + 0.10 * bool(item and (item.get("판단_이유") or item.get("similarity_reason")))
        )
        weighted_evidence += base_weight * gain * evidence_factor
        if rank >= _SECONDARY_FILL_THRESHOLD and (claim_key, label) in primary_gaps:
            hard_fill += 1
            filled_labels.append(label)
        if rank >= _SECONDARY_IMPROVE_THRESHOLD and (claim_key, label) in soft_gaps:
            soft_fill += 1
            filled_labels.append(label)

        if 0.15 < primary_sim <= 0.55 and sim >= 0.85:
            specific_fill += 1

    marginal_gain = weighted_gain / denominator if denominator else 0.0
    residual_coverage = weighted_residual / denominator if denominator else 0.0
    evidence_gain = weighted_evidence / denominator if denominator else 0.0
    sub_score = 0.60 * marginal_gain + 0.20 * residual_coverage + 0.20 * evidence_gain

    rationale_types: list[str] = []
    if hard_fill:
        rationale_types.append("gap_filling")
    if soft_fill:
        rationale_types.append("known_tech_application")
    if specific_fill:
        rationale_types.append("specific_selection")
    if len(set(filled_labels)) >= 2:
        rationale_types.append("aggregation")
    if not rationale_types and evidence_gain > 0:
        rationale_types.append("supporting_evidence")

    warnings: list[str] = []
    if marginal_gain > 0 and evidence_gain < marginal_gain * 0.75:
        warnings.append("점수 증가분에 비해 발췌·위치·판단 근거가 부족하므로 원문 근거를 점검해야 합니다.")

    if hard_fill:
        reason = "hard"
    elif soft_fill:
        reason = "soft"
    elif evidence_gain > 0:
        reason = "support"
    else:
        reason = None

    detail = {
        "sub_score": round(sub_score, 4),
        "marginal_gain": round(marginal_gain, 4),
        "residual_coverage": round(residual_coverage, 4),
        "evidence_adjusted_gain": round(evidence_gain, 4),
        "formula": "0.60*marginal_gain + 0.20*residual_coverage + 0.20*evidence_adjusted_gain",
        "hard_fill_count": hard_fill,
        "soft_fill_count": soft_fill,
        "candidate_rationale_types": list(dict.fromkeys(rationale_types)),
        "warnings": warnings,
        "secondary_reason": reason,
    }
    return round(sub_score * 100, 2), detail


def _critical_gap_evidence_gain(
    cache: Optional[Dict],
    primary_cache: Optional[Dict],
    critical_gaps: set,
    weights: Optional[Dict[tuple[str, str], float]] = None,
) -> float:
    """Return the weighted improvement for intrinsically technical core gaps only."""
    gain = 0.0
    for claim_key, label in critical_gaps:
        weight = float((weights or {}).get((claim_key, label), 3))
        candidate_item = _cache_item(cache, claim_key, label)
        primary_item = _cache_item(primary_cache, claim_key, label)
        candidate_similarity = _item_similarity(candidate_item)
        primary_similarity = _item_similarity(primary_item)
        if candidate_item and candidate_item.get("quote"):
            gain += weight * max(0.0, candidate_similarity - primary_similarity)
    return round(gain, 4)


def _build_gap_evidence_matrix(
    caches: Dict[int, Optional[Dict]],
    claims: List[ParsedClaim],
    primary_idx: int,
    num_docs: int,
) -> Dict[str, Dict]:
    """사건별 공백·보완 근거를 사람이 재검토할 수 있는 형태로 보존한다."""
    matrix: Dict[str, Dict] = {}
    for claim in claims:
        claim_key = str(claim.claim_number)
        rows = []
        for element in claim.elements:
            label = normalize_label(element.label)
            primary_item = _cache_item(caches.get(primary_idx), claim_key, label) or {}
            candidates = []
            for doc_idx in range(num_docs):
                if doc_idx == primary_idx:
                    continue
                item = _cache_item(caches.get(doc_idx), claim_key, label) or {}
                evidence = [
                    {
                        "limitation": str(ev.get("limitation", "") or ""),
                        "quote": str(ev.get("quote", "") or ""),
                        "chunk_id": str(ev.get("chunk_id", "") or ""),
                    }
                    for ev in (item.get("evidence") or [])
                    if isinstance(ev, dict) and ev.get("quote")
                ]
                if item.get("quote") or evidence:
                    candidates.append({
                        "doc_idx": doc_idx,
                        "judgment": item.get("judgment", "대응 없음"),
                        "reason": item.get("판단_이유", item.get("similarity_reason", "")),
                        "quote": item.get("quote", ""),
                        "chunk_id": item.get("chunk_id", ""),
                        "evidence": evidence,
                    })
            rows.append({
                "label": element.label,
                "claim_limitation": element.text,
                "importance": element.importance,
                "primary_judgment": primary_item.get("judgment", "대응 없음"),
                "primary_reason": primary_item.get("판단_이유", primary_item.get("similarity_reason", "")),
                "primary_evidence": primary_item.get("evidence") or [],
                "candidate_evidence": candidates,
            })
        matrix[claim_key] = {"elements": rows}
    return matrix


def _score_secondary_candidate(
    cache: Optional[Dict],
    primary_cache: Optional[Dict],
    claims: List[ParsedClaim],
    primary_gaps: set,
    soft_gaps: set,
    weights: Optional[Dict[tuple[str, str], int]] = None,
) -> tuple[float, Dict]:
    base_score, detail = _score_secondary_candidate_base(
        cache,
        primary_cache,
        claims,
        primary_gaps,
        soft_gaps,
        weights,
    )
    targets = primary_gaps | soft_gaps
    if not cache or not targets:
        return base_score, detail

    denominator = 0.0
    filled_target_count = 0
    filled_target_weight = 0.0
    motivation_weight = 0.0
    explicit_risk_labels: List[str] = []
    filled_cluster_weights: Dict[str, float] = {}
    target_clusters = _target_cluster_map(claims, targets)

    for claim_key, label in targets:
        base_weight = float((weights or {}).get((claim_key, label), 3))
        if (claim_key, label) in soft_gaps and (claim_key, label) not in primary_gaps:
            base_weight *= 0.65
        denominator += base_weight

        item = _cache_item(cache, claim_key, label)
        judgment = item.get("judgment", "\ub300\uc751 \uc5c6\uc74c") if item else "\ub300\uc751 \uc5c6\uc74c"
        sim = _similarity_for_judgment(judgment)
        if sim < 0.35:
            continue

        filled_target_count += 1
        filled_target_weight += base_weight
        if item and item.get("motivation_quote"):
            motivation_weight += base_weight
        risk = str((item or {}).get("combination_risk") or "").strip().lower()
        if risk in {"contrary_teaching", "principle_change", "incompatible"}:
            explicit_risk_labels.append(f"{claim_key}:{label}:{risk}")
        cluster = target_clusters.get((claim_key, label), "generic")
        filled_cluster_weights[cluster] = filled_cluster_weights.get(cluster, 0.0) + base_weight

    breadth = filled_target_weight / denominator if denominator else 0.0
    cluster_consistency = 0.0
    if filled_target_weight > 0 and filled_target_count >= 2:
        cluster_consistency = max(filled_cluster_weights.values()) / filled_target_weight
    single_feature_dominance_penalty = 0.0
    if len(targets) >= 3 and filled_target_count <= 1 and base_score > 0:
        single_feature_dominance_penalty = min(12.0, 8.0 + 0.06 * base_score)

    motivation_evidence = motivation_weight / denominator if denominator else 0.0
    explicit_risk_penalty = 40.0 if explicit_risk_labels else 0.0
    adjusted_score = (
        base_score
        + (8.0 * breadth)
        + (7.0 * cluster_consistency)
        + (3.0 * motivation_evidence)
        - single_feature_dominance_penalty
        - explicit_risk_penalty
    )
    enriched_detail = dict(detail)
    enriched_detail["sub_score"] = round(adjusted_score / 100.0, 4)
    enriched_detail["residual_breadth"] = round(breadth, 4)
    enriched_detail["cluster_consistency"] = round(cluster_consistency, 4)
    enriched_detail["single_feature_dominance_penalty"] = round(single_feature_dominance_penalty / 100.0, 4)
    enriched_detail["motivation_evidence"] = round(motivation_evidence, 4)
    enriched_detail["explicit_combination_risks"] = explicit_risk_labels
    if explicit_risk_labels:
        enriched_detail.setdefault("warnings", []).append(
            "보조문헌에 반대 교시 또는 기본 작동원리 변경 위험이 명시되어 문헌쌍 채택에서 제외해야 합니다."
        )
    return round(adjusted_score, 2), enriched_detail


def _combination_rationale_for(
    reason: Optional[str],
    candidate_types: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
    score_detail: Optional[Dict] = None,
) -> Dict:
    code_by_reason = {
        "hard": "gap_filling",
        "soft": "known_tech_application",
        "support": "supporting_evidence",
        "insufficient": "insufficient_support",
        None: "single_reference",
    }
    code = (candidate_types or [None])[0] or code_by_reason.get(reason, "gap_filling")
    if code not in _COMBINATION_RATIONALES:
        code = code_by_reason.get(reason, "gap_filling")
    data = dict(_COMBINATION_RATIONALES[code])
    data["type"] = code
    data["candidate_types"] = candidate_types or [code]
    data["warnings"] = warnings or []
    data["score_detail"] = score_detail or {}
    data["secondary_reason"] = reason
    return data


def _element_percents(cache: Optional[Dict], claim: ParsedClaim) -> Dict[str, float]:
    """캐시 판정을 구성요소 라벨 → 결손 보정 유사도로 변환한다.

    판정 라벨만 사용하면 `일부 유사`가 누락 제한과 직접성에 관계없이 경계값
    35로 고정되어 완전 커버로 오인될 수 있다. 하위 제한 evidence/missing 및
    directness를 반영하는 _item_similarity를 사용해 그 모순을 방지한다.
    """
    if not cache:
        return {}
    items = cache.get(str(claim.claim_number), [])
    if not isinstance(items, list):
        return {}
    out: Dict[str, float] = {}
    for item in items:
        if isinstance(item, dict):
            label = normalize_label(item.get("label"))
            out[label] = _item_similarity(item) * 100.0
    return out


def _claim_similarity(
    primary_cache: Optional[Dict],
    secondary_cache: Optional[Dict],
    claim: ParsedClaim,
    additional_caches: Optional[List[Optional[Dict]]] = None,
) -> Dict:
    """청구항 1개에 대한 주인용발명 단독/결합 가중 유사도를 계산한다 (LLM 호출 없음).

    가중치는 구성요소 importance(5/3/2)를 사용해 핵심 구성이 점수를 지배하게 한다.
    결합 유사도는 구성요소별 max(주, 보조) 퍼센트로 집계한다.
    """
    selected_caches = [primary_cache, secondary_cache, *(additional_caches or [])]
    p_pcts = _element_percents(primary_cache, claim)
    supporting_percent_maps = [_element_percents(cache, claim) for cache in selected_caches[1:]]

    p_num = c_num = den = 0.0
    uncovered: List[str] = []
    residual: List[str] = []
    for e in claim.elements:
        imp = int(e.importance) if str(e.importance).isdigit() else 3
        label = normalize_label(e.label)
        p = p_pcts.get(label, 0)
        c = max(p, *(pcts.get(label, 0) for pcts in supporting_percent_maps))
        p_num += imp * p
        c_num += imp * c
        den += imp * 100
        # 유사도 점수는 문헌쌍의 순위를 정하는 보조값일 뿐, 청구항 한정의
        # 완전 보완 여부를 대신할 수 없다. 특히 `일부 차이`는 발췌가 있어도
        # missing_limitations가 남아 있는 판정이므로 결합 후 잔여 구성으로
        # 보존한다. 선택 문헌 중 하나가 해당 구성을 `실질적 동일` 이상으로
        # 직접·완전하게 개시한 경우에만 보완 완료로 본다.
        fully_covered = False
        for cache in selected_caches:
            item = _cache_item(cache, str(claim.claim_number), label)
            if not item:
                continue
            judgment_rank = _JUDGMENT_SCORE.get(item.get("judgment", "대응 없음"), 0)
            missing = [value for value in (item.get("missing_limitations") or []) if str(value).strip()]
            if judgment_rank >= _SECONDARY_IMPROVE_THRESHOLD and not missing:
                fully_covered = True
                break
        if not fully_covered:
            residual.append(e.label)
        if c <= UNCOVERED_PERCENT_THRESHOLD:
            uncovered.append(e.label)

    if den == 0:
        return {
            "primary_similarity": 0.0,
            "combined_similarity": 0.0,
            "uncovered_labels": uncovered,
            "residual_labels": residual,
        }
    return {
        "primary_similarity": round(p_num / den * 100, 1),
        "combined_similarity": round(c_num / den * 100, 1),
        "uncovered_labels": uncovered,
        "residual_labels": residual,
    }


def _is_full_coverage(
    primary_cache: Optional[Dict],
    secondary_cache: Optional[Dict],
    claim_keys: Optional[set] = None,
) -> bool:
    """
    주인용발명 + 보조인용발명 결합 시 모든 구성요소가 커버되는지 확인.
    결합 커버 = (주 OR 보조) 각 구성요소에서 _SECONDARY_FILL_THRESHOLD 이상
    claim_keys가 주어지면 해당 청구항(독립항)만 본다.
    """
    if not primary_cache:
        return False

    for claim_key, items in primary_cache.items():
        if claim_keys is not None and claim_key not in claim_keys:
            continue
        if not isinstance(items, list):
            continue
        secondary_items = (secondary_cache or {}).get(claim_key, [])

        for item in items:
            label = item.get("label", "")
            p_score = _JUDGMENT_SCORE.get(item.get("judgment", "대응 없음"), 0)
            if p_score >= _PRIMARY_COVER_THRESHOLD:
                continue  # 주인용발명이 커버 → OK

            # 주인용발명이 못 커버하는 요소: 보조인용발명이 채우는지 확인
            s_item = next(
                (i for i in secondary_items
                 if normalize_label(i.get("label")) == normalize_label(label)),
                None,
            )
            s_score = _JUDGMENT_SCORE.get(s_item.get("judgment", "대응 없음"), 0) if s_item else 0
            if s_score < _SECONDARY_FILL_THRESHOLD:
                return False  # 둘 다 커버 못함 → 완전 커버 불가

    return True


_NOVELTY_DIRECT_JUDGMENTS = {"동일", "실질적 동일"}


def _single_document_disclosure(
    cache: Optional[Dict],
    claim: ParsedClaim,
) -> Dict:
    """단일 문헌이 청구항의 모든 필수 구성을 직접 개시하는지 심사한다.

    평균 유사도나 다른 문헌의 보완 가능성은 사용하지 않는다. 각 구성요소마다
    직접 근거, 동일·실질적 동일 판정, 누락 제한 부재가 모두 확인되어야 한다.
    이 게이트를 통과한 문헌만 신규성 판단용 단일 문헌 후보가 된다.
    """
    claim_key = str(claim.claim_number)
    items = (cache or {}).get(claim_key, [])
    by_label = _items_by_label(items if isinstance(items, list) else [])
    rows: List[Dict] = []

    for element in claim.elements:
        label = normalize_label(element.label)
        item = by_label.get(label) or {}
        judgment = item.get("judgment", "대응 없음")
        directness = str(item.get("directness") or "").strip().lower()
        missing = [str(value).strip() for value in (item.get("missing_limitations") or []) if str(value).strip()]
        has_direct_quote = bool(item.get("quote"))
        not_evaluated = bool(item.get("not_evaluated")) or str(
            item.get("evaluation_status") or ""
        ).startswith("not_evaluated")
        directly_disclosed = bool(
            not not_evaluated
            and has_direct_quote
            and judgment in _NOVELTY_DIRECT_JUDGMENTS
            and directness in {"", "direct"}
            and not missing
        )
        rows.append({
            "label": element.label,
            "judgment": judgment,
            "directness": directness or "direct",
            "has_quote": has_direct_quote,
            "missing_limitations": missing,
            "not_evaluated": not_evaluated,
            "evaluation_status": item.get("evaluation_status", "evaluated"),
            "directly_disclosed": directly_disclosed,
        })

    missing_labels = [row["label"] for row in rows if not row["directly_disclosed"]]
    return {
        "is_complete": bool(rows) and not missing_labels,
        "directly_disclosed_labels": [row["label"] for row in rows if row["directly_disclosed"]],
        "missing_or_indirect_labels": missing_labels,
        "not_evaluated_labels": [row["label"] for row in rows if row["not_evaluated"]],
        "elements": rows,
        "rule": "단일 문헌의 직접·명백한 완전 개시만 신규성 후보로 인정",
    }


def _novelty_screen(
    claim: ParsedClaim,
    caches: Dict[int, Optional[Dict]],
    num_docs: int,
) -> Dict:
    """모든 문헌을 서로 결합하지 않고 독립적으로 신규성 심사한다."""
    candidates: List[int] = []
    assessments: Dict[str, Dict] = {}
    scores: Dict[int, float] = {}
    for doc_idx in range(num_docs):
        assessment = _single_document_disclosure(caches.get(doc_idx), claim)
        assessments[str(doc_idx)] = assessment
        score, _count, _detail = _score_prior_cache(caches.get(doc_idx), [claim])
        scores[doc_idx] = score
        if assessment["is_complete"]:
            candidates.append(doc_idx)

    selected = max(candidates, key=lambda idx: (scores.get(idx, 0.0), -idx)) if candidates else None
    return {
        "selected_document": selected,
        "complete_documents": candidates,
        "document_assessments": assessments,
        "result": "single_document_complete" if selected is not None else "no_single_document_complete",
    }


# ---------------------------------------------------------------------------
# 청구항 패밀리 및 문헌쌍 평가
# ---------------------------------------------------------------------------

def _claim_family_groups(claims: List[ParsedClaim]) -> tuple[Dict[int, List[ParsedClaim]], List[ParsedClaim]]:
    """독립항을 루트로 하는 청구항 패밀리와 부모가 없는 종속항을 분리한다."""
    by_number = {claim.claim_number: claim for claim in claims}
    families: Dict[int, List[ParsedClaim]] = {
        claim.claim_number: [claim]
        for claim in claims
        if claim.claim_type == "independent"
    }
    orphans: List[ParsedClaim] = []

    for claim in claims:
        if claim.claim_type == "independent":
            continue
        seen: set[int] = set()
        current = claim
        root_number: Optional[int] = None
        while current.parent_claim and current.parent_claim not in seen:
            seen.add(current.parent_claim)
            parent = by_number.get(current.parent_claim)
            if parent is None:
                break
            if parent.claim_type == "independent":
                root_number = parent.claim_number
                break
            current = parent
        if root_number is None:
            orphans.append(claim)
        else:
            families.setdefault(root_number, []).append(claim)

    for family_claims in families.values():
        family_claims.sort(key=lambda item: (0 if item.claim_type == "independent" else 1, item.claim_number))
    return families, orphans


def _dependent_reuse_score(cache: Optional[Dict], dependent_claims: List[ParsedClaim]) -> float:
    """여러 종속항의 고유 추가 한정을 직접 뒷받침하는 정도를 0~1로 산정한다."""
    if not cache or not dependent_claims:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for depth_order, claim in enumerate(sorted(dependent_claims, key=lambda item: item.claim_number)):
        claim_key = str(claim.claim_number)
        by_label = _items_by_label(cache.get(claim_key, []) if isinstance(cache.get(claim_key, []), list) else [])
        # 직접 종속항을 조금 더 중시하되 깊은 종속항도 배제하지 않는다.
        depth_factor = max(0.65, 1.0 - 0.05 * depth_order)
        for element in claim.elements:
            weight = float(_importance_value(element.importance)) * depth_factor
            denominator += weight
            item = by_label.get(normalize_label(element.label), {})
            if not item.get("quote") or str(item.get("directness") or "direct").lower() == "absent":
                continue
            numerator += weight * _item_similarity(item)
    return numerator / denominator if denominator else 0.0


def _explicit_combination_risks(
    cache: Optional[Dict],
    targets: set,
) -> List[str]:
    """비교 캐시에 명시적으로 저장된 반대 교시만 문헌쌍 위험으로 사용한다."""
    risks: List[str] = []
    if not cache:
        return risks
    for claim_key, label in targets:
        item = _cache_item(cache, claim_key, label) or {}
        risk = str(item.get("combination_risk") or "").strip().lower()
        if risk in {"contrary_teaching", "incompatible", "principle_change"}:
            risks.append(f"{claim_key}:{label}:{risk}")
    return risks


def _compute_family_context(
    family_claims: List[ParsedClaim],
    caches: Dict[int, Optional[Dict]],
    num_docs: int,
) -> Optional[Dict]:
    """패밀리 단위의 가중치·주인용 점수·신규성 심사 결과를 한 번에 계산한다.

    문헌쌍 선정(_select_family_reference_pair)과 LLM 판정용 후보 추출
    (shortlist_primary_candidates)이 같은 값을 두 번 계산하지 않도록 분리했다.
    LLM 호출은 하지 않는 순수 함수이므로 골든셋 회귀 대상이 된다.
    """
    root = next((claim for claim in family_claims if claim.claim_type == "independent"), None)
    if root is None:
        return None
    dependent_claims = [claim for claim in family_claims if claim.claim_type == "dependent"]
    root_key = str(root.claim_number)
    weights = _element_weight_map([root])

    # 사건 내 문헌 희소성과 구성의 자체 복잡도로 차별적 핵심을 재식별한다.
    # 짧은 입출력 사양은 파서가 importance=5로 내더라도 상한을 둔다. 반대로
    # 뒤쪽에 있는 수치·관계·효과 결합 구성은 최소 핵심 가중치를 보장한다.
    # 희소성은 실제 원문 근거가 하나 이상 있을 때만 보너스로 쓰므로, 통합 비교의
    # 거짓 음성이 "희소한 핵심"으로 오인되어 가중치를 왜곡하지 않는다.
    dynamic_weights: Dict[tuple[str, str], float] = {}
    dynamic_weight_reasons: Dict[tuple[str, str], str] = {}
    disclosure_frequency: Dict[tuple[str, str], float] = {}
    for claim in [root]:
        claim_key = str(claim.claim_number)
        for element in claim.elements:
            label = normalize_label(element.label)
            match_count = 0
            evaluated_count = 0
            for doc_idx in range(num_docs):
                doc_cache = caches.get(doc_idx)
                if doc_cache:
                    item = _cache_item(doc_cache, claim_key, label) or {}
                    if item.get("not_evaluated") or str(
                        item.get("evaluation_status") or ""
                    ).startswith("not_evaluated"):
                        continue
                    evaluated_count += 1
                    directness = str(item.get("directness") or "direct").strip().lower()
                    if (
                        item.get("quote")
                        and directness != "absent"
                        and _similarity_for_judgment(item.get("judgment", "대응 없음")) >= 0.55
                    ):
                        match_count += 1

            match_ratio = match_count / evaluated_count if evaluated_count > 0 else 1.0
            disclosure_frequency[(claim_key, label)] = match_ratio
            original_importance = _importance_value(element.importance)

            is_generic_interface = _is_generic_interface_element(element)
            is_simple_component = _is_simple_conventional_component(element)
            is_distinctive = _is_distinctive_technical_element(element)
            if is_generic_interface:
                adjusted = min(2.0, max(1.0, original_importance * 0.5))
                weight_reason = "short_generic_interface_cap"
            elif is_simple_component:
                adjusted = min(2.0, max(1.0, original_importance * 0.4))
                weight_reason = "bare_conventional_component_cap"
            elif is_distinctive:
                adjusted = max(4.0, float(original_importance))
                if match_count > 0 and match_ratio <= 0.50:
                    adjusted = min(10.0, adjusted * 1.25)
                    weight_reason = "distinctive_with_rare_direct_evidence"
                elif match_ratio >= 0.70:
                    adjusted = max(4.0, adjusted * 0.85)
                    weight_reason = "distinctive_but_widely_disclosed"
                else:
                    weight_reason = "distinctive_minimum_core_weight"
            elif match_ratio >= 0.70:
                adjusted = max(1.0, original_importance * 0.5)
                weight_reason = "widely_disclosed_noncore"
            else:
                adjusted = float(original_importance)
                weight_reason = "declared_importance"

            dynamic_weights[(claim_key, label)] = adjusted
            dynamic_weight_reasons[(claim_key, label)] = weight_reason

    # 보조문헌의 공백 보완 점수도 같은 차별 구성 가중치를 사용해야, 주문헌은
    # 핵심 기준으로 고르면서 보조문헌은 범용 입출력 기준으로 고르는 불일치를 막는다.
    weights.update(dynamic_weights)

    primary_scores: Dict[int, float] = {}
    primary_details: Dict[int, Dict] = {}
    reuse_scores: Dict[int, float] = {}
    for doc_idx in range(num_docs):
        score, _count, detail = _score_prior_cache(caches.get(doc_idx), [root], dynamic_weights)
        detail["dynamic_weight_by_label"] = {
            label: round(weight, 4)
            for (claim_key, label), weight in dynamic_weights.items()
            if claim_key == root_key
        }
        detail["dynamic_weight_reason_by_label"] = {
            label: reason
            for (claim_key, label), reason in dynamic_weight_reasons.items()
            if claim_key == root_key
        }
        detail["direct_disclosure_frequency_by_label"] = {
            label: round(frequency, 4)
            for (claim_key, label), frequency in disclosure_frequency.items()
            if claim_key == root_key
        }
        detail["match_count"] = _count
        primary_scores[doc_idx] = score
        primary_details[doc_idx] = detail
        reuse_scores[doc_idx] = _dependent_reuse_score(caches.get(doc_idx), dependent_claims)

    # 1단계는 점수 경쟁이 아니라 단일 문헌 완전개시 심사다. 이 단계에서
    # 통과 문헌이 있으면 다른 문헌을 붙이지 않고 신규성용 단일 체인으로 확정한다.
    novelty_screen = _novelty_screen(root, caches, num_docs)

    return {
        "root": root,
        "root_key": root_key,
        "family_claims": family_claims,
        "weights": weights,
        "dynamic_weights": dynamic_weights,
        "dynamic_weight_reasons": dynamic_weight_reasons,
        "disclosure_frequency": disclosure_frequency,
        "primary_scores": primary_scores,
        "primary_details": primary_details,
        "reuse_scores": reuse_scores,
        "novelty_screen": novelty_screen,
    }


def _core_disclosure_labels(cache: Optional[Dict], root: ParsedClaim) -> List[str]:
    """차별적 핵심 구성을 원문 발췌로 직접 개시한 라벨 목록."""
    root_key = str(root.claim_number)
    labels: List[str] = []
    for element in root.elements:
        if not _is_distinctive_technical_element(element):
            continue
        label = normalize_label(element.label)
        item = _cache_item(cache, root_key, label)
        if not item or not item.get("quote"):
            continue
        if str(item.get("directness") or "direct").strip().lower() == "absent":
            continue
        if _item_similarity(item) >= CORE_DISCLOSURE_FORCE_INCLUDE_SIMILARITY:
            labels.append(label)
    return labels


def _core_disclosure_indices(
    caches: Dict[int, Optional[Dict]], root: ParsedClaim, num_docs: int
) -> List[int]:
    """차별적 핵심 구성을 원문으로 직접 개시한 문헌 인덱스."""
    return [
        doc_idx for doc_idx in range(num_docs)
        if _core_disclosure_labels(caches.get(doc_idx), root)
    ]


def _eligible_primary_indices(
    primary_details: Dict[int, Dict],
    num_docs: int,
    forced: Optional[List[int]] = None,
) -> List[int]:
    """주인용 자격을 통과한 문헌 인덱스. 최소 1개는 반드시 반환한다.

    `forced`는 차별적 핵심을 원문으로 직접 개시한 문헌이다. 자격 점수는
    핵심 구성 전체의 가중 평균이라, 핵심 하나를 정확히 짚었지만 나머지를
    놓친 문헌이 평균에 희석되어 탈락할 수 있다. 그런 문헌까지 주인용 후보에서
    빼면 자격 게이트가 원래 막으려던 것(명백히 열등한 주인용)이 아니라
    유효한 후보를 막게 되므로 항상 통과시킨다.
    """
    scores = {
        idx: float((primary_details.get(idx) or {}).get("distinctive_direct_coverage", 0.0))
        for idx in range(num_docs)
    }
    if not scores:
        return list(range(num_docs))
    best = max(scores.values())
    if best <= 0.0:
        # 어느 문헌도 핵심을 직접 개시하지 못한 사건에서는 자격 판단의 근거
        # 자체가 없으므로 게이트를 적용하지 않고 전수 탐색으로 되돌린다.
        return list(range(num_docs))
    eligible = {
        idx for idx, value in scores.items()
        if value >= best * PRIMARY_ELIGIBILITY_RATIO
        or value >= best - PRIMARY_ELIGIBILITY_ABS_MARGIN
    }
    eligible.update(idx for idx in (forced or []) if 0 <= idx < num_docs)
    return sorted(eligible) or [max(scores, key=lambda key: scores[key])]


def shortlist_primary_candidates(
    family_claims: List[ParsedClaim],
    caches: Dict[int, Optional[Dict]],
    num_docs: int,
    context: Optional[Dict] = None,
) -> Dict:
    """LLM 주인용 판정(LLM-A)에 넘길 후보를 추린다. LLM 호출 없는 순수 함수.

    자격 게이트를 통과한 문헌을 차별적 핵심 직접 개시량 순으로 정렬하고,
    전체 점수가 낮더라도 핵심 구성을 원문으로 직접 개시한 문헌은 정렬 순위와
    무관하게 후보에 포함시킨다(알고리즘 정렬이 놓친 문헌을 구제하는 장치).
    """
    context = context or _compute_family_context(family_claims, caches, num_docs)
    if not context:
        return {}
    root = context["root"]
    primary_details = context["primary_details"]
    primary_scores = context["primary_scores"]
    novelty_idx = context["novelty_screen"].get("selected_document")

    core_disclosers = _core_disclosure_indices(caches, root, num_docs)
    eligible = _eligible_primary_indices(primary_details, num_docs, core_disclosers)

    def _merit(doc_idx: int) -> tuple[float, float, float]:
        detail = primary_details.get(doc_idx) or {}
        return (
            float(detail.get("distinctive_direct_coverage", 0.0)),
            float(detail.get("core_coverage", 0.0)),
            float(primary_scores.get(doc_idx, 0.0)),
        )

    ordered = sorted(eligible, key=_merit, reverse=True)[:PRIMARY_SHORTLIST_MAX]

    forced: List[int] = []
    core_labels_by_doc: Dict[int, List[str]] = {}
    for doc_idx in range(num_docs):
        labels = _core_disclosure_labels(caches.get(doc_idx), root)
        core_labels_by_doc[doc_idx] = labels
        if labels and doc_idx not in ordered:
            forced.append(doc_idx)
    for doc_idx in sorted(forced, key=_merit, reverse=True):
        if len(ordered) >= PRIMARY_SHORTLIST_MAX:
            break
        ordered.append(doc_idx)

    candidates = [
        {
            "doc_idx": doc_idx,
            "algorithm_rank": position + 1,
            "primary_score": round(float(primary_scores.get(doc_idx, 0.0)), 2),
            "distinctive_direct_coverage": round(
                float((primary_details.get(doc_idx) or {}).get("distinctive_direct_coverage", 0.0)), 4
            ),
            "core_coverage": round(
                float((primary_details.get(doc_idx) or {}).get("core_coverage", 0.0)), 4
            ),
            "critical_gap_weight": round(
                float((primary_details.get(doc_idx) or {}).get("critical_gap_weight", 0.0)), 4
            ),
            "core_disclosure_labels": core_labels_by_doc.get(doc_idx, []),
            "forced_include": doc_idx in forced,
        }
        for position, doc_idx in enumerate(ordered)
    ]

    return {
        "root_claim": root.claim_number,
        "novelty_selected_document": novelty_idx,
        "eligible_indices": eligible,
        "algorithm_top1": ordered[0] if ordered else None,
        "candidates": candidates,
        "needs_adjudication": novelty_idx is None and len(candidates) >= PRIMARY_SHORTLIST_MIN,
    }


def shortlist_secondary_candidates(
    root: ParsedClaim,
    caches: Dict[int, Optional[Dict]],
    num_docs: int,
    primary_idx: int,
    weights: Optional[Dict[tuple[str, str], float]] = None,
) -> Dict:
    """확정된 주인용의 공백을 메우는 보조 후보를 추린다(Gate 2). LLM 호출 없음.

    반대 교시·기본 원리 변경·비양립이 명시된 문헌은 후보에서 제외하고 제외
    사유를 남긴다. 순위는 기존 정책과 동일하게 (핵심 공백 보완량, 보조 점수)다.
    """
    root_key = str(root.claim_number)
    root_keys = {root_key}
    primary_cache = caches.get(primary_idx)
    hard_gaps = _compute_primary_gaps(primary_cache, root_keys)
    soft_gaps = _compute_soft_gaps(primary_cache, root_keys)
    targets = hard_gaps | soft_gaps
    element_by_label = {normalize_label(e.label): e for e in root.elements}
    intrinsic_technical_gaps = {
        (root_key, label)
        for label in {normalize_label(e.label) for e in root.elements}
        if (root_key, label) in hard_gaps
        and label in element_by_label
        and _is_distinctive_technical_element(element_by_label[label])
    }

    candidates: List[Dict] = []
    rejected: Dict[str, List[str]] = {}
    for doc_idx in range(num_docs):
        if doc_idx == primary_idx:
            continue
        risks = _explicit_combination_risks(caches.get(doc_idx), targets)
        if risks:
            rejected[str(doc_idx)] = risks
            continue
        sub_score, sub_detail = _score_secondary_candidate(
            caches.get(doc_idx), primary_cache, [root], hard_gaps, soft_gaps, weights
        )
        if sub_score <= 0 or sub_detail.get("secondary_reason") not in {"hard", "soft", "support"}:
            continue
        filled: List[Dict] = []
        for claim_key, label in sorted(targets):
            item = _cache_item(caches.get(doc_idx), claim_key, label)
            if not item or _item_similarity(item) < 0.35:
                continue
            element = element_by_label.get(label)
            filled.append({
                "label": label,
                "claim_text": (element.text if element else ""),
                "judgment": item.get("judgment", "대응 없음"),
                "quote": item.get("quote", ""),
                "chunk_id": item.get("chunk_id", ""),
                "motivation_quote": item.get("motivation_quote", ""),
                "gap_type": "hard" if (claim_key, label) in hard_gaps else "soft",
            })
        candidates.append({
            "doc_idx": doc_idx,
            "sub_score": sub_score,
            "critical_gap_evidence_gain": _critical_gap_evidence_gain(
                caches.get(doc_idx), primary_cache, intrinsic_technical_gaps, weights
            ),
            "secondary_reason": sub_detail.get("secondary_reason"),
            "filled": filled,
            "detail": sub_detail,
        })

    candidates.sort(
        key=lambda item: (item["critical_gap_evidence_gain"], item["sub_score"]), reverse=True
    )
    candidates = candidates[:SECONDARY_SHORTLIST_MAX]
    for position, candidate in enumerate(candidates):
        candidate["algorithm_rank"] = position + 1

    return {
        "primary_idx": primary_idx,
        "hard_gaps": sorted(f"{key}:{label}" for key, label in hard_gaps),
        "soft_gaps": sorted(f"{key}:{label}" for key, label in soft_gaps),
        "candidates": candidates,
        "rejected_for_explicit_risk": rejected,
        "algorithm_top1": candidates[0]["doc_idx"] if candidates else None,
        "needs_adjudication": len(candidates) >= 1,
    }


def _select_family_reference_pair(
    family_claims: List[ParsedClaim],
    caches: Dict[int, Optional[Dict]],
    num_docs: int,
    adjudication: Optional[Dict] = None,
) -> Dict:
    """독립항의 차별적 핵심을 우선하면서 최적 보완 문헌쌍을 선택한다.

    adjudication이 주어지면 LLM 판정 결과를 우선 적용한다. 다만 후보 목록과
    커버리지 계산은 여전히 이 함수가 확정하며, LLM은 알고리즘이 만든 후보
    안에서만 선택할 수 있다(검증은 reference_adjudicator에서 수행).
    """
    context = _compute_family_context(family_claims, caches, num_docs)
    if context is None:
        return {}
    root = context["root"]
    root_key = context["root_key"]
    root_keys = {root_key}
    weights = context["weights"]
    dynamic_weights = context["dynamic_weights"]
    primary_scores = context["primary_scores"]
    primary_details = context["primary_details"]
    reuse_scores = context["reuse_scores"]
    novelty_screen = context["novelty_screen"]

    novelty_idx = novelty_screen.get("selected_document")
    if novelty_idx is not None:
        return {
            "root_claim": root.claim_number,
            "claim_numbers": [claim.claim_number for claim in family_claims],
            "primary_idx": novelty_idx,
            "secondary_idx": None,
            "secondary_reason": None,
            "secondary_detail": {},
            "combination_validity": {
                "coverage_complete": True,
                "critical_uncovered_labels": [],
                "remaining_uncovered_labels": [],
                "motivation_and_compatibility_status": "not_applicable_single_document",
                "rule": "단일 문헌이 모든 필수 구성을 직접 개시하므로 문헌 결합을 수행하지 않음",
            },
            "novelty_screen": novelty_screen,
            "analysis_track": "novelty_single_reference",
            "primary_scores": {str(key): value for key, value in primary_scores.items()},
            "primary_score_details": {str(key): value for key, value in primary_details.items()},
            "dependent_reuse_scores": {str(key): round(value, 4) for key, value in reuse_scores.items()},
            "near_primary_candidates": [novelty_idx],
            "eligible_primary_candidates": [novelty_idx],
            "pair_candidates": [],
            "llm_adjudication": {},
            "selection_method": "single_document_novelty_gate_v1",
        }

    best_primary_score = max(primary_scores.values(), default=0.0)
    best_distinctive_direct = max(
        (
            float(detail.get("distinctive_direct_coverage", 0.0))
            for detail in primary_details.values()
        ),
        default=0.0,
    )
    # 주인용 자격 게이트. 순서쌍 전수 탐색의 장점(결합 가능성까지 본 출발점
    # 선택)은 유지하되, 차별적 핵심 직접 개시량이 최고 문헌에 크게 못 미치는
    # 문헌은 보조문헌이 아무리 좋아도 주인용이 될 수 없게 한다. 공개일은 여전히
    # 법적 적격성 판단이나 후보 제외에 사용하지 않는다.
    eligible_pool = _eligible_primary_indices(
        primary_details, num_docs, _core_disclosure_indices(caches, root, num_docs)
    )
    near_primary_pool = eligible_pool

    # LLM 판정 결과는 reference_adjudicator에서 이미 검증(후보 소속·인덱스
    # 범위·근거 chunk 실재)을 마친 값이다. 여기서는 자격 게이트를 통과한
    # 문헌인지만 한 번 더 확인하고 적용한다.
    adjudicated_primary: Optional[int] = None
    adjudicated_secondary: Optional[int] = None
    adjudicated_no_secondary = False
    if adjudication:
        proposed_primary = adjudication.get("primary_idx")
        if isinstance(proposed_primary, int) and proposed_primary in eligible_pool:
            adjudicated_primary = proposed_primary
            near_primary_pool = [proposed_primary]
            proposed_secondary = adjudication.get("secondary_idx")
            if (
                isinstance(proposed_secondary, int)
                and 0 <= proposed_secondary < num_docs
                and proposed_secondary != proposed_primary
            ):
                adjudicated_secondary = proposed_secondary
            elif adjudication.get("secondary_explicitly_none"):
                # 기술 검토가 어느 후보도 결합 불가로 판정한 경우다. 알고리즘이
                # 보조문헌을 도로 끼워 넣으면 검토 결과와 어긋난 보고서가 된다.
                adjudicated_no_secondary = True
        else:
            logger.warning(
                "LLM 주인용 판정 doc[%s]이 자격 게이트를 통과하지 못해 무시합니다.",
                proposed_primary,
            )

    pair_candidates: List[Dict] = []
    for primary_idx in near_primary_pool:
        primary_cache = caches.get(primary_idx)
        hard_gaps = _compute_primary_gaps(primary_cache, root_keys)
        soft_gaps = _compute_soft_gaps(primary_cache, root_keys)
        targets = hard_gaps | soft_gaps
        intrinsic_technical_gaps = {
            (root_key, normalize_label(element.label))
            for element in root.elements
            if (root_key, normalize_label(element.label)) in hard_gaps
            and _is_distinctive_technical_element(element)
        }
        best_secondary: Optional[int] = None
        best_secondary_score = 0.0
        best_secondary_rank: tuple[float, float] = (-1.0, -1.0)
        best_secondary_detail: Dict = {}
        rejected_pair_risks: Dict[str, List[str]] = {}

        for secondary_idx in range(num_docs):
            if secondary_idx == primary_idx:
                continue
            if adjudicated_no_secondary and primary_idx == adjudicated_primary:
                break
            sub_score, sub_detail = _score_secondary_candidate(
                caches.get(secondary_idx),
                primary_cache,
                [root],
                hard_gaps,
                soft_gaps,
                weights,
            )
            reason = sub_detail.get("secondary_reason")
            risks = _explicit_combination_risks(caches.get(secondary_idx), targets)
            if risks:
                rejected_pair_risks[str(secondary_idx)] = risks
                continue
            if sub_score <= 0 or reason not in {"hard", "soft", "support"}:
                continue
            reuse_tiebreak = min(
                DEPENDENT_REUSE_TIEBREAK_MAX,
                reuse_scores.get(secondary_idx, 0.0) * DEPENDENT_REUSE_TIEBREAK_MAX,
            )
            adjusted_sub_score = sub_score + reuse_tiebreak
            critical_gap_gain = _critical_gap_evidence_gain(
                caches.get(secondary_idx),
                primary_cache,
                intrinsic_technical_gaps,
                weights,
            )
            # If the primary document still has a core gap, a quotation that
            # improves that gap outranks a stronger match limited to generic
            # interfaces.  The ordinary SubScore remains the tie-breaker.
            candidate_rank = (critical_gap_gain, adjusted_sub_score)
            # LLM-B가 이 주인용에 대한 보조문헌을 확정했으면 그 문헌만 채택한다.
            # 후보 목록 자체는 위 필터가 확정하므로, LLM은 알고리즘이 통과시킨
            # 문헌 중에서만 고를 수 있다.
            if adjudicated_secondary is not None and primary_idx == adjudicated_primary:
                if secondary_idx != adjudicated_secondary:
                    continue
                candidate_rank = (float("inf"), float("inf"))
            if candidate_rank > best_secondary_rank:
                best_secondary = secondary_idx
                best_secondary_score = adjusted_sub_score
                best_secondary_rank = candidate_rank
                best_secondary_detail = dict(sub_detail)
                best_secondary_detail["raw_sub_score"] = sub_score
                best_secondary_detail["dependent_reuse_tiebreak"] = round(reuse_tiebreak, 2)
                best_secondary_detail["critical_gap_evidence_gain"] = critical_gap_gain

        secondary_cache = caches.get(best_secondary) if best_secondary is not None else None
        similarity = _claim_similarity(primary_cache, secondary_cache, root)
        uncovered = set(similarity.get("uncovered_labels") or [])
        residual = set(similarity.get("residual_labels") or uncovered)
        distinctive_weight_by_label = {
            label: weight
            for (claim_key, label), weight in dynamic_weights.items()
            if claim_key == root_key
        }
        critical_uncovered = {
            label for label in uncovered
            if distinctive_weight_by_label.get(normalize_label(label), 3) >= _CORE_IMPORTANCE_THRESHOLD
        }
        critical_residual = {
            label for label in residual
            if distinctive_weight_by_label.get(normalize_label(label), 3) >= _CORE_IMPORTANCE_THRESHOLD
        }
        # 미커버 "개수"만 세면 사소한 구성 3개를 덮은 쌍이 핵심 구성 1개를 덮은
        # 쌍을 이긴다. 1순위 키는 이미 핵심 가중치로 걸러지므로, 2순위 키는
        # 가중합으로 두어 같은 왜곡이 아래 단계에서 반복되지 않게 한다.
        uncovered_weight = sum(
            distinctive_weight_by_label.get(normalize_label(label), 3.0)
            for label in uncovered
        )
        primary_reuse_tiebreak = min(
            DEPENDENT_REUSE_TIEBREAK_MAX,
            reuse_scores.get(primary_idx, 0.0) * DEPENDENT_REUSE_TIEBREAK_MAX,
        )
        primary_detail = primary_details.get(primary_idx, {})
        has_substantive_secondary = 1 if (
            best_secondary is not None
            and (
                best_secondary_detail.get("hard_fill_count", 0) > 0
                or best_secondary_detail.get("secondary_reason") == "hard"
            )
        ) else 0
        pair_rank = (
            -len(critical_uncovered),
            -uncovered_weight,
            (
                float(primary_detail.get("field_alignment_coverage", 0.0))
                if best_distinctive_direct < 0.35
                else 0.0
            ),
            float(similarity.get("combined_similarity", 0.0)),
            float(primary_detail.get("distinctive_direct_coverage", 0.0)),
            float(primary_detail.get("core_coverage", 0.0)),
            has_substantive_secondary,
            float(primary_detail.get("distinctive_strong_breadth", 0.0)),
            float(primary_detail.get("element_coverage", 0.0)),
            # 핵심 지표가 비슷한 경우에만 전체 기술 흐름·직접 개시 폭으로
            # 동률을 해소한다. 이 지표가 명확한 핵심 직접개시를 앞설 수는 없다.
            float(primary_detail.get("direct_disclosure_breadth", 0.0)),
            primary_scores.get(primary_idx, 0.0) + primary_reuse_tiebreak,
            best_secondary_score,
            -primary_idx,
        )
        secondary_reason = best_secondary_detail.get("secondary_reason") if best_secondary is not None else None
        combination_validity = {
            "coverage_complete": not residual,
            "critical_uncovered_labels": sorted(critical_residual),
            "remaining_uncovered_labels": sorted(residual),
            "explicit_contrary_teaching": bool(rejected_pair_risks),
            "rejected_pair_risks": rejected_pair_risks,
            "motivation_and_compatibility_status": (
                "report_substantive_review_required"
                if best_secondary is not None
                else "explicit_risk_blocks_available_pair"
                if rejected_pair_risks
                else "not_applicable"
            ),
            "rule": "구성 보완과 결합 동기·기술적 양립성은 별도 판단",
        }
        pair_candidates.append({
            "primary_idx": primary_idx,
            "secondary_idx": best_secondary,
            "primary_score": primary_scores.get(primary_idx, 0.0),
            "primary_reuse_score": round(reuse_scores.get(primary_idx, 0.0), 4),
            "primary_reuse_tiebreak": round(primary_reuse_tiebreak, 2),
            "secondary_score": round(best_secondary_score, 2),
            "secondary_detail": best_secondary_detail,
            "secondary_reason": secondary_reason,
            "hard_gaps": sorted([f"{claim_key}:{label}" for claim_key, label in hard_gaps]),
            "soft_gaps": sorted([f"{claim_key}:{label}" for claim_key, label in soft_gaps]),
            "similarity": similarity,
            "combination_validity": combination_validity,
            "rank": pair_rank,
        })

    selected = max(pair_candidates, key=lambda item: item["rank"]) if pair_candidates else {}
    for item in pair_candidates:
        item.pop("rank", None)
    if selected:
        selected = next(
            item for item in pair_candidates
            if item["primary_idx"] == selected["primary_idx"]
            and item["secondary_idx"] == selected["secondary_idx"]
        )
    return {
        "root_claim": root.claim_number,
        "claim_numbers": [claim.claim_number for claim in family_claims],
        "primary_idx": selected.get("primary_idx", 0),
        "secondary_idx": selected.get("secondary_idx"),
        "secondary_reason": selected.get("secondary_reason"),
        "secondary_detail": selected.get("secondary_detail", {}),
        "combination_validity": selected.get("combination_validity", {}),
        "primary_scores": {str(key): value for key, value in primary_scores.items()},
        "primary_score_details": {str(key): value for key, value in primary_details.items()},
        "dependent_reuse_scores": {str(key): round(value, 4) for key, value in reuse_scores.items()},
        "near_primary_candidates": near_primary_pool,
        "eligible_primary_candidates": eligible_pool,
        "pair_candidates": pair_candidates,
        "novelty_screen": novelty_screen,
        "llm_adjudication": adjudication or {},
        "analysis_track": "inventive_step_combination",
        "selection_method": (
            "novelty_gate_then_eligibility_gated_pair_v3_llm_adjudicated"
            if adjudicated_primary is not None
            else "novelty_gate_then_eligibility_gated_pair_v3"
        ),
    }


# ---------------------------------------------------------------------------
# 메인: 비교 캐시 기반 체인 빌드 (v3 — 보완성 기반 secondary 선정)
# ---------------------------------------------------------------------------

def build_citation_chain_from_comparisons(
    job_dir: str,
    claims: List[ParsedClaim],
    prior_docs: List[ExtractedDocument],
    adjudications: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """
    comparisons_{doc_idx}.json 에서 판정 점수를 집계하여 주인용발명을 선정하고,
    보완성(complementarity) 기준으로 보조인용발명을 선정한다.

    선정 원칙:
    1. 주인용발명: 청구항의 차별적 핵심 구조·작동관계·효과를 직접 개시하는 문헌
       - 여러 문헌에 반복되는 범용 입출력·처리 골격은 사건별 빈도로 감쇠
       - 소수 문헌에만 직접 나타나는 핵심 구성의 대응 강도와 원문 근거를 주점수로 반영
    2. 보조인용발명: 주인용발명이 커버하지 못하거나 약하게 커버한 중요 구성요소를 가장 잘 채우는 문헌
       - 보조인용발명이 공백 구성요소를 채워 결합 시 100% 커버 가능하면 채택
       - 공백을 전혀 못 채우면 Template A (단독 인용) 사용
    3. 종속항: 부모 청구항의 인용발명을 상속 + 추가 공백 채우는 다음 인용발명 추가
    """
    previous_chain: Dict = {}
    previous_chain_path = Path(job_dir) / "citation_chain.json"
    if previous_chain_path.exists():
        try:
            loaded_previous = json.loads(previous_chain_path.read_text(encoding="utf-8"))
            if isinstance(loaded_previous, dict):
                previous_chain = loaded_previous
        except (OSError, json.JSONDecodeError):
            logger.warning("기존 citation_chain.json을 읽지 못해 잠금 없이 재계산합니다.")
    # 정책 버전이 달라지면 과거 점수·예외 규칙으로 잠긴 문헌 체인을 승계하지 않는다.
    # 같은 정책 버전에서 이미 발행된 보고서만 번호와 체인을 안정적으로 유지한다.
    selection_locks = (
        dict(previous_chain.get("selection_locks") or {})
        if previous_chain.get("policy_version") == CITATION_CHAIN_POLICY_VERSION
        else {}
    )

    num_docs = len(prior_docs)
    if num_docs == 0:
        return {}

    independent_claims = [c for c in claims if c.claim_type == "independent"]
    indep_keys = {str(c.claim_number) for c in independent_claims}

    # 비교 캐시는 문서당 1회만 디스크에서 읽고 이후 단계(점수/공백/보완성)에서 재사용한다.
    caches: Dict[int, Optional[Dict]] = {i: _load_cache(job_dir, i) for i in range(num_docs)}
    element_weights = _element_weight_map(independent_claims)

    # ── 1단계: 인용발명별 주인용 적합도 점수 집계 ─────────────────────────
    inv_scores: Dict[int, float] = {i: 0.0 for i in range(num_docs)}
    inv_match_counts: Dict[int, int] = {i: 0 for i in range(num_docs)}
    primary_score_details: Dict[int, Dict] = {i: {} for i in range(num_docs)}

    for doc_idx in range(num_docs):
        cache = caches[doc_idx]
        if not cache:
            logger.warning(f"comparisons_{doc_idx}.json 없음 — 점수 0 처리")
            continue

        score, match_count, detail = _score_prior_cache(cache, independent_claims)
        inv_scores[doc_idx] = score
        inv_match_counts[doc_idx] = match_count
        primary_score_details[doc_idx] = detail

    # ── 2단계: 전역 fallback 주인용발명 선정 ───────────────────────────────
    # 실제 독립항 보고서는 아래 패밀리별 차별적 핵심 우선 선정을 사용한다.
    if all(s == 0 for s in inv_scores.values()):
        logger.warning("모든 인용발명 점수 0 — 인용발명 1을 주인용발명으로 기본 설정")
        primary_inv_idx = 0
    else:
        primary_inv_idx = max(inv_scores, key=lambda k: inv_scores[k])

    primary_score = inv_scores[primary_inv_idx]
    primary_candidates = [
        {
            "doc_idx": i,
            "score": inv_scores[i],
            "match_count": inv_match_counts[i],
            "detail": primary_score_details.get(i, {}),
        }
        for i in sorted(inv_scores, key=lambda k: inv_scores[k], reverse=True)[:3]
    ]
    logger.info(f"주인용발명 선정: doc[{primary_inv_idx}] {prior_docs[primary_inv_idx].filename} "
                f"(총점 {primary_score})")

    # ── 3단계: 보조인용발명 선정 (보완성 기준) ──────────────────────────────
    primary_gaps = _compute_primary_gaps(caches[primary_inv_idx], indep_keys)
    gap_count = len(primary_gaps)

    secondary_inv_idx = None
    secondary_reason = None  # "hard": 공백 보완 / "soft": 약점(일부 차이) 문헌 보강
    complementarity_scores: Dict[int, int] = {}
    secondary_candidate_details: Dict[int, Dict] = {}

    if primary_gaps:
        for i in range(num_docs):
            if i == primary_inv_idx:
                continue
            comp_score = _compute_complementarity_score(caches[i], primary_gaps, element_weights)
            complementarity_scores[i] = comp_score

        if complementarity_scores:
            best_comp_idx = max(complementarity_scores, key=lambda k: complementarity_scores[k])
            best_comp_score = complementarity_scores[best_comp_idx]

            if best_comp_score > 0:
                secondary_inv_idx = best_comp_idx
                secondary_reason = "hard"
                covers_all = _is_full_coverage(caches[primary_inv_idx], caches[secondary_inv_idx], indep_keys)
                logger.info(
                    f"보조인용발명 선정: doc[{secondary_inv_idx}] "
                    f"{prior_docs[secondary_inv_idx].filename} "
                    f"(보완점수 {best_comp_score}, "
                    f"공백 {gap_count}개, "
                    f"결합 100% 커버={'예' if covers_all else '부분'})"
                )
            else:
                logger.info(f"보조인용발명 없음: 어떤 문헌도 주인용발명 공백({gap_count}개)을 보완 못함 → Template A")
    else:
        logger.info(f"주인용발명이 모든 구성요소 커버 (공백 없음)")

    # ── 3.5단계: 소프트 공백(일부 차이) 문헌 보강 ──────────────────────────
    # 하드 공백 기준으로 보조가 선정되지 않았어도, '일부 차이' 약점을
    # '실질적 동일' 이상으로 개시하는 문헌이 있으면 채택한다.
    # → 차이점 극복 논리를 주지관용 대신 문헌 근거로 작성할 수 있게 한다.
    soft_gaps = _compute_soft_gaps(caches[primary_inv_idx], indep_keys)
    soft_gap_count = len(soft_gaps)

    if secondary_inv_idx is None and soft_gaps:
        soft_scores: Dict[int, int] = {}
        for i in range(num_docs):
            if i == primary_inv_idx:
                continue
            soft_scores[i] = _compute_soft_improvement_score(caches[i], soft_gaps, element_weights)

        if soft_scores:
            best_soft_idx = max(soft_scores, key=lambda k: soft_scores[k])
            best_soft_score = soft_scores[best_soft_idx]
            if best_soft_score > 0:
                secondary_inv_idx = best_soft_idx
                secondary_reason = "soft"
                complementarity_scores[best_soft_idx] = best_soft_score
                logger.info(
                    f"보조인용발명 선정(소프트): doc[{secondary_inv_idx}] "
                    f"{prior_docs[secondary_inv_idx].filename} "
                    f"(약점 {soft_gap_count}개 중 '실질적 동일' 이상 보강점수 {best_soft_score})"
                )
            else:
                logger.info(f"약점({soft_gap_count}개) 보강 문헌 없음 → Template A (주지관용 논거 사용)")

    # ── 3.6단계: 불완전하지만 명시적인 보조 문헌 근거 ─────────────────────
    # hard/soft 기준을 충족하지 못해도, 주인용발명의 공백·약점에 대해 보조 문헌이
    # '일부 유사' 이상의 발췌 근거를 갖고 있으면 Template B로 올려 결합 논리에서
    # 먼저 검토한다. 단, 이는 완전 보완 판정이 아니므로 Phase 1/2에는 잔여 차이를
    # 그대로 남기고, 주지관용은 문헌 근거로 설명되지 않는 잔여 차이에만 사용한다.
    if secondary_inv_idx is None:
        support_targets = primary_gaps | soft_gaps
        if support_targets:
            support_scores: Dict[int, int] = {}
            for i in range(num_docs):
                if i == primary_inv_idx:
                    continue
                support_scores[i] = _compute_supporting_evidence_score(
                    caches[i], support_targets, element_weights
                )

            if support_scores:
                best_support_idx = max(support_scores, key=lambda k: support_scores[k])
                best_support_score = support_scores[best_support_idx]
                if best_support_score > 0:
                    secondary_inv_idx = best_support_idx
                    secondary_reason = "support"
                    complementarity_scores[best_support_idx] = best_support_score
                    logger.info(
                        f"보조인용발명 선정(문헌 근거): doc[{secondary_inv_idx}] "
                        f"{prior_docs[secondary_inv_idx].filename} "
                        f"(공백/약점 {len(support_targets)}개 중 불완전 명시근거 점수 {best_support_score})"
                    )

    # Policy-level SubScore reranking:
    # 0.55 gap/weakness fill + 0.20 quote explicitness
    # + 0.10 field/problem relatedness + 0.10 applicability/predictability
    # + 0.05 reportability. This intentionally does not over-filter candidates
    # by "motivation to combine"; it flags weak combinations for the report.
    subscore_targets = primary_gaps | soft_gaps
    secondary_candidate_scores: Dict[int, float] = {}
    if subscore_targets:
        for i in range(num_docs):
            if i == primary_inv_idx:
                continue
            sub_score, sub_detail = _score_secondary_candidate(
                caches[i],
                caches[primary_inv_idx],
                independent_claims,
                primary_gaps,
                soft_gaps,
                element_weights,
            )
            secondary_candidate_scores[i] = sub_score
            secondary_candidate_details[i] = sub_detail

        viable_scores = {
            i: score
            for i, score in secondary_candidate_scores.items()
            if score > 0
            and secondary_candidate_details.get(i, {}).get("secondary_reason")
            in {"hard", "soft", "support"}
        }
        if viable_scores:
            best_sub_idx = max(viable_scores, key=lambda k: viable_scores[k])
            best_detail = secondary_candidate_details.get(best_sub_idx, {})
            secondary_inv_idx = best_sub_idx
            secondary_reason = best_detail.get("secondary_reason") or secondary_reason
            complementarity_scores[best_sub_idx] = viable_scores[best_sub_idx]
            logger.info(
                f"보조인용발명 SubScore 선정: doc[{secondary_inv_idx}] "
                f"{prior_docs[secondary_inv_idx].filename} "
                f"(SubScore {viable_scores[best_sub_idx]}, reason={secondary_reason})"
            )

    if secondary_inv_idx is None and not soft_gaps and not primary_gaps:
        logger.info("주인용발명이 공백·약점 모두 없음 → Template A")

    # ── 4단계: doc_name_mapping 생성 ────────────────────────────────────────
    # 순서: 주인용발명=1, 보조인용발명=2, 나머지는 점수 내림차순
    ordered = [primary_inv_idx]
    if secondary_inv_idx is not None:
        ordered.append(secondary_inv_idx)
    remaining = sorted(
        [i for i in range(num_docs) if i not in ordered],
        key=lambda k: inv_scores[k],
        reverse=True,
    )
    ordered += remaining
    doc_name_mapping = {str(doc_idx): f"인용발명 {rank + 1}"
                        for rank, doc_idx in enumerate(ordered)}

    # ── 4.5단계: 독립항별 신뢰도(가중 유사도) 계산 — 캐시 재사용, LLM 없음 ──
    # 소프트 보강으로 채택된 보조인용발명은 "단독 충분" 청구항에서 제외한다:
    # 주인용발명 단독 가중 유사도 ≥ SINGLE_SUFFICIENT_SIMILARITY 이면 결합 불필요.
    # (하드 공백 보완은 면제 대상 아님 — 미개시 구성요소는 문헌 근거가 필요하다.)
    secondary_cache = caches[secondary_inv_idx] if secondary_inv_idx is not None else None
    confidence: Dict[str, Dict] = {}
    single_sufficient_claims: set = set()
    for claim in independent_claims:
        key = str(claim.claim_number)
        p_conf = _claim_similarity(caches[primary_inv_idx], None, claim)
        if (secondary_reason in {"soft", "support"}
                and not primary_gaps
                and p_conf["primary_similarity"] >= SINGLE_SUFFICIENT_SIMILARITY):
            single_sufficient_claims.add(key)
            confidence[key] = p_conf
            logger.info(
                f"청구항 {claim.claim_number}: 주인용 단독 {p_conf['primary_similarity']}% "
                f"≥ {SINGLE_SUFFICIENT_SIMILARITY}% → 단독 충분, 소프트 보강 제외 (Template A)"
            )
            continue
        conf = _claim_similarity(caches[primary_inv_idx], secondary_cache, claim)
        confidence[key] = conf
        logger.info(
            f"청구항 {claim.claim_number} 신뢰도: 주인용 {conf['primary_similarity']}%, "
            f"결합 {conf['combined_similarity']}%, 미커버 {conf['uncovered_labels'] or '없음'}"
        )

    # ── 5단계: 독립항 패밀리별 주문헌·보조문헌 쌍 및 청구항 체인 구성 ─────
    # 여러 독립항을 하나의 전역 주문헌으로 강제하지 않는다. 각 독립항을 루트로
    # 독립적인 문헌쌍을 선정하고, 그 종속항만 해당 체인을 상속한다.
    family_groups, orphan_dependents = _claim_family_groups(claims)
    family_selections: Dict[str, Dict] = {}
    chains: Dict[str, Dict] = {}
    family_single_sufficient: set[str] = set()

    for root_number, family_claims in sorted(family_groups.items()):
        selection = _select_family_reference_pair(
            family_claims,
            caches,
            num_docs,
            adjudication=(adjudications or {}).get(str(root_number)),
        )
        if not selection:
            continue
        lock_data = selection_locks.get(str(root_number))
        locked_total_raw = lock_data.get("total", []) if isinstance(lock_data, dict) else lock_data or []
        locked_total = []
        for value in locked_total_raw:
            try:
                doc_idx = int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= doc_idx < num_docs and doc_idx not in locked_total:
                locked_total.append(doc_idx)
        selection_locked = bool(locked_total)
        if selection_locked:
            proposed = {
                "primary_idx": selection.get("primary_idx"),
                "secondary_idx": selection.get("secondary_idx"),
                "selection_method": selection.get("selection_method"),
            }
            family_primary = locked_total[0]
            family_secondary = locked_total[1] if len(locked_total) > 1 else None
            root_claim = next(claim for claim in family_claims if claim.claim_number == root_number)
            locked_supporting = [caches.get(doc_idx) for doc_idx in locked_total[1:]]
            locked_similarity = _claim_similarity(
                caches.get(family_primary),
                locked_supporting[0] if locked_supporting else None,
                root_claim,
                additional_caches=locked_supporting[1:],
            )
            locked_residual = locked_similarity.get("residual_labels", [])
            locked_critical = [
                element.label
                for element in root_claim.elements
                if element.label in locked_residual
                and _importance_value(element.importance) >= _CORE_IMPORTANCE_THRESHOLD
            ]
            selection.update({
                "primary_idx": family_primary,
                "secondary_idx": family_secondary,
                "selection_locked": True,
                "lock_reason": (
                    lock_data.get("reason", "independent_claim_report_issued")
                    if isinstance(lock_data, dict) else "independent_claim_report_issued"
                ),
                "proposed_recalculation": proposed,
                "combination_validity": {
                    "coverage_complete": not locked_residual,
                    "critical_uncovered_labels": locked_critical,
                    "remaining_uncovered_labels": locked_residual,
                    "motivation_and_compatibility_status": "locked_report_chain",
                    "rule": "확정된 독립항 문헌 체인과 번호를 종속항에서 변경하지 않음",
                },
            })
            previous_root_chain = dict((previous_chain.get("chains") or {}).get(str(root_number)) or {})
            previous_root_chain.update({
                "total": locked_total,
                "inherited": [],
                "added": locked_total,
                "parent": None,
                "family_root": root_number,
                "family_primary_idx": family_primary,
                "family_secondary_idx": family_secondary,
                "selection_locked": True,
                "lock_reason": selection.get("lock_reason"),
                "combination_validity": selection.get("combination_validity", {}),
            })
            if not previous_root_chain.get("reference_roles"):
                previous_root_chain["reference_roles"] = {
                    str(doc_idx): "primary" if position == 0 else "substantive_secondary"
                    for position, doc_idx in enumerate(locked_total)
                }
            chains[str(root_number)] = previous_root_chain
        family_selections[str(root_number)] = selection
        family_primary = int(selection.get("primary_idx", primary_inv_idx))
        family_secondary = selection.get("secondary_idx")
        root_claim = next(claim for claim in family_claims if claim.claim_number == root_number)
        primary_only = _claim_similarity(caches.get(family_primary), None, root_claim)
        selected_pair = next(
            (
                candidate for candidate in selection.get("pair_candidates", [])
                if candidate.get("primary_idx") == family_primary
                and candidate.get("secondary_idx") == family_secondary
            ),
            {},
        )
        if (
            family_secondary is not None
            and selection.get("secondary_reason") in {"soft", "support"}
            and not selected_pair.get("hard_gaps")
            and primary_only.get("primary_similarity", 0) >= SINGLE_SUFFICIENT_SIMILARITY
        ):
            family_single_sufficient.add(str(root_number))

        _build_chains_recursive(
            claims=family_claims,
            primary_inv_idx=family_primary,
            secondary_inv_idx=family_secondary,
            inv_scores=inv_scores,
            num_docs=num_docs,
            chains=chains,
            single_sufficient_claims=family_single_sufficient,
            caches=caches,
        )
        root_chain = chains.get(str(root_number), {})
        root_chain["family_root"] = root_number
        root_chain["family_primary_idx"] = family_primary
        root_chain["family_secondary_idx"] = family_secondary
        root_chain["combination_validity"] = selection.get("combination_validity", {})
        root_chain["analysis_track"] = selection.get("analysis_track", "inventive_step_combination")
        root_chain["selection_method"] = selection.get(
            "selection_method", "novelty_gate_then_distinctive_core_gap_pair_v2"
        )
        root_chain["novelty_screen"] = selection.get("novelty_screen", {})
        secondary_detail = selection.get("secondary_detail", {})
        if selection_locked and root_chain.get("combination_rationale"):
            family_rationale = root_chain["combination_rationale"]
        elif family_secondary is None and selection.get("combination_validity", {}).get("remaining_uncovered_labels"):
            family_rationale = _combination_rationale_for("insufficient", score_detail=secondary_detail)
        else:
            family_rationale = _combination_rationale_for(
                selection.get("secondary_reason"),
                candidate_types=secondary_detail.get("candidate_rationale_types"),
                warnings=secondary_detail.get("warnings"),
                score_detail=secondary_detail,
            )
        root_chain["combination_rationale"] = family_rationale
        root_chain["combination_rationale_type"] = family_rationale["type"]

    # 부모항이 없는 종속항은 기존의 보수적 fallback을 사용한다. 이미 생성된 패밀리
    # 루트/자식은 건너뛰므로 정상 패밀리의 문헌쌍에는 영향을 주지 않는다.
    if orphan_dependents or not family_groups:
        _build_chains_recursive(
            claims=claims,
            primary_inv_idx=primary_inv_idx,
            secondary_inv_idx=secondary_inv_idx,
            inv_scores=inv_scores,
            num_docs=num_docs,
            chains=chains,
            single_sufficient_claims=single_sufficient_claims,
            caches=caches,
        )

    conventional_doc_order = _apply_conventional_support_policy(
        chains,
        claims,
        caches,
        num_docs,
        element_weights,
    )

    # The independent-claim exception policy above can add or remove a limited
    # conventional-support reference. Rebuild dependent entries afterwards so
    # every child inherits the parent's final, report-visible chain exactly.
    for claim in claims:
        if claim.claim_type == "dependent":
            chains.pop(str(claim.claim_number), None)
    _build_chains_recursive(
        claims=claims,
        primary_inv_idx=primary_inv_idx,
        secondary_inv_idx=secondary_inv_idx,
        inv_scores=inv_scores,
        num_docs=num_docs,
        chains=chains,
        single_sufficient_claims=single_sufficient_claims,
        caches=caches,
    )

    # Rebuild display order after all per-claim policies.  References first
    # adopted by dependent claims follow the inherited independent references
    # in claim order, so a chain such as claim 1 -> claim 2 -> claim 3 is shown
    # naturally as references 1,2 -> 1,2,3 -> 1,2,3,4.
    ordered: List[int] = []
    ordered_claims = sorted(
        claims,
        key=lambda item: (0 if item.claim_type == "independent" else 1, item.claim_number),
    )
    for claim in ordered_claims:
        for doc_idx in (chains.get(str(claim.claim_number), {}).get("total") or []):
            if doc_idx not in ordered:
                ordered.append(doc_idx)
    if not ordered and num_docs:
        ordered.append(primary_inv_idx)
    for doc_idx in conventional_doc_order:
        if doc_idx not in ordered:
            ordered.append(doc_idx)
    remaining = sorted(
        [i for i in range(num_docs) if i not in ordered],
        key=lambda k: inv_scores[k],
        reverse=True,
    )
    ordered += remaining
    locked_mapping = dict(previous_chain.get("doc_name_mapping") or {}) if selection_locks else {}
    if selection_locks and not locked_mapping:
        for lock_value in selection_locks.values():
            if isinstance(lock_value, dict) and lock_value.get("doc_name_mapping"):
                locked_mapping = dict(lock_value["doc_name_mapping"])
                break
    doc_name_mapping: Dict[str, str] = {}
    used_numbers: set[int] = set()
    for raw_idx, name in locked_mapping.items():
        try:
            doc_idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        match = re.fullmatch(r"인용발명\s+(\d+)", str(name).strip())
        if not (0 <= doc_idx < num_docs) or not match:
            continue
        number = int(match.group(1))
        if number in used_numbers:
            continue
        doc_name_mapping[str(doc_idx)] = f"인용발명 {number}"
        used_numbers.add(number)
    next_number = max(used_numbers, default=0) + 1
    for doc_idx in ordered:
        if str(doc_idx) in doc_name_mapping:
            continue
        while next_number in used_numbers:
            next_number += 1
        doc_name_mapping[str(doc_idx)] = f"인용발명 {next_number}"
        used_numbers.add(next_number)
        next_number += 1
    ordered = sorted(
        range(num_docs),
        key=lambda doc_idx: int(doc_name_mapping[str(doc_idx)].split()[-1]),
    )

    # Confidence must follow the final per-claim chain, including an exceptional
    # third conventional-support reference or a removed weak secondary.
    for claim in independent_claims:
        key = str(claim.claim_number)
        total_refs = chains.get(key, {}).get("total", [primary_inv_idx])
        supporting = [caches.get(doc_idx) for doc_idx in total_refs[1:]]
        confidence[key] = _claim_similarity(
            caches.get(total_refs[0]),
            supporting[0] if supporting else None,
            claim,
            additional_caches=supporting[1:],
        )

    # 기존 API의 전역 필드는 첫 번째 독립항 패밀리를 대표값으로 유지하되, 실제
    # 보고서와 감사 데이터는 아래 `families` 및 청구항별 chain 값을 사용한다.
    first_family_key = sorted(family_selections, key=lambda value: int(value))[0] if family_selections else None
    representative_family = family_selections.get(first_family_key, {}) if first_family_key else {}
    if representative_family:
        primary_inv_idx = int(representative_family.get("primary_idx", primary_inv_idx))
        secondary_inv_idx = representative_family.get("secondary_idx")
        secondary_reason = representative_family.get("secondary_reason")
    selected_secondary_detail = (
        representative_family.get("secondary_detail", {})
        if representative_family
        else secondary_candidate_details.get(secondary_inv_idx, {})
        if secondary_inv_idx is not None else {}
    )
    representative_remaining = (
        representative_family.get("combination_validity", {}).get("remaining_uncovered_labels", [])
        if representative_family else []
    )
    if secondary_inv_idx is None and (representative_remaining or primary_gaps or soft_gaps):
        combination_rationale = _combination_rationale_for(
            "insufficient",
            score_detail=selected_secondary_detail,
        )
    else:
        combination_rationale = _combination_rationale_for(
            secondary_reason,
            candidate_types=selected_secondary_detail.get("candidate_rationale_types"),
            warnings=selected_secondary_detail.get("warnings"),
            score_detail=selected_secondary_detail,
        )
    secondary_candidates = [
        {
            "doc_idx": i,
            "score": secondary_candidate_scores.get(i, 0),
            "detail": secondary_candidate_details.get(i, {}),
        }
        for i in sorted(
            secondary_candidate_scores,
            key=lambda k: secondary_candidate_scores.get(k, 0),
            reverse=True,
        )[:3]
    ]
    representative_pair = next(
        (
            candidate for candidate in representative_family.get("pair_candidates", [])
            if candidate.get("primary_idx") == primary_inv_idx
            and candidate.get("secondary_idx") == secondary_inv_idx
        ),
        {},
    )
    if representative_pair:
        gap_count = len(representative_pair.get("hard_gaps", []))
        soft_gap_count = len(representative_pair.get("soft_gaps", []))

    result = {
        "policy_version": CITATION_CHAIN_POLICY_VERSION,
        "primary_inv_idx": primary_inv_idx,
        "primary_inv_name": doc_name_mapping[str(primary_inv_idx)],
        "scoring_method": "single_document_novelty_gate_then_distinctive_core_combination_v2",
        "score_semantics": {
            "report_similarity_bands_are_separate": True,
            "internal_label_anchors": _LABEL_PERCENT,
            "internal_ordinal_ranks": _JUDGMENT_SCORE,
            "atomic_limitation_adjustment": True,
            "average_score_never_overrides_missing_limitation": True,
        },
        "inv_scores": {str(k): v for k, v in inv_scores.items()},
        "inv_match_counts": {str(k): v for k, v in inv_match_counts.items()},
        "primary_score_details": {str(k): v for k, v in primary_score_details.items()},
        "primary_candidates": primary_candidates,
        "secondary_candidate_scores": {str(k): v for k, v in secondary_candidate_scores.items()},
        "secondary_candidate_details": {str(k): v for k, v in secondary_candidate_details.items()},
        "secondary_candidates": secondary_candidates,
        "families": family_selections,
        "claim_family_policy": {
            "single_document_novelty_first": True,
            "novelty_requires_direct_complete_disclosure": True,
            "no_cross_document_mosaic_for_novelty": True,
            "separate_primary_secondary_per_independent_claim": True,
            "primary_near_tie_margin": PRIMARY_NEAR_TIE_MARGIN,
            "dependent_reuse_tiebreak_max": DEPENDENT_REUSE_TIEBREAK_MAX,
            "dependent_reuse_is_tiebreak_only": True,
            "ordered_pair_evaluation": True,
            "distinctive_core_direct_disclosure_first": True,
            "generic_breadth_is_tiebreak_only": True,
            "distinctive_core_near_tie_margin": DISTINCTIVE_CORE_NEAR_TIE_MARGIN,
        },
        "analysis_tracks": {
            family_key: selection.get("analysis_track", "inventive_step_combination")
            for family_key, selection in family_selections.items()
        },
        "novelty_screens": {
            family_key: selection.get("novelty_screen", {})
            for family_key, selection in family_selections.items()
        },
        "gap_evidence_matrix": _build_gap_evidence_matrix(
            caches, claims, primary_inv_idx, num_docs
        ),
        "family_gap_evidence_matrices": {
            family_key: _build_gap_evidence_matrix(
                caches,
                family_groups.get(int(family_key), []),
                int(selection.get("primary_idx", primary_inv_idx)),
                num_docs,
            )
            for family_key, selection in family_selections.items()
        },
        "doc_name_mapping": doc_name_mapping,
        "selection_locks": selection_locks,
        "primary_gaps_count": gap_count,
        "soft_gaps_count": soft_gap_count,
        "secondary_reason": secondary_reason,
        "combination_rationale": combination_rationale,
        "combination_rationale_type": combination_rationale["type"],
        "single_sufficient_claims": sorted(set(single_sufficient_claims) | set(family_single_sufficient)),
        "secondary_comp_score": (
            selected_secondary_detail.get("raw_sub_score", selected_secondary_detail.get("sub_score", 0))
            if representative_family
            else complementarity_scores.get(secondary_inv_idx, 0)
            if secondary_inv_idx is not None else 0
        ),
        "conventional_support_policy": {
            "normal_max_references": MAX_INDEPENDENT_REFS,
            "exceptional_max_references": MAX_INDEPENDENT_REFS_WITH_CONVENTIONAL_SUPPORT,
            "third_reference_role": "conventional_support",
        },
        "dependent_claim_policy": {
            "inherit_parent_chain": True,
            "max_new_references_per_claim": MAX_DEPTH_INCREMENT,
            "require_one_new_reference_to_cover_all_remaining_elements": True,
        },
        "confidence": confidence,
        "quantitative_assessment": assess_claims(
            claims,
            [caches.get(i) for i in range(num_docs)],
            chains,
        ),
        "chains": chains,
    }

    out_path = Path(job_dir) / "citation_chain.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    score_summary = ", ".join(
        f"{doc_name_mapping[str(i)]}({prior_docs[i].filename})={inv_scores[i]}점"
        for i in ordered
    )
    logger.info(f"Citation chain 저장: {score_summary}")
    return result


# ---------------------------------------------------------------------------
# 체인 재귀 빌드
# ---------------------------------------------------------------------------

def _build_chains_recursive(
    claims: List[ParsedClaim],
    primary_inv_idx: int,
    secondary_inv_idx: Optional[int],
    inv_scores: Dict[int, int],
    num_docs: int,
    chains: Dict[str, Dict],
    single_sufficient_claims: Optional[set] = None,
    caches: Optional[Dict[int, Optional[Dict]]] = None,
) -> None:
    """
    독립항: primary + secondary (보완성으로 선정된 것)
    종속항: 부모 체인 전체 상속 + 해당 종속항에서 추가 인용발명 최대 1개

    종속항 추가 인용발명 선정:
      1순위: 종속항 자체 대비 캐시(자기 청구항 번호 키) — 상속 문헌이 못 채운
             추가 구성을 실제로 개시하는 문헌만 추가 (공백 없으면 추가 안 함)
      자체 대비 캐시가 없으면 근거 없는 새 문헌을 추가하지 않음
    """
    # 처리 순서: 독립항 먼저, 그 다음 종속항 (부모가 chains에 있어야 함)
    ordered_claims = sorted(claims, key=lambda c: (0 if c.claim_type == "independent" else 1, c.claim_number))
    known_claim_numbers = {item.claim_number for item in claims}

    for claim in ordered_claims:
        key = str(claim.claim_number)
        if key in chains:
            continue

        if claim.claim_type == "independent":
            total = [primary_inv_idx]
            if secondary_inv_idx is not None and key not in (single_sufficient_claims or set()):
                total.append(secondary_inv_idx)
            chains[key] = {
                "total": total[:MAX_INDEPENDENT_REFS],
                "inherited": [],
                "added": total[:MAX_INDEPENDENT_REFS],
                "parent": None,
            }

        else:  # dependent
            parent_num = claim.parent_claim
            parent_key = str(parent_num) if parent_num else None
            parent_available = bool(parent_num and parent_num in known_claim_numbers)

            if parent_available and parent_key and parent_key in chains:
                inherited = chains[parent_key]["total"][:]
            elif parent_num and not parent_available:
                # 참조한 부모항의 실체가 없으면 상속 근거를 만들지 않고,
                # 이 종속항에 직접 기재된 추가 기술 특징만 문헌과 대비한다.
                inherited = []
            else:
                inherited = [primary_inv_idx]
                if secondary_inv_idx is not None:
                    inherited.append(secondary_inv_idx)

            # 추가 인용발명: 종속항 자체 판정 캐시가 있을 때만 선정한다.
            dep_has_cache = caches is not None and any(
                (caches.get(i) or {}).get(key) for i in range(num_docs)
            )
            if dep_has_cache:
                expected_labels = {
                    normalize_label(element.label)
                    for element in claim.elements
                    if normalize_label(element.label)
                }
                added, dependent_trace = _dependent_added_inv(
                    key,
                    inherited,
                    caches,
                    num_docs,
                    expected_labels=expected_labels,
                    expected_text_by_label={
                        normalize_label(element.label): element.text
                        for element in claim.elements
                        if normalize_label(element.label)
                    },
                    expected_importance_by_label={
                        normalize_label(element.label): _importance_value(element.importance)
                        for element in claim.elements
                        if normalize_label(element.label)
                    },
                )
                added = added[:MAX_DEPTH_INCREMENT]
            else:
                # 종속항 자체 대비 근거가 없으면 독립항 점수가 높은 문헌을
                # 임의로 추가하지 않는다. 새 문헌은 해당 종속항 구성과의
                # 실제 대응 결과가 있을 때에만 채택한다.
                expected_labels = {
                    normalize_label(element.label)
                    for element in claim.elements
                    if normalize_label(element.label)
                }
                added = []
                dependent_trace = {
                    "assessment_scope": "additional_limitations",
                    "selection_basis": "comparison_evidence_missing",
                    "candidate_scores": [],
                    "selected_document": None,
                }
            total = inherited + added
            uncovered_labels = _dependent_uncovered_labels(
                key,
                total,
                caches or {},
                expected_labels,
            )

            chains[key] = {
                "total": total,
                "inherited": inherited,
                "added": added,
                "parent": parent_num,
                "family_root": (
                    chains.get(parent_key, {}).get("family_root", parent_num)
                    if parent_key else None
                ),
                "parent_available": parent_available,
                "coverage_complete": not uncovered_labels,
                "uncovered_labels": sorted(uncovered_labels),
                "max_new_references": MAX_DEPTH_INCREMENT,
                "decision_trace": {
                    **dependent_trace,
                    "parent_claim": parent_num,
                    "parent_available": parent_available,
                    "inherited_documents": inherited,
                    "added_documents": added,
                    "coverage_complete": not uncovered_labels,
                    "remaining_uncovered_labels": sorted(uncovered_labels),
                    "decision_status": (
                        "parent_context_missing"
                        if parent_num and not parent_available
                        else "additional_limitations_covered"
                        if not uncovered_labels
                        else "partial_support_remaining_gap"
                        if added
                        else "no_single_reference_covers_remaining_gaps"
                    ),
                },
            }


# ---------------------------------------------------------------------------
# 다음 추가 인용발명 선정 (종속항용)
# ---------------------------------------------------------------------------

def _dependent_added_inv(
    claim_key: str,
    inherited: List[int],
    caches: Dict[int, Optional[Dict]],
    num_docs: int,
    expected_labels: Optional[set[str]] = None,
    expected_text_by_label: Optional[Dict[str, str]] = None,
    expected_importance_by_label: Optional[Dict[str, int]] = None,
) -> tuple[List[int], Dict]:
    """추가 한정에 대한 순수 증가분과 근거 품질로 새 문헌 하나를 선정한다."""
    def _items(doc_idx: int) -> list:
        items = (caches.get(doc_idx) or {}).get(claim_key, [])
        return items if isinstance(items, list) else []

    def _cross_claim_reuse_count(doc_idx: int) -> int:
        """동점 후보 중 다른 종속항에도 직접 재사용 가능한 문헌을 우선한다."""
        cache = caches.get(doc_idx) or {}
        count = 0
        for other_key, values in cache.items():
            if str(other_key).startswith("_") or str(other_key) == claim_key or not isinstance(values, list):
                continue
            if any(
                item.get("quote")
                and _JUDGMENT_SCORE.get(item.get("judgment", "대응 없음"), 0) >= _SECONDARY_FILL_THRESHOLD
                for item in values
                if isinstance(item, dict)
            ):
                count += 1
        return count

    inherited_best: Dict[str, int] = {}
    all_labels: set = set(expected_labels or set())
    for i in range(num_docs):
        for item in _items(i):
            label = normalize_label(item.get("label", ""))
            all_labels.add(label)
            if i in inherited:
                score = _JUDGMENT_SCORE.get(item.get("judgment", "대응 없음"), 0)
                inherited_best[label] = max(inherited_best.get(label, 0), score)

    gaps = {l for l in all_labels if inherited_best.get(l, 0) < _PRIMARY_COVER_THRESHOLD}
    soft_gaps = {l for l in all_labels if inherited_best.get(l, 0) == _SOFT_GAP_SCORE}
    targets = gaps or soft_gaps
    trace = {
        "assessment_scope": "additional_limitations",
        "additional_labels": sorted(all_labels),
        "inherited_best_scores": inherited_best,
        "hard_gaps": sorted(gaps),
        "soft_gaps": sorted(soft_gaps),
        "candidate_scores": [],
        "selected_document": None,
        "selection_basis": "covered_by_inherited" if not targets else "no_candidate",
    }
    if not targets:
        return [], trace

    candidates = []
    for doc_idx in range(num_docs):
        if doc_idx in inherited:
            continue
        by_label = {
            normalize_label(item.get("label", "")): item
            for item in _items(doc_idx)
        }
        total_weight = 0.0
        weighted_gain = 0.0
        evidence_gain = 0.0
        strongly_covered = set()
        improved = set()
        label_details = []
        for label in sorted(targets):
            item = by_label.get(label, {})
            rank = _JUDGMENT_SCORE.get(item.get("judgment", "대응 없음"), 0)
            inherited_rank = inherited_best.get(label, 0)
            weight = float((expected_importance_by_label or {}).get(label, 3))
            gain = max(0, rank - inherited_rank) / 5.0
            evidence_factor = (
                0.55
                + 0.25 * bool(item.get("quote"))
                + 0.10 * bool(item.get("chunk_id") or item.get("paragraph_no"))
                + 0.10 * bool(item.get("판단_이유") or item.get("similarity_reason"))
            )
            total_weight += weight
            weighted_gain += weight * gain
            evidence_gain += weight * gain * evidence_factor
            if rank >= (_SECONDARY_FILL_THRESHOLD if gaps else _SECONDARY_IMPROVE_THRESHOLD):
                strongly_covered.add(label)
            if rank > inherited_rank and item.get("quote"):
                improved.add(label)
            label_details.append({
                "label": label,
                "inherited_rank": inherited_rank,
                "candidate_rank": rank,
                "gain": round(gain * 100, 1),
                "has_quote": bool(item.get("quote")),
            })
        denominator = total_weight or 1.0
        marginal_gain = weighted_gain / denominator
        evidence_adjusted_gain = evidence_gain / denominator
        full_cover = strongly_covered == targets
        candidate_score = 0.75 * marginal_gain + 0.25 * evidence_adjusted_gain
        candidate = {
            "doc_idx": doc_idx,
            "score": round(candidate_score * 100, 2),
            "marginal_gain": round(marginal_gain * 100, 1),
            "evidence_adjusted_gain": round(evidence_adjusted_gain * 100, 1),
            "full_cover": full_cover,
            "strongly_covered_labels": sorted(strongly_covered),
            "improved_with_quote_labels": sorted(improved),
            "cross_claim_reuse_count": _cross_claim_reuse_count(doc_idx),
            "labels": label_details,
        }
        trace["candidate_scores"].append(candidate)
        if candidate_score > 0 and improved:
            candidates.append(candidate)

    full_candidates = [candidate for candidate in candidates if candidate["full_cover"]]
    if full_candidates:
        pool = full_candidates
    elif candidates:
        # 완전 보완 문헌이 없더라도 직접 근거가 있는 문헌 하나만 부분 근거로
        # 보존하고, 나머지 공백은 coverage_complete/uncovered_labels에 남긴다.
        pool = candidates
    elif len(targets) == 1:
        # 단순히 관련 발췌가 존재한다는 이유만으로 `차이` 문헌을 실제 결합 체인에
        # 승격하지 않는다. 최소한 일부 유사 판정이 있는 문헌만 부분 근거로 보존한다.
        target_label = next(iter(targets))
        pool = []
        for doc_idx in range(num_docs):
            if doc_idx in inherited:
                continue
            item = next(
                (
                    value for value in _items(doc_idx)
                    if normalize_label(value.get("label", "")) == target_label
                ),
                None,
            )
            if item and item.get("quote"):
                rank = _JUDGMENT_SCORE.get(item.get("judgment", "대응 없음"), 0)
                if rank < _DEPENDENT_PARTIAL_SUPPORT_THRESHOLD:
                    continue
                pool.append({
                    "doc_idx": doc_idx,
                    "score": float(rank),
                    "full_cover": False,
                })
    else:
        pool = []
    if not pool:
        return [], trace
    selected = max(
        pool,
        key=lambda candidate: (
            candidate["score"],
            candidate.get("cross_claim_reuse_count", 0),
            -candidate["doc_idx"],
        ),
    )
    trace["selected_document"] = selected["doc_idx"]
    trace["selection_basis"] = (
        "single_document_full_gap_filler"
        if selected["full_cover"] and gaps
        else "single_document_soft_gap_improver"
        if selected["full_cover"]
        else "partial_support_only"
    )
    logger.info(
        "청구항 %s (종속항): doc[%s] 추가 선정 (%s, 점수 %.2f)",
        claim_key,
        selected["doc_idx"],
        trace["selection_basis"],
        selected["score"],
    )
    return [selected["doc_idx"]], trace


def _dependent_uncovered_labels(
    claim_key: str,
    total: List[int],
    caches: Dict[int, Optional[Dict]],
    expected_labels: set[str],
) -> set[str]:
    """Return dependent-claim elements not covered by the allowed chain."""
    best_scores = {label: 0 for label in expected_labels}
    for doc_idx in total:
        items = (caches.get(doc_idx) or {}).get(claim_key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            label = normalize_label(item.get("label", ""))
            if label not in best_scores:
                continue
            score = _JUDGMENT_SCORE.get(item.get("judgment", "대응 없음"), 0)
            best_scores[label] = max(best_scores[label], score)
    return {
        label
        for label, score in best_scores.items()
        if score < _PRIMARY_COVER_THRESHOLD
    }


# ---------------------------------------------------------------------------
# 저장 / 로드 / 조회
# ---------------------------------------------------------------------------

def get_claim_chain_info(chain_data: Dict, claim_number: int) -> Optional[Dict]:
    if not chain_data:
        return None
    chain = chain_data.get("chains", {}).get(str(claim_number))
    if chain is None:
        return None
    # doc_name_mapping·신뢰도 병합하여 반환 (report_generator에서 사용)
    chain_with_mapping = dict(chain)
    chain_with_mapping["doc_name_mapping"] = chain_data.get("doc_name_mapping", {})
    chain_with_mapping["confidence"] = chain_data.get("confidence", {}).get(str(claim_number))
    chain_with_mapping["quantitative_assessment"] = (
        chain_data.get("quantitative_assessment", {}).get(str(claim_number))
    )
    if (
        len(chain_with_mapping.get("total", [])) == 1
        and chain_with_mapping.get("common_general_knowledge")
    ):
        rationale = _combination_rationale_for(
            None, candidate_types=["common_general_knowledge"]
        )
        chain_with_mapping["combination_rationale"] = rationale
        chain_with_mapping["combination_rationale_type"] = "common_general_knowledge"
    elif len(chain_with_mapping.get("total", [])) > 1:
        conventional_support = chain_with_mapping.get("conventional_support") or {}
        if conventional_support.get("position") == 2:
            rationale = dict(_COMBINATION_RATIONALES["conventional_support"])
            rationale.update({
                "type": "conventional_support",
                "candidate_types": ["conventional_support"],
                "warnings": [],
                "score_detail": {},
                "secondary_reason": "conventional_support",
            })
            chain_with_mapping["combination_rationale"] = rationale
            chain_with_mapping["combination_rationale_type"] = "conventional_support"
        else:
            chain_with_mapping["combination_rationale"] = (
                chain.get("combination_rationale") or chain_data.get("combination_rationale")
            )
            chain_with_mapping["combination_rationale_type"] = (
                chain.get("combination_rationale_type") or chain_data.get("combination_rationale_type")
            )
    elif chain.get("combination_rationale"):
        chain_with_mapping["combination_rationale"] = chain.get("combination_rationale")
        chain_with_mapping["combination_rationale_type"] = chain.get("combination_rationale_type")
    family_root = chain.get("family_root")
    if family_root is not None:
        chain_with_mapping["family_selection"] = chain_data.get("families", {}).get(str(family_root), {})
    return chain_with_mapping


def format_inv_list(indices: List[int], doc_name_mapping: Optional[Dict[str, str]] = None) -> str:
    if not indices:
        return ""
    if doc_name_mapping:
        names = [doc_name_mapping.get(str(i), f"인용발명 {i + 1}") for i in indices]
    else:
        names = [f"인용발명 {i + 1}" for i in indices]

    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} 및 {names[1]}"
    return ", ".join(names[:-1]) + f" 및 {names[-1]}"
