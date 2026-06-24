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
"""
from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from backend.models.schemas import ExtractedDocument, ParsedClaim
from backend.services.citation_extractor import normalize_label

logger = logging.getLogger(__name__)

MAX_INDEPENDENT_REFS = 2   # 독립항에 사용할 최대 인용발명 수
MAX_INDEPENDENT_REFS_WITH_CONVENTIONAL_SUPPORT = 3
MAX_DEPTH_INCREMENT = 1    # 종속항 1단계당 추가 허용 인용발명 수
CITATION_CHAIN_POLICY_VERSION = 6

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
# "일부 유사"는 완전 보완 근거로 보지는 않지만, 주지관용 논거보다 문헌 근거로
# 먼저 검토할 가치가 있으므로 마지막 단계의 보조 인용발명 후보로 인정한다.
_SECONDARY_SUPPORT_THRESHOLD = 2  # "일부 유사" 이상
# 주인용발명 단독 가중 유사도가 이 이상이면 소프트 공백이 있어도 결합 불필요 (단독 충분)
SINGLE_SUFFICIENT_SIMILARITY = 91.0

# 판정 라벨 → 유사도 퍼센트 (system_compare.txt 의 판정 밴드 대표값과 동일 기준)
# 신뢰도 계산용 — 대비 LLM이 이미 내린 판정을 재사용하므로 추가 LLM 호출 없음.
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
UNCOVERED_PERCENT_THRESHOLD = 35  # 이 미만이면 해당 구성요소 '미커버'로 간주

# 주/보조 인용발명 선정 점수:
# - 주인용발명은 심사관식 "가장 가까운 출발점"이 되도록 전체 골격 커버리지를 우선한다.
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
    r"컨트롤러|메모리|저장부|통신부|송수신부|인터페이스|입력부|출력부|표시부|"
    r"디스플레이|전원부|배터리|바퀴|휠|모터|센서|하우징|프레임|서버|단말"
    r")",
    re.IGNORECASE,
)
_SPECIALIZED_CONSTRAINT_RE = re.compile(
    r"(?:\d|피드백|전역|국소|학습|암호|복호|보정|적응|동기|임계|특정\s*조건|"
    r"상호\s*작용|연동|based\s+on|in\s+response\s+to|feedback|global|local|"
    r"adaptive|threshold|synchron|encrypt|decrypt|calibrat)",
    re.IGNORECASE,
)
_FUNCTIONAL_VERB_RE = re.compile(
    r"(?:제어|수신|저장|비교|판단|생성|전송|변환|검출|처리|구동|control|receive|"
    r"store|compare|determin|generat|transmit|convert|detect|process|drive)",
    re.IGNORECASE,
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
            weights[(claim_key, normalize_label(element.label))] = _importance_value(element.importance)
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


def _weighted_average(rows: list[Dict], value_key: str = "similarity") -> float:
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        weight = float(row.get("importance", 3))
        numerator += weight * float(row.get(value_key, 0.0))
        denominator += weight
    return numerator / denominator if denominator else 0.0


def _comparison_rows(cache: Optional[Dict], claims: List[ParsedClaim]) -> list[Dict]:
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
            rows.append({
                "claim_key": str(claim.claim_number),
                "label": label,
                "element_index": idx,
                "importance": _importance_value(element.importance),
                "item": item,
                "judgment": judgment,
                "rank": _JUDGMENT_SCORE.get(judgment, 0),
                "similarity": _similarity_for_judgment(judgment),
                "has_quote": bool(item.get("quote")),
            })
    return rows


def _score_prior_cache(cache: Optional[Dict], claims: List[ParsedClaim]) -> tuple[float, int, Dict]:
    """인용발명별 주 인용발명 후보 점수와 의미 있는 매칭 수를 계산한다.

    주인용발명은 결합 거절의 출발점이므로, 핵심 구성 하나의 보유 여부보다
    청구항 전체 구조에 대한 가까운 정도를 우선한다.
    """
    if not cache:
        return 0.0, 0, {}

    rows = _comparison_rows(cache, claims)
    if not rows:
        return 0.0, 0, {}

    element_coverage = _weighted_average(rows)
    core_rows = [r for r in rows if r["importance"] >= _CORE_IMPORTANCE_THRESHOLD]
    if not core_rows:
        core_rows = [r for r in rows if r["element_index"] < 2] or rows
    core_coverage = _weighted_average(core_rows)

    strong_weight = sum(r["importance"] for r in rows if r["rank"] >= _PRIMARY_COVER_THRESHOLD)
    total_weight = sum(r["importance"] for r in rows) or 1
    match_ratio = strong_weight / total_weight

    # Purpose/effect-specific fields are not yet stored in comparisons_*.json.
    # Use conservative local proxies so candidate ranking remains deterministic
    # and token-free until a later extraction step provides direct metrics.
    field_target = min(1.0, 0.70 * element_coverage + 0.30 * match_ratio)
    purpose_rows = [r for r in rows if r["importance"] >= _CORE_IMPORTANCE_THRESHOLD or r["element_index"] == 0]
    purpose_problem = _weighted_average(purpose_rows or rows)
    effect_function = min(1.0, 0.65 * element_coverage + 0.35 * core_coverage)
    embodiment_context = min(1.0, 0.60 * element_coverage + 0.40 * match_ratio)

    main_score = (
        0.30 * core_coverage
        + 0.25 * field_target
        + 0.20 * purpose_problem
        + 0.15 * effect_function
        + 0.10 * embodiment_context
    )
    match_count = sum(1 for r in rows if r["rank"] >= _SECONDARY_FILL_THRESHOLD)
    detail = {
        "main_score": round(main_score, 4),
        "core_coverage": round(core_coverage, 4),
        "element_coverage": round(element_coverage, 4),
        "field_target_proximity": round(field_target, 4),
        "purpose_problem_proximity": round(purpose_problem, 4),
        "effect_function_proximity": round(effect_function, 4),
        "embodiment_context_proximity": round(embodiment_context, 4),
        "match_ratio": round(match_ratio, 4),
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
        if j_score >= _SECONDARY_SUPPORT_THRESHOLD:
            score += j_score * (weights or {}).get((claim_key, label), 3)
    return score


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
        # if it gives strong, explicit evidence. Otherwise keep Template A and
        # identify the limitations as common-general-knowledge review targets.
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


def _score_secondary_candidate(
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

    weighted_fill = 0.0
    weighted_quote = 0.0
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
        sim = _similarity_for_judgment(judgment)
        rank = _JUDGMENT_SCORE.get(judgment, 0)
        weighted_fill += base_weight * sim
        if item and item.get("quote") and sim >= 0.35:
            weighted_quote += base_weight
        if rank >= _SECONDARY_FILL_THRESHOLD and (claim_key, label) in primary_gaps:
            hard_fill += 1
            filled_labels.append(label)
        if rank >= _SECONDARY_IMPROVE_THRESHOLD and (claim_key, label) in soft_gaps:
            soft_fill += 1
            filled_labels.append(label)

        primary_item = _cache_item(primary_cache, claim_key, label)
        primary_sim = _similarity_for_judgment(primary_item.get("judgment", "대응 없음")) if primary_item else 0.0
        if 0.15 < primary_sim <= 0.55 and sim >= 0.85:
            specific_fill += 1

    gap_fill = weighted_fill / denominator if denominator else 0.0
    quote_explicitness = weighted_quote / denominator if denominator else 0.0
    _, _, primary_like_detail = _score_prior_cache(cache, claims)
    field_problem = float(primary_like_detail.get("field_target_proximity", 0.0))
    applicability = min(1.0, 0.65 * field_problem + 0.35 * gap_fill)
    reportability = min(1.0, 0.70 * quote_explicitness + 0.30 * gap_fill)

    sub_score = (
        0.55 * gap_fill
        + 0.20 * quote_explicitness
        + 0.10 * field_problem
        + 0.10 * applicability
        + 0.05 * reportability
    )

    rationale_types: list[str] = []
    if hard_fill:
        rationale_types.append("gap_filling")
    if soft_fill:
        rationale_types.append("known_tech_application")
    if specific_fill:
        rationale_types.append("specific_selection")
    if hard_fill and field_problem >= 0.35:
        rationale_types.append("problem_solution")
    if applicability >= 0.55:
        rationale_types.append("obvious_to_try")
    if gap_fill > 0 and field_problem < 0.35:
        rationale_types.append("design_variation")
    if len(set(filled_labels)) >= 2:
        rationale_types.append("aggregation")
    if not rationale_types and quote_explicitness > 0:
        rationale_types.append("supporting_evidence")

    warnings: list[str] = []
    if field_problem < 0.25 and gap_fill > 0:
        warnings.append("기술분야/문제 관련성이 낮을 수 있으므로 보고서에서 결합 가능성을 별도 점검해야 합니다.")
    if gap_fill > 0 and applicability < 0.35:
        warnings.append("적용 가능성 또는 예측 가능성이 약할 수 있으므로 무리한 결합 단정은 피해야 합니다.")

    if hard_fill:
        reason = "hard"
    elif soft_fill:
        reason = "soft"
    elif quote_explicitness > 0:
        reason = "support"
    else:
        reason = None

    detail = {
        "sub_score": round(sub_score, 4),
        "gap_fill": round(gap_fill, 4),
        "quote_explicitness": round(quote_explicitness, 4),
        "field_problem_relatedness": round(field_problem, 4),
        "applicability_predictability": round(applicability, 4),
        "reportability": round(reportability, 4),
        "hard_fill_count": hard_fill,
        "soft_fill_count": soft_fill,
        "candidate_rationale_types": list(dict.fromkeys(rationale_types)),
        "warnings": warnings,
        "secondary_reason": reason,
    }
    return round(sub_score * 100, 2), detail


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


def _element_percents(cache: Optional[Dict], claim: ParsedClaim) -> Dict[str, int]:
    """캐시 판정을 구성요소 라벨 → 유사도 퍼센트로 변환한다 (라벨 정규화 매칭)."""
    if not cache:
        return {}
    items = cache.get(str(claim.claim_number), [])
    if not isinstance(items, list):
        return {}
    out: Dict[str, int] = {}
    for item in items:
        if isinstance(item, dict):
            label = normalize_label(item.get("label"))
            out[label] = _LABEL_PERCENT.get(item.get("judgment", "대응 없음"), 0)
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
    p_pcts = _element_percents(primary_cache, claim)
    supporting_percent_maps = [_element_percents(secondary_cache, claim)]
    supporting_percent_maps.extend(
        _element_percents(cache, claim) for cache in (additional_caches or [])
    )

    p_num = c_num = den = 0.0
    uncovered: List[str] = []
    for e in claim.elements:
        imp = int(e.importance) if str(e.importance).isdigit() else 3
        label = normalize_label(e.label)
        p = p_pcts.get(label, 0)
        c = max(p, *(pcts.get(label, 0) for pcts in supporting_percent_maps))
        p_num += imp * p
        c_num += imp * c
        den += imp * 100
        if c < UNCOVERED_PERCENT_THRESHOLD:
            uncovered.append(e.label)

    if den == 0:
        return {"primary_similarity": 0.0, "combined_similarity": 0.0, "uncovered_labels": uncovered}
    return {
        "primary_similarity": round(p_num / den * 100, 1),
        "combined_similarity": round(c_num / den * 100, 1),
        "uncovered_labels": uncovered,
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


# ---------------------------------------------------------------------------
# 메인: 비교 캐시 기반 체인 빌드 (v3 — 보완성 기반 secondary 선정)
# ---------------------------------------------------------------------------

def build_citation_chain_from_comparisons(
    job_dir: str,
    claims: List[ParsedClaim],
    prior_docs: List[ExtractedDocument],
) -> Dict:
    """
    comparisons_{doc_idx}.json 에서 판정 점수를 집계하여 주인용발명을 선정하고,
    보완성(complementarity) 기준으로 보조인용발명을 선정한다.

    선정 원칙:
    1. 주인용발명: 청구항 전체 골격에 가장 가까운 문헌
       - 낮은 중요도의 주지관용 구성도 발명의 출발점 판단에는 최소 가중치로 반영
       - importance>=4 핵심 구성의 강한 대응/미커버는 보조적 보너스·패널티로 반영
    2. 보조인용발명: 주인용발명이 커버하지 못하거나 약하게 커버한 중요 구성요소를 가장 잘 채우는 문헌
       - 보조인용발명이 공백 구성요소를 채워 결합 시 100% 커버 가능하면 채택
       - 공백을 전혀 못 채우면 Template A (단독 인용) 사용
    3. 종속항: 부모 청구항의 인용발명을 상속 + 추가 공백 채우는 다음 인용발명 추가
    """
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

    # ── 2단계: 주인용발명 선정 (전체 골격에 가장 가까운 문헌) ─────────────
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

    # ── 5단계: 청구항별 체인 구성 ───────────────────────────────────────────
    chains: Dict[str, Dict] = {}
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
    ordered = [primary_inv_idx]
    ordered_claims = sorted(
        claims,
        key=lambda item: (0 if item.claim_type == "independent" else 1, item.claim_number),
    )
    for claim in ordered_claims:
        for doc_idx in (chains.get(str(claim.claim_number), {}).get("total") or [])[1:]:
            if doc_idx not in ordered:
                ordered.append(doc_idx)
    for doc_idx in conventional_doc_order:
        if doc_idx not in ordered:
            ordered.append(doc_idx)
    remaining = sorted(
        [i for i in range(num_docs) if i not in ordered],
        key=lambda k: inv_scores[k],
        reverse=True,
    )
    ordered += remaining
    doc_name_mapping = {
        str(doc_idx): f"인용발명 {rank + 1}"
        for rank, doc_idx in enumerate(ordered)
    }

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

    selected_secondary_detail = (
        secondary_candidate_details.get(secondary_inv_idx, {})
        if secondary_inv_idx is not None else {}
    )
    if secondary_inv_idx is None and (primary_gaps or soft_gaps):
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

    result = {
        "policy_version": CITATION_CHAIN_POLICY_VERSION,
        "primary_inv_idx": primary_inv_idx,
        "primary_inv_name": doc_name_mapping[str(primary_inv_idx)],
        "scoring_method": "main_score_closest_starting_point_sub_score_gap_filler",
        "inv_scores": {str(k): v for k, v in inv_scores.items()},
        "inv_match_counts": {str(k): v for k, v in inv_match_counts.items()},
        "primary_score_details": {str(k): v for k, v in primary_score_details.items()},
        "primary_candidates": primary_candidates,
        "secondary_candidate_scores": {str(k): v for k, v in secondary_candidate_scores.items()},
        "secondary_candidate_details": {str(k): v for k, v in secondary_candidate_details.items()},
        "secondary_candidates": secondary_candidates,
        "doc_name_mapping": doc_name_mapping,
        "primary_gaps_count": gap_count,
        "soft_gaps_count": soft_gap_count,
        "secondary_reason": secondary_reason,
        "combination_rationale": combination_rationale,
        "combination_rationale_type": combination_rationale["type"],
        "single_sufficient_claims": sorted(single_sufficient_claims),
        "secondary_comp_score": complementarity_scores.get(secondary_inv_idx, 0)
                                 if secondary_inv_idx is not None else 0,
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
                added = _dependent_added_inv(
                    key,
                    inherited,
                    caches,
                    num_docs,
                    expected_labels=expected_labels,
                )[:MAX_DEPTH_INCREMENT]
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
                "parent_available": parent_available,
                "coverage_complete": not uncovered_labels,
                "uncovered_labels": sorted(uncovered_labels),
                "max_new_references": MAX_DEPTH_INCREMENT,
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
) -> List[int]:
    """종속항 자체 대비 캐시로 추가 인용발명을 선정한다 (독립항 보완성 로직과 동일 기준).

    상속 문헌이 _PRIMARY_COVER_THRESHOLD 미만으로 남긴 추가 구성(공백)을
    _SECONDARY_FILL_THRESHOLD 이상으로 모두 채우는 단일 문헌 중 보완점수
    최고를 고른다. 서로 다른 두 문헌을 합쳐야만 공백이 채워지는 경우에는
    어느 문헌도 추가하지 않는다.
    """
    def _items(doc_idx: int) -> list:
        items = (caches.get(doc_idx) or {}).get(claim_key, [])
        return items if isinstance(items, list) else []

    inherited_best: Dict[str, int] = {}
    all_labels: set = set(expected_labels or set())
    for i in range(num_docs):
        for item in _items(i):
            label = normalize_label(item.get("label", ""))
            all_labels.add(label)
            if i in inherited:
                score = _JUDGMENT_SCORE.get(item.get("judgment", "대응 없음"), 0)
                inherited_best[label] = max(inherited_best.get(label, 0), score)

    def _best_filler(target_labels: set, min_score: int) -> tuple[Optional[int], int]:
        best_idx, best_score = None, 0
        for i in range(num_docs):
            if i in inherited:
                continue
            score = 0
            covered: set[str] = set()
            for item in _items(i):
                label = normalize_label(item.get("label", ""))
                if label in target_labels:
                    j_score = _JUDGMENT_SCORE.get(item.get("judgment", "대응 없음"), 0)
                    if j_score >= min_score:
                        score += j_score
                        covered.add(label)
            # 한 종속항에 새 문헌을 두 개 더하는 방식은 금지한다. 따라서
            # 후보 하나가 남은 대상 전부를 커버하는 경우만 채택 가능하다.
            if covered == target_labels and score > best_score:
                best_score, best_idx = score, i
        return best_idx, best_score

    gaps = {l for l in all_labels if inherited_best.get(l, 0) < _PRIMARY_COVER_THRESHOLD}
    if gaps:
        best_idx, best_score = _best_filler(gaps, _SECONDARY_FILL_THRESHOLD)
        if best_idx is not None:
            logger.info(
                f"청구항 {claim_key} (종속항): 추가 인용발명 doc[{best_idx}] 선정 "
                f"(공백 {len(gaps)}개, 보완점수 {best_score})"
            )
            return [best_idx]
        logger.info(
            f"청구항 {claim_key} (종속항): 남은 공백 {len(gaps)}개를 "
            "단독으로 모두 보완하는 인용발명이 없어 새 문헌을 추가하지 않음"
        )
        return []

    # 소프트 공백: 상속 문헌이 '일부 차이'로만 커버한 추가 구성을 '실질적 동일'
    # 이상으로 개시하는 문헌이 있으면 추가한다 (독립항 3.5단계와 동일 기준).
    # → 차이점 극복 논리를 주지관용 대신 문헌 근거로 작성할 수 있게 한다.
    soft_gaps = {l for l in all_labels if inherited_best.get(l, 0) == _SOFT_GAP_SCORE}
    if soft_gaps:
        best_idx, best_score = _best_filler(soft_gaps, _SECONDARY_IMPROVE_THRESHOLD)
        if best_idx is not None:
            logger.info(
                f"청구항 {claim_key} (종속항): 추가 인용발명 doc[{best_idx}] 선정(소프트) "
                f"(약점 {len(soft_gaps)}개, 보강점수 {best_score})"
            )
            return [best_idx]

    return []


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
    if len(chain_with_mapping.get("total", [])) > 1:
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
            chain_with_mapping["combination_rationale"] = chain_data.get("combination_rationale")
            chain_with_mapping["combination_rationale_type"] = chain_data.get("combination_rationale_type")
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
