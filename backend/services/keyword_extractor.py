"""
keyword_extractor.py — 특허 검색 키워드 추출 서비스

extract_local_keywords() — LLM 없음, 즉시 반환
ParsedClaim.elements에서 규칙 기반으로 한국어 기술 명사를 추출
"""
from __future__ import annotations

import re

from backend.models.schemas import ParsedClaim

# ---------------------------------------------------------------------------
# 불용어 목록 — 특허 상투 표현 / 조사 / 기능어
# ---------------------------------------------------------------------------
_KO_STOPWORDS = {
    # 특허 상투 표현
    "있어서", "이루어진", "포함하는", "구성된", "특징으로", "하는", "되는",
    "위한", "의한", "따른", "관한", "대한", "관련", "것을", "것이",
    "장치", "방법", "시스템", "수단", "단계", "구성", "부분",
    "제1", "제2", "제3", "제4", "상기", "해당", "각각", "적어도",
    "하나", "이상", "이하", "미만", "초과", "이내", "범위",
    # 조사·어미
    "으로", "에서", "에게", "에게서", "로부터", "에의", "에의한",
    "및", "또는", "그리고", "하여", "하며", "하고",
    # 기타
    "통해", "통하여", "기반", "기초", "이용", "사용", "활용",
    "제어", "처리", "관리", "수행", "실행", "동작",
}

# 한국어 기술 명사 추출 패턴 (2~10자 한자·한글 단어)
_KO_TERM_RE = re.compile(r"[가-힣]{2,10}")
# 영문 기술 단어 패턴 (3자 이상, 대소문자 혼합)
_EN_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")
# 수치 한정 패턴 (예: 0.1 ~ 5.0 nm, 500℃ 이하)
_NUM_LIMIT_RE = re.compile(
    r"[\d.,]+\s*(?:~|내지|이상|이하|미만|초과)\s*[\d.,]+\s*(?:[a-zA-Z㎛㎚㎠㎡°℃%]*)?"
    r"|[\d.,]+\s*(?:[a-zA-Z㎛㎚㎠㎡°℃%]+)"
)


def _extract_terms_from_text(text: str) -> list[str]:
    """단일 텍스트에서 기술 용어를 추출한다."""
    terms = []
    # 한국어 명사 추출
    for m in _KO_TERM_RE.finditer(text):
        w = m.group()
        if w not in _KO_STOPWORDS and len(w) >= 2:
            terms.append(w)
    # 영문 기술 단어
    for m in _EN_TERM_RE.finditer(text):
        w = m.group()
        if len(w) >= 3:
            terms.append(w)
    # 수치 한정 표현
    for m in _NUM_LIMIT_RE.finditer(text):
        terms.append(m.group().strip())
    return terms


def _is_preamble_element(element_text: str) -> bool:
    """구성요소가 전제부(preamble)인지 판별한다.
    '~에 있어서', '~장치', '~방법' 등 기존 공지기술 부분은 제외.
    """
    preamble_patterns = [
        r"에 있어서",
        r"에 관한",
        r"을 위한",
        r"을 포함하는\s*(?:장치|시스템|방법|기기|모듈)$",
    ]
    for p in preamble_patterns:
        if re.search(p, element_text):
            return True
    return False


# ---------------------------------------------------------------------------
# 공개 API: 로컬 키워드 추출 (LLM 없음)
# ---------------------------------------------------------------------------

def extract_local_keywords(claim: ParsedClaim) -> dict:
    """ParsedClaim의 elements에서 규칙 기반으로 키워드를 추출한다.

    Returns:
        {
          "claim_number": int,
          "keywords": [
            { "term": str, "type": "ko"|"en"|"numeric",
              "importance": "core"|"secondary",
              "element_label": str }
          ]
        }
    """
    seen: set[str] = set()
    keywords: list[dict] = []

    for i, elem in enumerate(claim.elements):
        text = elem.text.strip()

        # 전제부 구성요소는 건너뜀
        if _is_preamble_element(text):
            continue

        # 중요도 결정
        # - 수치 한정이 있거나 첫 2개 비-전제부 구성요소 → core
        has_numeric = bool(_NUM_LIMIT_RE.search(text))
        is_early = i <= 1  # elements 인덱스 0,1
        importance = "core" if (has_numeric or is_early) else "secondary"

        # elem.importance 필드(별점)도 반영
        try:
            star = int(elem.importance)
            if star >= 4:
                importance = "core"
        except (ValueError, TypeError):
            pass

        terms = _extract_terms_from_text(text)

        for term in terms:
            if term in seen or term.lower() in _KO_STOPWORDS:
                continue
            seen.add(term)

            # 타입 분류
            if _NUM_LIMIT_RE.fullmatch(term.strip()):
                t = "numeric"
            elif _EN_TERM_RE.fullmatch(term):
                t = "en"
            else:
                t = "ko"

            keywords.append({
                "term": term,
                "type": t,
                "importance": importance,
                "element_label": elem.label,
            })

    return {
        "claim_number": claim.claim_number,
        "keywords": keywords,
    }
