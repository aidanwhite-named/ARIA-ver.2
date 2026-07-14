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
_JUDGMENT_ORDER = {
    "동일": 5,
    "실질적 동일": 4,
    "일부 차이": 3,
    "일부 유사": 2,
    "차이": 1,
    "대응 없음": 0,
}

def _normalize_match_judgment(judgment: str) -> str:
    normalized = re.sub(r"\s+", "", judgment or "")
    return {
        "실질적동일": "실질적 동일",
        "일부차이": "일부 차이",
        "일부유사": "일부 유사",
        "대응없음": "대응 없음",
    }.get(normalized, (judgment or "").strip())

def _normalize_report_markdown(md: str) -> str:
    """Normalize report field labels that the LLM sometimes styles inconsistently."""
    if not md:
        return md
    normalized = re.sub(
        r"(?m)^-\s*(?:\*\*)?\s*인용발명\s*(?:\*\*)?\s*대응\s*원문\s*(?:\*\*)?\s*:[ \t]*(?:\*\*)?",
        "- **인용발명 대응 원문:**",
        md,
    )
    quote_block_pattern = re.compile(
        r"(?ms)(^- \*\*인용발명 대응 원문:\*\*\s*\n)(.*?)(?=\n- (?:개시 상태|청구항 구성|청구항 추가 구성|인용발명 대응 부분 요약|판단 이유|판단 근거|보완 검토)\s*:|\n#{1,6}\s|\Z)"
    )

    def clean_quote_block(match: re.Match) -> str:
        body = re.sub(r"(?:라고|고)?\s*기재되어 있습니다\.", ".", match.group(2))
        body = re.sub(r"\.{2,}", ".", body)
        return match.group(1) + body

    return quote_block_pattern.sub(clean_quote_block, normalized)

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
### [구성요소]

(알파벳 기호) (동일 / 실질적동일 / 일부차이 / 일부유사 / 대응안됨 중 택1) + 해당 판정 구간의 구체적 퍼센트

- 청구항 구성:

- **인용발명 대응 원문:**
  [한국어 원문인 경우] 원문 그대로 적고 단락/본문 위치를 병기
  발췌문 뒤에 `~고 기재되어 있습니다` 같은 보고형 문구를 덧붙이지 않음
  [외국어 원문인 경우(일본어·중국어·영어 등)] 반드시 아래 2줄 구조 사용:
    한국어 직역 또는 준직역 문장
    (단락 [실제번호] 또는 본문 실제 페이지, "원문 발췌") — 안내용 표기를 그대로 출력하지 않음
  [외국어 원문 추가 규칙] 괄호 안 따옴표의 원문 발췌는 중국어 문헌이면 중국어, 일본어 문헌이면 일본어, 영어 문헌이면 영어 그대로 적고 다른 언어로 바꾸지 않음
  [대응 없음인 경우] (인용발명 1에서 해당 구성 확인 불가)

- 인용발명 대응 부분 요약:

- 판단 이유: (판정명이나 퍼센트를 반복 설명하지 말고, 청구항 제한과 인용발명 대응 원문이 구조·기능·입력·출력·처리 조건 중 어느 점에서 대응되는지 또는 어느 핵심 제한이 빠져 대응이 불충분한지만 1~2문장으로 작성)


[종합분석요약]

- 결론: (1~2문장으로만 작성한다. 1문장째에는 핵심 일치 구성과 남는 차이를 바탕으로 최종 판단을 적고, 필요하면 2문장째에 추가 근거 필요 여부 또는 거절 근거 구성 가능 여부를 정리한다)

- 유사점 요약: (1~2문장으로만 작성한다. 1문장째에는 핵심 일치 구성이 어디까지 확인되는지를 적고, 필요하면 2문장째에 그 의미를 정리한다)

- 차이점: (차이점이 없으면 `없음`으로 적고, 차이점이 있으면 각 항목을 줄바꿈으로 구분하되 `[차이점 1]` 같은 소제목은 쓰지 않는다. 각 항목은 차이가 나는 청구항 제한 문언으로 시작하되 반드시 `구성 (알파벳)에 대해` 또는 `[청구항 제한]인 구성 (알파벳)에 대해`처럼 어느 구성요소의 차이인지 알파벳을 자연스럽게 포함한다. 각 항목을 적을 때는 아래의 **작성 서식 규칙**을 철저히 준수한다.
  차이점에는 기술적 차이와 문헌 근거만 작성한다. 정량 점수, 커버리지, 평가 신뢰도, 문헌 의존도, 구성 채택 여부 또는 거절 근거 구성 가능성 같은 평가·판정 내용은 작성하지 않는다.
  [작성 서식 규칙]
  ① 구성대비가 되는 내용(동일 기조 내용)은 하나의 자연스러운 한 문장의 흐름으로 완성한다. 인용발명 1에서 확인되지 않는 하위 제한이 무엇인지 매끄럽게 연결하여 "구성요소 (알파벳)과 관련하여, 인용발명 1에서는 ~한 하위 제한이 확인되지 않아 ~한 기술적 차이가 존재합니다." 형태로 쓴다.
  ② 대비되지 않는 부분, 남는 차이가 통상적 치환·확장으로 극복 가능한지 여부, 추가 근거 필요성 등은 논리 관계에 맞는 접속사를 사용하여 자연스럽게 연결한다. 반대·한계에는 `다만,`, `그러나,`, `하지만,`, 보충에는 `또한,`, `더욱이,`, `아울러,`, 결과에는 `따라서,`, `이에 따라,` 등을 사용하며, 접속사가 필요하지 않으면 생략한다. 문장을 분리하는 경우에는 줄바꿈한다.
  ③ 외국어 문헌의 괄호 안 따옴표 원문은 반드시 해당 외국어 원문 그대로 쓰고, 영어로 바꾸지 않는다.)"""

# Template B(복수 인용발명) Phase 1 전용 형식
DEFAULT_PHASE1_FORMAT_COMBO = """\
### [구성요소]

(알파벳 기호) (동일 / 실질적동일 / 일부차이 / 일부유사 / 대응안됨 중 택1) + 해당 판정 구간의 구체적 퍼센트

- 청구항 구성:

- **인용발명 대응 원문:**
  복수 문헌의 발췌를 함께 적는 경우 문헌별로 분리하고 `인용발명 N:` 꼬리표만 붙인다. 발췌문 뒤에 `~고 기재되어 있습니다` 같은 보고형 문구를 덧붙이지 않는다.
  [한국어 원문인 경우] 원문 그대로 적고 단락/본문 위치를 병기
  [외국어 원문인 경우(일본어·중국어·영어 등)] 반드시 아래 2줄 구조 사용:
    한국어 직역 또는 준직역 문장
    (단락 [실제번호] 또는 본문 실제 페이지, "원문 발췌") — 안내용 표기를 그대로 출력하지 않음
  [외국어 원문 추가 규칙] 괄호 안 따옴표의 원문 발췌는 중국어 문헌이면 중국어, 일본어 문헌이면 일본어, 영어 문헌이면 영어 그대로 적고 다른 언어로 바꾸지 않음

- 인용발명 대응 부분 요약:

- 판단 이유: (판정명이나 퍼센트를 반복 설명하지 말고, 청구항 제한과 인용발명 1의 대응 원문이 구조·기능·입력·출력·처리 조건 중 어느 점에서 대응되는지 또는 어느 핵심 제한이 빠져 대응이 불충분한지만 1~2문장으로 작성)

- 보완 검토: (인용발명 1의 판정이 `동일` 또는 `실질적동일`이면 `불필요`라고만 적는다. 인용발명 1에 미개시·약한 하위 제한이 있고 인용발명 2 이상의 직접 발췌가 제공된 경우에만, `보완 문헌 → 보완되는 하위 제한 → 원문 발췌와 위치 → 보완 후 남는 제한` 순서로 간결하게 적는다. 구성 보완 여부와 결합 동기·기술적 양립성 판단을 혼동하지 말고, 결합 가능성의 최종 판단은 `[종합분석요약]`에서 한다. 예외적 제3문헌은 표시된 주지관용 구성의 명시 근거로만 사용한다. 보완 근거가 없으면 `직접 보완 근거 없음`으로 적는다.)


[종합분석요약]

- 결론: (1~2문장으로만 작성한다. 1문장째에는 핵심 일치 구성과 남는 차이를 바탕으로 최종 판단을 적고, 필요하면 2문장째에 문헌 근거의 부족 여부 또는 거절 근거 구성 가능 여부를 정리한다)

- 유사점 요약: (1~2문장으로만 작성한다. 1문장째에는 핵심 일치 구성이 어디까지 확인되는지를 적고, 필요하면 2문장째에 그 의미를 정리한다)

- 차이점: (유사도가 `동일` 또는 `실질적동일`인 구성요소는 적지 않고 `없음`으로 처리한다. 실제 차이가 남는 구성요소만 각 항목을 줄바꿈으로 구분하되 `[차이점 1]`, `[차이점 2]` 같은 소제목은 쓰지 않는다. 각 항목을 적을 때는 아래의 **작성 서식 규칙**을 철저히 준수한다.
  차이점에는 기술적 차이와 문헌 근거만 작성한다. 정량 점수, 커버리지, 평가 신뢰도, 문헌 의존도, 구성 채택 여부 또는 거절 근거 구성 가능성 같은 평가·판정 내용은 작성하지 않는다.
  [작성 서식 규칙]
  ① 구성대비가 되는 내용(동일 기조 내용)은 하나의 자연스러운 한 문장의 흐름으로 완성한다. 인용발명 1에서 확인되지 않는 하위 제한과 인용발명 2의 개시 내용 및 출처(예: 한국어 직역 또는 준직역문 + 괄호 안 원문 및 페이지)를 매끄럽게 연결하여 "구성요소 (알파벳)과 관련하여, 인용발명 1에서는 ~가 확인되지 않으나 인용발명 2에는 ~[직역문](페이지, \"원문\")~는 구성이 기재되어 있으며 이는 ~로 보완되는 범위에 해당합니다/판단됩니다." 형태로 쓴다. (절대로 "~ 확인되지 않습니다. 인용발명 2에서는 다음 구성이 개시되어 있습니다."와 같이 문장을 분리하거나 별도 개시 안내를 나열식으로 쓰지 않는다)
  ② 보강 후 실질적인 차이가 남는 경우에는 줄바꿈하여 논리 관계에 맞는 접속사로 자연스럽게 이어 쓴다. 반대·한계에는 `다만,`, `그러나,`, `하지만,`, 보충에는 `또한,`, `더욱이,`, `아울러,`, 결과에는 `따라서,`, `이에 따라,` 등을 사용하며, 접속사가 필요하지 않으면 생략한다. 단순한 적용 대상 차이, 데이터 표현 차이 또는 통상적인 입력·출력 연결만으로 유기적 결합이 불충분하다고 판단하지 않는다. 결합 곤란성을 인정하려면 기술적 비호환성, 반대 교시, 작동 원리 변경 또는 예측하기 어려운 효과에 관한 구체적인 근거를 제시한다.
  ③ 외국어 문헌의 괄호 안 따옴표 원문은 반드시 해당 외국어 원문 그대로 쓰고, 영어로 바꾸지 않는다. `차이점 요지:`, `인용발명 2 발췌:`, `대응 이유:`, `번역:`, `발췌:` 같은 꼬리표는 쓰지 않는다. 독립항 결합형에서 `주지관용 구성 입증자료`로 표시된 예외적 제3문헌만 표시된 일반 구성의 입증자료로 제한한다. 종속항에 새로 추가된 인용발명 3 이상에는 이 제한을 적용하지 않는다. SubScore 등 내부 점수는 출력 금지)"""

DEFAULT_PHASE1_FORMAT_DEPENDENT = """\
### [추가 구성]

(A) (동일 / 실질적동일 / 일부차이 / 일부유사 / 대응안됨 중 택1) + 해당 판정 구간의 구체적 퍼센트

- 청구항 추가 구성: (`제~항에 있어서, ...` 문구를 포함하여 작성)

- **인용발명 대응 원문:**
  [한국어 원문인 경우] 원문 그대로 적고 단락/본문 위치를 병기.
  발췌문 뒤에 `~고 기재되어 있습니다` 같은 보고형 문구를 덧붙이지 않음.
  [외국어 원문인 경우(일본어·중국어·영어 등)] 한국어 직역 또는 준직역 문장 다음 줄에 원문 발췌와 단락/본문 위치를 병기.
  [외국어 원문 추가 규칙] 괄호 안 따옴표의 원문 발췌는 중국어 문헌이면 중국어, 일본어 문헌이면 일본어, 영어 문헌이면 영어 그대로 적고 다른 언어로 바꾸지 않음.

- 인용발명 대응 부분 요약:

- 판단 이유: (판정명이나 퍼센트를 반복 설명하지 말고, 추가 구성과 인용발명 대응 원문이 구조·기능·입력·출력·처리 조건 중 어느 점에서 대응되는지 또는 어느 핵심 제한이 빠져 대응이 불충분한지만 1~2문장으로 작성)


[종합분석요약]

- 결론: (추가 구성의 대응 강도와 남은 차이에 따라 거절 근거 구성 가능 여부를 중립적으로 작성하되, `구성 (A)` 같은 임시 라벨 표현은 쓰지 않음)

- 유사점 요약:

- 차이점: (임시 라벨이 있더라도 `구성 (A)`처럼 쓰지 말고 `추가 구성` 또는 해당 기술 특징 문구로 작성한다. 일반적인 기술분야 유사성만 쓰지 말고 어떤 하위 제한이 원문으로 확인되지 않는지 특정한다. 각 항목을 적을 때는 아래의 **작성 서식 규칙**을 철저히 준수한다.
  차이점에는 추가 구성의 기술적 차이와 문헌 근거만 작성한다. 정량 점수, 커버리지, 평가 신뢰도, 문헌 의존도, 추가 구성의 평가·채택 여부 또는 거절 근거 구성 가능성은 작성하지 않는다.
  [작성 서식 규칙]
  ① 구성대비가 되는 내용(동일 기조 내용)은 하나의 자연스러운 한 문장의 흐름으로 완성한다. 인용발명 1에서 확인되지 않는 하위 제한과 인용발명 2의 개시 내용 및 출처(예: 한국어 직역 또는 준직역문 + 괄호 안 원문 및 페이지)를 매끄럽게 연결하여 "추가 구성과 관련하여, 인용발명 1에서는 ~가 확인되지 않으나 인용발명 2에는 ~[직역문](페이지, \"원문\")~는 구성이 기재되어 있으며 이는 ~로 보완되는 범위에 해당합니다/판단됩니다." 형태로 쓴다. (절대로 "~ 확인되지 않습니다. 인용발명 2에서는 다음 구성이 개시되어 있습니다."와 같이 문장을 분리하거나 별도 개시 안내를 나열식으로 쓰지 않는다)
  ② 보강이 된 이후에도 남는 하위 제한, 유기적 결합 관계의 차이, 추가 근거 필요성 등(대비되지 않는 부분)은 논리 관계에 맞는 접속사를 사용하여 자연스럽게 연결한다. 반대·한계에는 `다만,`, `그러나,`, `하지만,`, 보충에는 `또한,`, `더욱이,`, `아울러,`, 결과에는 `따라서,`, `이에 따라,` 등을 사용하며, 접속사가 필요하지 않으면 생략한다. 문장을 분리하는 경우에는 줄바꿈한다.
  ③ 외국어 문헌의 괄호 안 따옴표 원문은 반드시 해당 외국어 원문 그대로 쓰고, 영어로 바꾸지 않는다. `차이점 요지:`, `인용발명 2 발췌:`, `대응 이유:`, `번역:`, `발췌:` 같은 꼬리표는 쓰지 않는다.)"""

_BASE_SYSTEM = """당신은 대한민국 특허청 심사관 수준의 특허 분석 전문가입니다.

[절대 금지 표현]
- "신규성이 없다", "신규성이 있다" → 사용 금지
- "진보성이 없다", "진보성이 있다" → 사용 금지
- "특허성이 없다/있다" → 사용 금지

[문체 기준]
- 문장은 짧고 단정하게 쓰되, 문헌 근거 없이 단정하지 말 것
- `~에 대응된다`, `~로 볼 수 있다`, `~가 기재되어 있다`, `~는 확인되지 않는다`, `따라서 ~로 판단된다` 같은 실무형 문장을 우선 사용할 것
- 단, `인용발명 대응 원문` 칸은 발췌문과 위치만 쓰고 `~고 기재되어 있습니다` 같은 보고형 종결을 붙이지 말 것
- 같은 취지를 반복하지 말고, 차이점 항목은 인용발명 1과 인용발명 2 이상의 대비 내용(동일 기조 내용)을 한 문장으로 매끄럽게 연결할 것. 후속 문장은 논리 관계에 따라 접속사를 선택하고, 반대·한계에는 `다만`, `그러나`, `하지만`, 보충에는 `또한`, `더욱이`, `아울러`, 결과에는 `따라서`, `이에 따라` 등을 사용하며, 접속사가 필요하지 않으면 생략할 것

[인용 규칙]
- 한국어 인용: 원문 그대로 기재 후 실제 단락번호 형식
- 외국어 인용(일본어·중국어·영어 등 모든 외국어): 한국어 직역 또는 준직역 문장 다음 줄에 실제 단락번호 또는 본문 페이지와 원문 발췌를 기재합니다. 안내용 표기를 그대로 출력하지 마십시오.
- 괄호 안 따옴표의 원문 발췌는 중국어 문헌이면 중국어, 일본어 문헌이면 일본어, 영어 문헌이면 영어 그대로 적고 다른 언어로 바꾸지 말 것
- 번역문은 발췌 부분의 직역 또는 준직역으로 작성하고, 요약·의역·평가를 섞지 말 것

[출력 형식]
- 마크다운으로 출력
- 독립항 Phase 1의 각 구성요소는 `### [구성요소]` 헤더 다음 줄에 `(A) 실질적동일 92%` 형식으로 시작할 것
- 종속항 Phase 1의 각 추가 구성은 `### [추가 구성]` 헤더 다음 줄에 `(A) 일부차이 87%` 형식으로 시작할 것
"""


def _reference_role_names(chain_info: Optional[Dict]) -> List[str]:
    """Return canonical display names in primary/support role order."""
    if not chain_info:
        return []
    mapping = chain_info.get("doc_name_mapping") or {}
    return [
        mapping.get(str(idx), f"인용발명 {int(idx) + 1}")
        for idx in (chain_info.get("total") or [])
    ]


def _remap_static_reference_roles(text: str, chain_info: Optional[Dict]) -> str:
    """Map static role labels (reference 1, 2, ...) to locked display names."""
    role_names = _reference_role_names(chain_info)
    if not role_names:
        return text
    by_role = {position: name for position, name in enumerate(role_names, start=1)}
    return re.sub(
        r"인용발명\s+(\d+)",
        lambda match: by_role.get(int(match.group(1)), match.group(0)),
        text,
    )


def _phase1_format_text(
    combo: bool = False,
    chain_info: Optional[Dict] = None,
) -> str:
    """Phase 1 출력 형식 템플릿 반환 (파일 우선, 없으면 기본값)."""
    if combo:
        text = load_prompt("format_phase1_combo.txt", DEFAULT_PHASE1_FORMAT_COMBO)
    else:
        text = load_prompt("format_phase1_independent.txt", DEFAULT_PHASE1_FORMAT)
    return _remap_static_reference_roles(text, chain_info)


def _build_system(
    settings: Settings,
    claim_type: str = "independent",
    chain_info: Optional[Dict] = None,
) -> str:
    """claim_type: 'independent' | 'combo' | 'dependent'"""
    system = load_prompt("system_report_base.txt", _BASE_SYSTEM)
    return _remap_static_reference_roles(system, chain_info)


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
        lines.append(f"=== 청구항 {entry['claim_number']} 분석 결과 요약 ===")
        lines.append(entry.get("report_summary", "(요약 없음)"))
        lines.append("")
    lines.append("─" * 60)
    lines.append("")
    return "\n".join(lines)


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


def _format_citation_location(match: ElementMatch, prior_docs: List[ExtractedDocument]) -> str:
    """Format a user-facing paragraph/page citation for an extracted match."""
    chunk_id = match.chunk_id
    if not chunk_id:
        return "(위치 정보 없음)"
    doc = prior_docs[match.cited_invention_index] if match.cited_invention_index < len(prior_docs) else None
    if doc and doc.document_type == "non_patent":
        anchor = chunk_id.replace("[P", "").split("-")[0] if "[P" in chunk_id else chunk_id
        return f"(본문 {anchor} 페이지)"
    return f"(단락 {_paragraph_location(chunk_id, doc)})"


def _paragraph_location(chunk_id: str, doc: Optional[ExtractedDocument]) -> str:
    """Convert an internal chunk ID to the paragraph number shown to users."""
    if doc:
        for chunk in (doc.paragraph_chunks or []) + (doc.group_chunks or []):
            if chunk.chunk_id != chunk_id:
                continue
            numbers = chunk.paragraph_range or ([chunk.paragraph_no] if chunk.paragraph_no else [])
            if numbers:
                return numbers[0] if len(numbers) == 1 else f"{numbers[0]}~{numbers[-1]}"
    match = re.search(r"(?:-P-)(\[[^]]+\])", chunk_id)
    return match.group(1) if match else chunk_id


def _format_evidence_lines(match: ElementMatch, prior_docs: List[ExtractedDocument], indent: str = "") -> list[str]:
    if not match.evidence:
        return []
    lines = [f"{indent}하위 제한별 근거:"]
    for ev in match.evidence[:5]:
        quote = (ev.quote or "").strip()
        if not quote:
            continue
        label = (ev.limitation or "근거").strip()
        chunk_id = (ev.chunk_id or "").strip()
        location = ""
        if chunk_id:
            doc = prior_docs[match.cited_invention_index] if match.cited_invention_index < len(prior_docs) else None
            if doc and doc.document_type == "non_patent":
                anchor = chunk_id.replace("[P", "").split("-")[0] if "[P" in chunk_id else "?"
                location = f" (본문 {anchor} 페이지)"
            else:
                location = f" (단락 {_paragraph_location(chunk_id, doc)})"
        lines.append(f"{indent}- {label}: {quote}{location}")
    return lines if len(lines) > 1 else []


def _combination_rationale_prompt_block(chain_info: Optional[Dict]) -> str:
    if not chain_info:
        return ""
    family_selection = chain_info.get("family_selection") or {}
    novelty_screen = chain_info.get("novelty_screen") or family_selection.get("novelty_screen") or {}
    rationale = chain_info.get("combination_rationale") or {}
    lines = []
    mapping = chain_info.get("doc_name_mapping") or {}
    total = chain_info.get("total") or []
    primary_idx = total[0] if total else family_selection.get("primary_idx")
    primary_name = mapping.get(str(primary_idx), f"인용발명 {int(primary_idx) + 1}") if primary_idx is not None else "주 인용발명"
    analysis_track = chain_info.get("analysis_track") or family_selection.get("analysis_track")
    if analysis_track == "novelty_single_reference":
        lines.append(f"판단 순서: {primary_name} 단일 문헌의 직접·완전 개시가 확인되어 신규성 경로를 우선 적용")
    else:
        lines.append("판단 순서: 단일 문헌 신규성 심사 후 차이점 기반 진보성 결합 심사")
        if novelty_screen.get("result") == "no_single_document_complete":
            lines.append("신규성 선행심사: 어느 한 문헌도 모든 필수 구성을 직접·명백하게 개시하지 않음")
        assessments = novelty_screen.get("document_assessments") or {}
        for raw_idx, assessment in assessments.items():
            try:
                assessment_idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            if total and assessment_idx not in total:
                continue
            missing = assessment.get("missing_or_indirect_labels") or []
            if not missing:
                continue
            doc_name = mapping.get(str(raw_idx), f"인용발명 {assessment_idx + 1}")
            lines.append(f"- {doc_name}의 단일문헌 잔여 구성: {', '.join(map(str, missing))}")
        if primary_idx is not None:
            lines.append(
                f"주 인용발명: {primary_name}. 독립항의 핵심 구조·작동관계와 직접 근거의 폭을 우선하여 선정"
            )
    label = rationale.get("label") or rationale.get("type")
    if label:
        lines.append(f"결합 논리 유형: {label}")
    if rationale.get("description"):
        lines.append(f"판단 취지: {rationale['description']}")
    lines.extend(f"주의: {warning}" for warning in rationale.get("warnings", []))
    validity = chain_info.get("combination_validity") or {}
    if validity:
        remaining = validity.get("remaining_uncovered_labels") or []
        critical = validity.get("critical_uncovered_labels") or []
        lines.append(
            "구성 보완 상태: "
            + ("모든 구성의 보완 근거가 있음" if validity.get("coverage_complete") else "잔여 미보완 구성 존재")
        )
        if remaining:
            lines.append("잔여 구성: " + ", ".join(map(str, remaining)))
        if critical:
            lines.append("핵심 잔여 구성: " + ", ".join(map(str, critical)))
        lines.append("결합 타당성 상태: 구성 보완과 별도로 본문 발췌에 근거한 실질 검토 필요")
    text = "\n".join(lines)
    return (
        "[신규성 선행심사 및 인용발명 선정 근거]\n"
        f"{text}\n"
        "단일 문헌 결과와 복수 문헌 결합 결과를 혼합하지 마십시오. 결합 유형은 작성 출발점일 뿐 "
        "결합 가능성의 결론이 아닙니다. 실제 발췌에 근거하여 "
        "공통 문제·개선 필요성, 적용 가능한 입출력과 처리 관계, 기본 작동원리 유지, 반대 교시, "
        "효과의 예측 가능성을 구분해 검토하고, 근거가 없으면 결합 동기가 확인되지 않는다고 명시하십시오.\n\n"
    )


def _reference_scope_prompt_block(chain_info: Optional[Dict]) -> str:
    """최종 인용 체인에 포함된 문헌만 보고서가 사용하도록 범위를 고정한다."""
    if not chain_info:
        return ""
    mapping = chain_info.get("doc_name_mapping") or {}
    total = [int(value) for value in (chain_info.get("total") or [])]
    if not total:
        return ""
    allowed = [mapping.get(str(idx), f"인용발명 {idx + 1}") for idx in total]
    lines = [
        "[보고서에서 사용 가능한 인용발명 범위]",
        "허용 문헌: " + ", ".join(allowed),
        "허용 목록에 없는 인용발명 번호, 문헌명, 발췌 또는 구성대비를 보고서에 쓰지 마십시오.",
    ]
    if len(total) == 1:
        if _has_common_knowledge_basis(chain_info):
            lines.append(
                "이 보고서는 허용된 단일 문헌과 주지관용 검토만 사용합니다. "
                "다른 문헌의 내용을 차이점 또는 보완 검토에 넣지 마십시오."
            )
        else:
            lines.append("이 보고서는 허용된 단일 문헌만 사용합니다.")
    if chain_info.get("strict_reference_scope_retry"):
        lines.append("이전 출력에 허용되지 않은 인용발명이 포함되어 재작성 중이므로 위 범위를 절대 위반하지 마십시오.")
    return "\n".join(lines) + "\n\n"


def find_unselected_reference_mentions(text: str, chain_info: Optional[Dict]) -> List[str]:
    """보고서 본문이 최종 체인 밖의 인용발명 번호를 사용했는지 반환한다."""
    if not chain_info:
        return []
    mapping = chain_info.get("doc_name_mapping") or {}
    allowed_numbers: set[int] = set()
    for idx in chain_info.get("total") or []:
        name = mapping.get(str(idx), f"인용발명 {int(idx) + 1}")
        match = re.search(r"인용발명\s*(\d+)", str(name))
        if match:
            allowed_numbers.add(int(match.group(1)))
    mentioned = {int(value) for value in re.findall(r"인용발명\s*(\d+)", str(text or ""))}
    return [f"인용발명 {number}" for number in sorted(mentioned - allowed_numbers)]


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


def sanitize_report_status_icons(text: str) -> str:
    """Preserve status icons while removing replacement-character artifacts."""
    cleaned = str(text or "")
    cleaned = re.sub(r"[\uFFFD�]+", "", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+(\r?\n)", r"\1", cleaned)
    return cleaned


def polish_phase1_summary_text(text: str) -> str:
    """Make LLM-generated summary prose less template-like without changing findings."""
    polished = str(text or "")
    polished = polished.replace(
        "다만 이 구성에 대한 직접적인 대응 관계를 보완할 보조 인용발명은 기재되어 있지 않습니다.",
        "이 구성의 직접적인 대응 관계를 보완할 인용발명도 확인되지 않습니다.",
    )
    polished = polished.replace(
        "이 구성의 직접적인 대응 관계를 보완할 보조 인용발명도 확인되지 않습니다.",
        "이 구성의 직접적인 대응 관계를 보완할 인용발명도 확인되지 않습니다.",
    )
    polished = polished.replace(
        "다만 인용발명 1로도 확인되지 않는 하위 제한은 충족하지 못합니다.",
        "인용발명 1에서 확인되지 않은 하위 제한도 여전히 남아 있습니다.",
    )
    polished = polished.replace(
        "다만 인용발명 2로도 확인되지 않는 하위 제한은 충족하지 못합니다.",
        "인용발명 2로도 확인되지 않는 하위 제한은 여전히 남아 있습니다.",
    )
    polished = re.sub(
        r"(?m)^이는\s+(.+?)\s*부재함을 의미합니다\.\n"
        r"인용발명 1에서 확인되지 않은 하위 제한도 여전히 남아 있습니다\.",
        r"따라서 \1 부재하고, 인용발명 1에서 확인되지 않은 하위 제한도 여전히 남아 있습니다.",
        polished,
    )
    polished = re.sub(
        r"(?m)^이는\s+(.+?)\s*부재함을 의미합니다\.\n"
        r"인용발명 2로도 확인되지 않는 하위 제한은 여전히 남아 있습니다\.",
        r"따라서 \1 부재하고, 인용발명 2로도 확인되지 않는 하위 제한은 여전히 남아 있습니다.",
        polished,
    )
    polished = re.sub(
        r"(?m)^다만\s+(인용발명\s+\d+에는\s+\".+?\"이라는 내용이 기재되어 있습니다)",
        r"\1",
        polished,
    )
    polished = re.sub(
        r"(?m)^따라서\s+(.+?)\s+통상의 기술자가 추가 근거 없이 용이하게 도출할 수 있는지 검토가 필요합니다\.",
        r"따라서 \1 추가 문헌 근거 없이 통상의 기술자가 용이하게 도출할 수 있는지는 별도로 검토해야 합니다.",
        polished,
    )
    return polished


def enforce_phase1_judgment_headers(
    text: str,
    matches: List[ElementMatch],
) -> str:
    """구조화된 비교 판정과 보고서의 판정명·퍼센트가 어긋나지 않게 고정한다."""
    display = {
        "동일": ("동일", 97),
        "실질적 동일": ("실질적동일", 92),
        "일부 차이": ("일부차이", 87),
        "일부 유사": ("일부유사", 82),
        "차이": ("대응안됨", 70),
        "대응 없음": ("대응안됨", 0),
    }
    result = str(text or "")
    for match in matches:
        judgment, percent = display.get(match.judgment, ("대응안됨", 0))
        label = re.escape(match.label)
        result = re.sub(
            rf"(?m)^\(\s*{label}\s*\)\s*"
            r"(?:동일|실질적\s*동일|실질적동일|일부\s*차이|일부차이|"
            r"일부\s*유사|일부유사|차이|대응\s*없음|대응안됨)"
            r"(?:\s+\d{1,3}(?:~\d{1,3})?%)?\s*$",
            f"({match.label}) {judgment} {percent}%",
            result,
        )
        if len(matches) == 1:
            result = re.sub(
                r"(?m)^(?!\s*\()\s*"
                r"(?:동일|실질적\s*동일|실질적동일|일부\s*차이|일부차이|"
                r"일부\s*유사|일부유사|차이|대응\s*없음|대응안됨)"
                r"(?:\s+\d{1,3}(?:~\d{1,3})?%)?\s*$",
                f"{judgment} {percent}%",
                result,
                count=1,
            )
        if (
            match.quote
            and match.judgment in {"동일", "실질적 동일"}
            and match.directness in {"", "direct"}
            and not match.missing_limitations
        ):
            disclosure_status = "직접 개시"
        elif match.quote or match.found:
            disclosure_status = "부분 개시"
        else:
            disclosure_status = "미개시"
        result = re.sub(
            rf"(?m)^\({label}\)\s+{re.escape(judgment)}\s+{percent}%\s*\n"
            r"(?:\s*\n)?(?:-\s*개시\s*상태\s*:\s*[^\n]*\n(?:\s*\n)?)?",
            f"({match.label}) {judgment} {percent}%\n\n- 개시 상태: {disclosure_status}\n\n",
            result,
            count=1,
        )
    return result


def _dedupe_phase1_sections(phase1_md: str) -> str:
    """Phase 1 LLM 출력에서 반복된 구성요소 섹션을 제거한다 (첫 번째 등장만 유지)."""
    sections = re.split(r'\n(?=###\s)', "\n" + phase1_md.strip())
    seen_labels: set = set()
    seen_unlabeled_component = False
    seen_summary: bool = False
    kept: list = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        header = sec.split('\n', 1)[0].strip()
        m = re.match(r'###\s+\[?(?:구성요소|추가\s*구성|전제부)(?:\s+\(((?:P|[A-J](?:-\d+)?))\))?\]?', header)
        if not m:
            m = re.search(r'^\(((?:P|[A-J](?:-\d+)?))\)', sec, re.MULTILINE)
        if m:
            label = m.group(1) if m.lastindex else None
            if not label:
                label_m = re.search(r'^\(((?:P|[A-J](?:-\d+)?))\)', sec, re.MULTILINE)
                label = label_m.group(1) if label_m else None
            if not label:
                if not seen_unlabeled_component:
                    seen_unlabeled_component = True
                    kept.append(sec)
                continue
            if label not in seen_labels:
                seen_labels.add(label)
                kept.append(sec)
        elif re.search(r'종합\s*분석\s*요약|종합분석요약', header):
            if not seen_summary:
                seen_summary = True
                kept.append(sec)
        else:
            kept.append(sec)
    return "\n\n".join(kept)


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
    chain_total = (chain_info or {}).get("total", [])
    all_no_match = all(m.judgment in NO_MATCH_LABELS for m in matches)
    if all_no_match and len(chain_total) <= 1:
        result = await _generate_rejection_impossible_report(
            claim, matches, prior_docs, settings, chain_info
        )
        yield result
        return

    novelty_track = (chain_info or {}).get("analysis_track") == "novelty_single_reference"
    all_directly_disclosed = bool(matches) and all(
        m.judgment in {"동일", "실질적 동일"}
        and m.directness in {"", "direct"}
        and not m.missing_limitations
        and bool(m.quote)
        for m in matches
    )
    if (novelty_track or all_directly_disclosed) and len(chain_total) <= 1:
        primary_idx = (chain_info.get("total") or [0])[0] if chain_info else 0
        full = await _generate_novelty_rejection_report(
            claim, matches, prior_docs, settings, primary_idx, chain_info
        )
        idx = full.find("# [Phase 2]")
        yield full[:idx].strip() if idx >= 0 else full
        return

    # 결합(Template B) 여부는 인용 체인이 결정한다(total 길이). 체인이 없을 때만
    # 구성요소 귀속(any idx>0)으로 추정한다. — 주인용발명이 doc 0이 아니어도 정확.
    if chain_info:
        needs_combination = (
            len(chain_info.get("total", [])) > 1
            or _has_common_knowledge_basis(chain_info)
        )
    else:
        needs_combination = any(m.cited_invention_index > 0 for m in matches)

    if needs_combination:
        system = _build_system(settings, "combo", chain_info)
        prompt = _make_phase1_b_prompt(claim, matches, prior_docs, chain_info, settings, prev_context,
                                       secondary_matches=secondary_matches)
    else:
        system = _build_system(settings, "independent", chain_info)
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
    support_names = [
        doc_name_mapping.get(str(idx), f"인용발명 {idx + 1}")
        if doc_name_mapping else f"인용발명 {idx + 1}"
        for idx in total_invs[1:]
    ]
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
                evidence_lines.extend(_format_evidence_lines(match, prior_docs))
                if match.motivation_quote:
                    evidence_lines.append(f"문제·개선효과 직접 근거: {match.motivation_quote}")
                if match.combination_risk in {"contrary_teaching", "principle_change"}:
                    evidence_lines.append(
                        f"명시적 결합 위험: {match.combination_risk} - "
                        f"{match.combination_risk_reason or '구체적 근거 확인 필요'}"
                    )
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
    comp_text = _format_component_comparison(
        matches,
        prior_docs,
        primary_idx=primary_idx,
        doc_name_mapping=doc_name_mapping,
        combo=combo,
        secondary_matches=secondary_matches,
        total_invs=total_invs,
    )
    combination_rationale_block = (
        _reference_scope_prompt_block(chain_info)
        + _combination_rationale_prompt_block(chain_info)
    )
    conventional_policy_block = _conventional_policy_prompt_block(chain_info)
    context_block = _build_context_block(prev_context)
    fmt = _phase1_format_text(combo=combo, chain_info=chain_info)
    if combo and len(support_names) > 1:
        combo_hint = f"각 구성요소의 판정과 판단 이유는 {inv1_name} 기준으로 작성합니다. {inv1_name}의 미개시·약한 하위 제한은 같은 구성요소의 `보완 검토`에 {', '.join(support_names)}의 직접 발췌, 보완 범위, 잔여 제한을 표시하고, 결합 동기·기술적 양립성의 최종 판단은 종합 분석 요약에서 별도로 작성합니다. 예외적 주지관용 입증 문헌은 표시된 주지관용 구성의 명시 근거로만 제한합니다."
    elif combo and support_names:
        combo_hint = f"각 구성요소의 판정과 판단 이유는 {inv1_name} 기준으로 작성합니다. {inv1_name}의 미개시·약한 하위 제한은 같은 구성요소의 `보완 검토`에 {support_names[0]}의 직접 발췌, 보완 범위, 잔여 제한을 표시하고, 결합 동기·기술적 양립성의 최종 판단은 종합 분석 요약에서 별도로 작성합니다."
    elif combo:
        combo_hint = f"각 구성요소의 판정과 판단 이유는 {inv1_name} 기준으로 작성합니다. 다른 문헌을 추가하지 말고 허용된 주지관용 근거만 별도로 검토하며, 최종 판단은 종합 분석 요약에서 작성합니다."
    else:
        combo_hint = f"각 구성요소는 {inv1_name} 기준으로 작성합니다."

    return render_prompt(
        "prompt_phase1_main.txt",
        context_block=context_block,
        claim_number=str(claim.claim_number),
        claim_text=claim.text,
        elements_text=elements_text,
        primary_inv_name=inv1_name,
        inv1_filename=inv1_filename,
        comp_text=comp_text,
        inv2_block=inv2_block,
        combination_rationale_block=combination_rationale_block,
        conventional_policy_block=conventional_policy_block,
        fmt=fmt,
        combo_hint=combo_hint,
    )


def _has_common_knowledge_basis(chain_info: Optional[Dict]) -> bool:
    if not chain_info:
        return False
    return bool(
        chain_info.get("common_general_knowledge")
        or chain_info.get("conventional_support")
    )


def format_rejection_basis_header(
    inv1_name: str,
    inv2_name: str = "",
    inv3_name: str = "",
    is_combo: bool = False,
    chain_info: Optional[Dict] = None,
    is_novelty: bool = False,
) -> str:
    has_common_knowledge = _has_common_knowledge_basis(chain_info)
    primary_name = (inv1_name or "인용발명 1").strip()
    secondary_name = (inv2_name or "").strip()
    if is_combo and inv2_name:
        if has_common_knowledge:
            return f"[{primary_name}과 {secondary_name}의 결합 및 주지관용(진보성)]"
        if inv3_name:
            return f"[{primary_name}, {secondary_name} 및 {inv3_name.strip()}의 결합(진보성)]"
        return f"[{primary_name}과 {secondary_name}의 결합(진보성)]"
    if has_common_knowledge:
        return f"[{primary_name} + 주지관용(진보성)]"
    if is_novelty:
        return f"[{primary_name} 단독(신규성)]"
    return f"[{primary_name} 단독(진보성 검토)]"


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

# ---------------------------------------------------------------------------
# 신규성 부정 (차이점 없음)
# ---------------------------------------------------------------------------

async def _generate_novelty_rejection_report(
    claim: ParsedClaim,
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
    primary_idx: int = 0,
    chain_info: Optional[Dict] = None,
) -> str:
    inv1_doc = prior_docs[primary_idx] if primary_idx < len(prior_docs) else (prior_docs[0] if prior_docs else None)
    mapping = (chain_info or {}).get("doc_name_mapping") or {}
    primary_inv_name = mapping.get(str(primary_idx), f"인용발명 {primary_idx + 1}")
    elements_text = _format_elements(claim)
    comp_text = _format_component_comparison(
        matches,
        prior_docs,
        primary_idx=primary_idx,
        doc_name_mapping=mapping,
    )

    prompt = render_prompt(
        "prompt_novelty_rejection.txt",
        claim_number=str(claim.claim_number),
        claim_text=claim.text,
        elements_text=elements_text,
        primary_inv_name=primary_inv_name,
        inv1_filename=inv1_doc.filename if inv1_doc else "인용발명 1",
        comp_text=comp_text,
    )
    result = await call_ai(
        prompt,
        _build_system(settings, "independent", chain_info),
        settings,
        agent="report",
    )
    return _normalize_report_markdown(result)


# ---------------------------------------------------------------------------
# 진보성 부정 불가
# ---------------------------------------------------------------------------

async def _generate_rejection_impossible_report(
    claim: ParsedClaim,
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
    chain_info: Optional[Dict] = None,
) -> str:
    claim_type = "independent" if claim.claim_type == "independent" else "dependent"
    parent_prefix = f"제{claim.parent_claim}항에 있어서 " if claim.parent_claim else ""
    mapping = (chain_info or {}).get("doc_name_mapping") or {}
    evidence_lines = []
    for match in matches:
        doc_name = mapping.get(
            str(match.cited_invention_index),
            f"인용발명 {match.cited_invention_index + 1}",
        )
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
        reference_scope_block=_reference_scope_prompt_block(chain_info),
        comp_text="\n".join(evidence_lines).strip() or "(확인된 대응 근거 없음)",
    )
    result = await call_ai(
        prompt,
        _build_system(settings, claim_type, chain_info),
        settings,
        agent="report",
    )
    return _normalize_report_markdown(result)


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


def _dependent_decision_basis(chain_info: Optional[Dict]) -> str:
    trace = (chain_info or {}).get("decision_trace") or {}
    basis = trace.get("selection_basis", "")
    labels = trace.get("additional_labels") or []
    remaining = trace.get("remaining_uncovered_labels") or []
    basis_text = {
        "covered_by_inherited": "상속 문헌이 추가 한정까지 대응하여 새 문헌을 추가하지 않음",
        "single_document_full_gap_filler": "새 문헌 하나가 상속 문헌의 잔여 추가 한정을 모두 보완함",
        "single_document_soft_gap_improver": "새 문헌 하나가 상속 문헌의 약한 대응을 명시적 대응으로 보강함",
        "partial_support_only": "새 문헌이 일부 추가 한정만 보완하여 잔여 차이가 남음",
        "no_candidate": "잔여 추가 한정을 하나의 새 문헌이 보완하지 못함",
        "comparison_evidence_missing": "종속항 자체의 문헌 대비 근거가 없어 새 문헌을 추가하지 않음",
    }.get(basis, "구조화된 추가 한정 대비 결과에 따름")
    return (
        f"{basis_text}; 평가한 추가 한정={', '.join(map(str, labels)) or '없음'}; "
        f"잔여 미커버={', '.join(map(str, remaining)) or '없음'}"
    )


def _meaningful_dependent_matches(
    matches: Optional[List[ElementMatch]],
    secondary_matches: Optional[List[ElementMatch]] = None,
) -> List[ElementMatch]:
    items: List[ElementMatch] = []
    seen: set[tuple[str, int, str, str]] = set()
    for match in (matches or []) + (secondary_matches or []):
        if match.judgment in NO_MATCH_LABELS:
            continue
        key = (
            match.label,
            int(match.cited_invention_index),
            match.quote or "",
            match.similarity_reason or "",
        )
        if key in seen:
            continue
        seen.add(key)
        items.append(match)
    return items


def _dependent_display_chain_info(
    chain_info: Optional[Dict],
    matches: Optional[List[ElementMatch]] = None,
    secondary_matches: Optional[List[ElementMatch]] = None,
) -> Dict:
    base = dict(chain_info or {})
    inherited = list(base.get("inherited", []))
    added = list(base.get("added", []))
    total = list(base.get("total", inherited))
    parent_available = bool(base.get("parent_available", True))
    evidence_docs: List[int] = []
    for match in _meaningful_dependent_matches(matches, secondary_matches):
        doc_idx = int(match.cited_invention_index)
        if doc_idx not in evidence_docs:
            evidence_docs.append(doc_idx)

    if total:
        base["reporting_docs"] = evidence_docs or total
        return base

    if evidence_docs:
        reporting_added = evidence_docs[:]
        reporting_inherited = inherited if parent_available else []
        reporting_total = reporting_inherited + [
            idx for idx in reporting_added if idx not in reporting_inherited
        ]
        base["inherited"] = reporting_inherited
        base["added"] = reporting_added
        base["total"] = reporting_total
        base["reporting_docs"] = reporting_total
        return base

    base["reporting_docs"] = total
    return base


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
    # 부정불가 자동 전환 처리
    all_no_match = all(m.judgment in NO_MATCH_LABELS for m in matches)
    if all_no_match:
        return await _generate_rejection_impossible_report(
            claim, matches, prior_docs, settings, chain_info
        )

    display_chain = _dependent_display_chain_info(
        chain_info,
        matches=matches,
        secondary_matches=secondary_matches,
    )
    parent_num = claim.parent_claim or 1
    inherited_invs = display_chain.get("inherited", [0]) if display_chain else [0]
    added_invs = display_chain.get("added", []) if display_chain else []
    total_invs = display_chain.get("total", inherited_invs) if display_chain else inherited_invs
    mapping = display_chain.get("doc_name_mapping") if display_chain else None
    parent_available, parent_context_status = _dependent_parent_context_status(claim, display_chain)

    inherited_str = format_inv_list(inherited_invs, mapping) if inherited_invs else ("인용발명 1" if parent_available else "없음 (부모항 없음)")
    # added가 없을 때 전체 체인 기준 설명을 생성하되 부모항 상속 인용발명도 포함
    current_inv_str = format_inv_list(added_invs, mapping) if added_invs else (inherited_str if parent_available else (format_inv_list(total_invs, mapping) if total_invs else "추가 문헌 없음"))
    final_str = format_inv_list(total_invs, mapping) if total_invs else ("인용발명 1" if parent_available else "문헌 없음")
    added_inv_str = format_inv_list(added_invs, mapping) if added_invs else "없음"
    if display_chain and display_chain.get("coverage_complete") is False:
        if added_invs:
            coverage_status = "부분 대응 — 새 인용발명이 추가 구성의 일부 기술적 특징을 보완하지만 차이가 남아 있습니다."
        else:
            coverage_status = "일부 추가 구성 미대응"
    else:
        coverage_status = "추가 구성이 모두 대응되었습니다."

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
        decision_basis=_dependent_decision_basis(display_chain),
        added_doc_filename=added_doc.filename if added_doc else current_inv_str,
        quotes_text=quotes_text if quotes_text else "(대응 구성 확인 필요 — 인용발명 원문 기반으로 직접 판단)",
        phase1_dep_fmt=phase1_dep_fmt,
    )
    result = await call_ai(
        prompt,
        _build_system(settings, "dependent", display_chain),
        settings,
        agent="report",
    )
    return _normalize_report_markdown(result)


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
        if (chain_info or {}).get("coverage_complete") is False:
            if added_invs:
                coverage_status = "부분 대응 — 새 인용발명이 추가 구성의 일부 기술적 특징을 보완하지만 차이가 남아 있습니다."
            else:
                coverage_status = "대응 불충분 — 새 인용발명 1개만으로 커버되지 않은 추가 구성이 있습니다."
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
            f"- 종속항 판단 근거: {_dependent_decision_basis(chain_info)}\n\n"
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
    result = await call_ai(prompt, _build_system(settings, "dependent"), settings, agent="report")
    return _normalize_report_markdown(result)


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
    """독립항의 거절근거로 채택되지 않은 인용발명을 관련도 A 섹션으로 조립한다.

    chain_info.total(보고서에 실제 사용된 인용발명)에 없는 인용발명을 대상으로,
    comparisons_{idx}.json 의 독립 판정을 읽어 문헌별 고정 형식으로 요약한다.
    구성 라벨은 내부 선택에만 사용하고 출력하지 않는다(LLM 호출 없음).
    """
    from backend.services.citation_extractor import load_comparisons

    if claim.claim_type != "independent" or not chain_info or len(prior_docs) <= 1:
        return ""

    used = set(chain_info.get("total", []))
    doc_name_mapping = chain_info.get("doc_name_mapping", {})
    claim_key = str(claim.claim_number)

    label_to_text = {
        str(getattr(elem, "label", "")).strip(): getattr(elem, "text", "").strip()
        for elem in (claim.elements or [])
        if getattr(elem, "label", None)
    }

    def _summarize_claim_text(text: str) -> str:
        clean = re.sub(r"\s+", " ", (text or "").strip())
        clean = re.sub(r"^(?:상기\s*)?", "", clean)
        clean = re.sub(r",?\s*무선전력 수신장치\.?$", "", clean)
        clean = clean.rstrip(" ,.;")
        if len(clean) > 90:
            clean = clean[:87].rstrip(" ,.;") + "..."
        return clean

    def _item_reason(item: Dict, inv_name: str = "", doc_idx: Optional[int] = None) -> str:
        reason = (item.get("similarity_reason") or item.get("판단_이유") or "").strip()
        if inv_name and doc_idx is not None:
            for raw_name in (f"인용발명 {doc_idx}", f"인용발명 {doc_idx + 1}"):
                reason = reason.replace(raw_name, inv_name)
        return re.sub(r"\s+", " ", reason)

    def _excerpt(item: Dict, prefix: str = "") -> str:
        quote = re.sub(
            r"\s+",
            " ",
            (item.get(f"{prefix}quote") or "").strip(),
        )
        translation = re.sub(
            r"\s+",
            " ",
            (item.get(f"{prefix}quote_translation") or "").strip(),
        )
        chunk_id = (item.get("chunk_id") or "").strip()
        item_doc_idx = int(item.get("doc_index", item.get("cited_invention_index", 0)) or 0)
        doc = prior_docs[item_doc_idx] if item_doc_idx < len(prior_docs) else None
        if translation:
            location = f"단락 {_paragraph_location(chunk_id, doc)}" if chunk_id else "단락 번호 확인 필요"
            if quote:
                return f'{translation} ({location}, "{quote}")'
            return f"{translation} ({location})"
        if quote and chunk_id:
            return f"{quote} (단락 {_paragraph_location(chunk_id, doc)})"
        if quote:
            return quote
        if prefix:
            return ""
        evidence = item.get("evidence") or []
        if isinstance(evidence, list):
            for ev in evidence:
                if not isinstance(ev, dict):
                    continue
                ev_quote = re.sub(
                    r"\s+",
                    " ",
                    (ev.get("quote") or "").strip(),
                )
                ev_translation = re.sub(
                    r"\s+",
                    " ",
                    (ev.get("quote_translation") or "").strip(),
                )
                ev_chunk_id = (ev.get("chunk_id") or "").strip()
                ev_doc = prior_docs[item_doc_idx] if item_doc_idx < len(prior_docs) else None
                if ev_translation:
                    location = f"단락 {_paragraph_location(ev_chunk_id, ev_doc)}" if ev_chunk_id else "단락 번호 확인 필요"
                    if ev_quote:
                        return f'{ev_translation} ({location}, "{ev_quote}")'
                    return f"{ev_translation} ({location})"
                if ev_quote and ev_chunk_id:
                    return f"{ev_quote} (단락 {_paragraph_location(ev_chunk_id, ev_doc)})"
                if ev_quote:
                    return ev_quote
        return ""

    judgment_rank = {
        "대응 없음": 0,
        "차이": 1,
        "일부 유사": 2,
        "일부 차이": 3,
        "실질적 동일": 4,
        "실질적동일": 4,
        "동일": 5,
    }

    def _rank(item: Dict) -> int:
        return judgment_rank.get(_normalize_match_judgment(item.get("judgment") or ""), 0)

    def _purpose_effect_line(items: List[Dict], inv_name: str, doc_idx: int) -> str:
        candidates = [
            item for item in items
            if (item.get("purpose_effect_similarity") or "").strip()
        ]
        if candidates:
            return re.sub(
                r"\s+",
                " ",
                (max(candidates, key=_rank).get("purpose_effect_similarity") or "").strip(),
            )
        best = max(items, key=_rank, default={})
        reason = _item_reason(best, inv_name, doc_idx)
        return reason or "독립항과 기술적 목적 또는 기대 효과가 일부 관련됩니다."

    def _closest_line(items: List[Dict], inv_name: str, doc_idx: int) -> str:
        candidates = [item for item in items if _excerpt(item)]
        best = max(candidates or items, key=_rank, default={})
        reason = _item_reason(best, inv_name, doc_idx)
        excerpt = _excerpt(best)
        body = reason or "독립항과 가장 가까운 기술적 대응 내용입니다."
        return f"{body} ({excerpt})" if excerpt else body

    def _difference_line(items: List[Dict]) -> str:
        different = [
            item for item in items
            if _normalize_match_judgment(item.get("judgment") or "") not in {"동일", "실질적 동일"}
        ]
        claim_parts = []
        for item in sorted(different, key=_rank):
            claim_text = _summarize_claim_text(label_to_text.get((item.get("label") or "").strip(), ""))
            if claim_text and claim_text not in claim_parts:
                claim_parts.append(claim_text)
            if len(claim_parts) == 2:
                break
        if claim_parts:
            return f"독립항은 {' 및 '.join(claim_parts)} 부분에서 이 인용발명과 차이가 있습니다."
        return "주된 거절근거로 선택되지 않았으며, 독립항과의 명확한 구성 차이는 확인되지 않았습니다."

    def _display_order(doc_idx: int) -> tuple[int, int]:
        inv_name = doc_name_mapping.get(str(doc_idx), f"인용발명 {doc_idx + 1}")
        match = re.search(r"인용발명\s*(\d+)", inv_name)
        if match:
            return (int(match.group(1)), doc_idx)
        return (doc_idx + 1, doc_idx)

    blocks = []
    for doc_idx in sorted(range(len(prior_docs)), key=_display_order):
        if doc_idx in used:
            continue
        cache = load_comparisons(job_dir, doc_idx)
        items = cache.get(claim_key) if cache else None
        comparison_missing = not items
        items = items or []
        inv_name = doc_name_mapping.get(str(doc_idx), f"인용발명 {doc_idx + 1}")
        similar_items = [
            it for it in items
            if _normalize_match_judgment(it.get("judgment") or "") not in NO_MATCH_LABELS
        ]
        block_lines = [
            f"**{inv_name}** ({prior_docs[doc_idx].filename})",
        ]
        if comparison_missing:
            block_lines.extend([
                "- 목적·효과 관련 유사점: 대응 결과 미저장 — 확인 필요",
                "- 가장 가까운 대응 내용: 대응 결과 미저장 — 확인 필요",
                "- 독립항과의 차이점: 대응 결과 미저장 — 확인 필요",
            ])
        elif similar_items:
            block_lines.extend([
                f"- 목적·효과 관련 유사점: {_purpose_effect_line(similar_items, inv_name, doc_idx)}",
                f"- 가장 가까운 대응 내용: {_closest_line(similar_items, inv_name, doc_idx)}",
                f"- 독립항과의 차이점: {_difference_line(items)}",
            ])
        else:
            block_lines.extend([
                "- 목적·효과 관련 유사점: 직접 확인되는 유사점이 없습니다.",
                "- 가장 가까운 대응 내용: 직접 발췌할 수 있는 대응 내용이 없습니다.",
                f"- 독립항과의 차이점: {_difference_line(items)}",
            ])

        blocks.append("\n".join(block_lines))

    if not blocks:
        return ""
    intro = (
        "아래 문헌은 독립항의 신규성 또는 진보성 거절근거로 사용되지 않은 인용발명입니다. "
        "문헌별로 목적·효과 관련 유사점, 가장 가까운 대응 발췌 및 독립항과의 차이점을 정리합니다."
    )
    return intro + "\n\n## 관련도 A 인용발명\n\n" + "\n\n".join(blocks)


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
        # 1차: 어미 패턴 정확 매칭 ("제N항에 있어서" / "청구항 N에 있어서" / "제N항의")
        m = re.search(
            r"(?:제\s*(\d+)\s*항|청구항\s*(\d+))"
            r"(?:\s*(?:내지|또는|및)\s*(?:제\s*\d+\s*항|청구항\s*\d+|\d+))?"
            r"(?:\s*중\s*어느\s*한\s*항)?"
            r"(?:에\s*있어서|의)",
            clean_text,
        )
        if m:
            candidate = int(m.group(1) or m.group(2))
            if candidate != claim_number:
                inferred_parent = candidate

    if inferred_parent is None:
        # 2차: 어두 오타·생략 대비 — 앞부분 150자 안에 "제N항"/"청구항 N" + N < 현재 번호
        # finditer로 전체 후보를 보는 이유: "제10항. 제1항에 있이서…"처럼 자기 번호가
        # 맨 앞에 오면 re.search 첫 결과가 자기 자신(N == 현재)이라 조건에 걸린다.
        for m2 in re.finditer(r"(?:제\s*(\d+)\s*항|청구항\s*(\d+))", clean_text[:150]):
            candidate = int(m2.group(1) or m2.group(2))
            if candidate < claim_number:
                inferred_parent = candidate
                break

    if inferred_parent is None:
        # 3차: 후미형 종속항 — 어두가 독립항처럼 생겼지만 본문/후미에
        # "제X항 [내지/또는/및 제Y항]을 포함하는/인용하는/에 따른" 형태로 참조하는 경우.
        # 독립항이 다른 청구항을 단순 언급하는 것과 구분하기 위해 의존 동사구를 필수로 요구한다.
        m3 = re.search(
            r"(?:제\s*(\d+)\s*항|청구항\s*(\d+))"
            r"(?:\s*(?:내지|또는|및)\s*(?:제\s*\d+\s*항|청구항\s*\d+|\d+))?"
            r"\s*(?:(?:을|를)\s*(?:포함|인용|청구|참조|기재)\s*(?:하는|하여|하고)?|에\s*따른|에\s*의한)",
            clean_text,
        )
        if m3:
            candidate = int(m3.group(1) or m3.group(2))
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
            imp = _guess_component_importance(seg, "5" if i == 0 else ("3" if i < 3 else "2"))
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
            imp = _guess_component_importance(content, "5" if i == 0 else ("3" if i < 3 else "2"))
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
        imp = _guess_component_importance(clean, "5" if label_idx == 0 else ("3" if label_idx < 3 else "2"))
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


def _guess_component_importance(text: str, default: str) -> str:
    """Apply conservative local importance hints before optional LLM reparsing."""
    source = text or ""
    if _SELECTION_OR_RE.search(source) and _SELECTION_CONDITION_RE.search(source):
        return "5"
    return default


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
7. 각 구성요소의 importance는 아래 기준으로 "5", "3", "2" 중 하나를 선택
   - "5": 핵심 ★★★. 발명의 기술적 과제 해결수단, 신규·진보성 판단의 중심이 되는 구조/처리/조건/상호작용
   - "3": 보조 ★★☆. 핵심 구성을 보조하거나 입출력·처리 흐름·동작 조건을 구체화하지만 단독 핵심은 아닌 구성
   - "2": 관용 ★☆☆. 프로세서, 메모리, 통신부, 디스플레이, 저장부, 일반 센서처럼 통상적으로 부가되는 일반 구성
8. 단순히 청구항 앞쪽에 나온다는 이유만으로 "5"를 주지 말고, 기술적 의미와 과제 해결 기여도를 기준으로 판단
9. `A 또는 B`, `A 및/또는 B`, `A와 B 중 적어도 하나` 같은 선택식 표현에 `~에 따라`, `~를 고려하여`, `~에 기초하여`, `~를 판단하여`, `~인 경우`, `~에 대응하여`, `선택`, `전환`, `분기`, `종류`, `유형`, `모드` 등 조건·기준·분기 문언이 결합되어 있고, 그 조건이 선택지 중 어느 처리/구조/동작을 수행할지 좌우하면 신규·진보성 판단의 중심이 될 수 있으므로 importance를 원칙적으로 "5"로 둠
10. 다만 선택식 표현이 단순 나열, 대체 가능한 출력 형식, 적용 대상 예시, 부가 옵션에 불과하고 조건·기준 문언이 선택지 수행을 좌우하지 않으면 그 기술적 기여도에 따라 "3" 또는 "2"로 둠
11. 서브구성은 부모 구성의 핵심 제한을 구체화하면 부모와 같은 중요도를 줄 수 있고, 단순 참조·부가 설명이면 "3"으로 둠

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
    return "\n".join(
        f"({'전제부 P' if e.label == 'P' else e.label}) {e.text}"
        for e in claim.elements
    )


def _format_component_comparison(
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    primary_idx: int = 0,
    doc_name_mapping: Optional[Dict[str, str]] = None,
    combo: bool = False,
    secondary_matches: Optional[List[ElementMatch]] = None,
    total_invs: Optional[List[int]] = None,
) -> str:
    def _doc_name(doc_idx: int) -> str:
        if doc_name_mapping:
            return doc_name_mapping.get(str(doc_idx), f"인용발명 {doc_idx + 1}")
        return f"인용발명 {doc_idx + 1}"

    def _format_single(match: Optional[ElementMatch], fallback_doc_idx: int) -> list[str]:
        doc_name = _doc_name(fallback_doc_idx)
        if not match or match.judgment in NO_MATCH_LABELS or not match.quote:
            return [f"- {doc_name}: 대응 없음", "  (해당 인용발명에서 구성 확인 불가)"]

        lines = [f"- {doc_name}: {match.judgment}"]
        lines.append(f"  {match.quote}")
        if match.chunk_id:
            lines.append(f"  {_format_citation_location(match, prior_docs)}")
        lines.extend(_format_evidence_lines(match, prior_docs, indent="  "))
        if match.directness:
            lines.append(f"  직접성: {match.directness}")
        if match.missing_limitations:
            lines.append(f"  누락 제한요소: {', '.join(match.missing_limitations)}")
        if match.similarity_reason:
            lines.append(f"  판단 이유: {match.similarity_reason}")
        return lines

    if combo:
        total_invs = total_invs or [primary_idx]
        primary_by_label = {}
        for m in matches:
            if m.cited_invention_index != primary_idx:
                continue
            prev = primary_by_label.get(m.label)
            if prev is None or _JUDGMENT_ORDER.get(m.judgment, 0) > _JUDGMENT_ORDER.get(prev.judgment, 0):
                primary_by_label[m.label] = m

        ordered_labels: list[str] = []
        for match in matches:
            if match.label not in ordered_labels:
                ordered_labels.append(match.label)

        lines = []
        for label in ordered_labels:
            primary_match = primary_by_label.get(label)
            if primary_match is None:
                primary_match = ElementMatch(
                    label=label,
                    found=False,
                    quote="",
                    chunk_id="",
                    judgment="대응 없음",
                    cited_invention_index=primary_idx,
                    similarity_reason="",
                    directness="absent",
                    missing_limitations=[],
                )
            lines.extend(_format_single(primary_match, primary_idx))
            lines.append("")
        return "\n".join(lines)

    lines = []
    for m in matches:
        doc_name = _doc_name(m.cited_invention_index)
        lines.append(f"({m.label}) {m.judgment} ({doc_name})")
        if m.quote:
            lines.append(m.quote)
        if m.chunk_id:
            lines.append(_format_citation_location(m, prior_docs))
        lines.extend(_format_evidence_lines(m, prior_docs))
        if m.directness:
            lines.append(f"직접성: {m.directness}")
        if m.missing_limitations:
            lines.append(f"누락 제한요소: {', '.join(m.missing_limitations)}")
        if m.similarity_reason:
            lines.append(f"판단 이유: {m.similarity_reason}")
        lines.append("")
    return "\n".join(lines)
