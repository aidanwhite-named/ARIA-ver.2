"""미커버 구성을 위한 빠른 적응형 선행기술 검색."""
from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable, Dict, List

from backend.models.schemas import Settings
from backend.services.ai_engine import call_ai


SearchRunner = Callable[..., Awaitable[str]]

_URL_RE = re.compile(r"https?://[^\s)<>\]]+", re.IGNORECASE)
_DOC_ID_RE = re.compile(
    r"\b(?:US|WO|EP|KR|JP|CN)\s*[-/]?\s*\d{4,}[A-Z]\d?\b|"
    r"\barXiv\s*:?\s*\d{4}\.\d{4,5}(?:v\d+)?\b|"
    r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
    re.IGNORECASE,
)

SEARCH_SYSTEM = """당신은 특허 선행기술 검색 보조자입니다.
반드시 실제 웹 검색을 수행하고 실제로 확인한 URL만 사용하십시오.
검색 결과 페이지, 블로그, 뉴스, 위키보다 특허 원문, 논문 원문, 공식 출판 페이지를 우선하십시오.
제목만 비슷한 후보를 직접 대응 문헌으로 단정하지 마십시오.
미커버 구성은 단순 키워드가 아니라 청구항에서 빠진 기술적 역할(gap role)로 해석하십시오.
보조 후보는 같은 기술분야 문헌이 아니라 해당 gap role을 직접 메우는 원문 근거가 있는 문헌이어야 합니다.
응답은 간결한 한국어 Markdown으로 작성하십시오."""


def build_parallel_search_prompts(
    claim_text: str,
    targets: List[Dict],
    used_documents: List[Dict],
    additional_query: str = "",
) -> List[Dict[str, str]]:
    common = f"""[청구항]
{claim_text}

[검색할 미커버 구성]
{targets}

[이미 사용된 문헌]
{used_documents}

[추가 조건]
{additional_query or "없음"}

먼저 각 미커버 구성을 다음 방식으로 검색 역할로 바꾸십시오.
1. gap role을 지정합니다. 예: 입력/소스 데이터 누락, 산출값/파라미터 누락, 측정/감지 누락, 결합/fusion 로직 누락, 선택/분기 기준 누락, 제어 대상 누락, 변환/출력 생성 누락, 구조적 호환성 누락, 적용 분야 호환성 누락.
2. 구현어와 기능어를 각각 도출합니다. 구현어는 구체 구조, 데이터 유형, 모듈, 재료, 신호, 파라미터, 처리 단계입니다. 기능어는 추상 동작, 관계, 판단 기준, 산출 결과, 변환, 제어 대상, 출력, 기술적 효과입니다.
3. 특정 예시 단어를 고정 키워드로 쓰지 말고, 실제 청구항 문언과 이미 사용된 문헌에서 빠진 제한으로부터 검색어를 도출합니다.
4. 복합 구성인 경우에는 빠진 산출값 자체, 그 산출값의 입력/소스, 결합·선택·제어 규칙, 주 인용발명과의 장치/분야 호환성 순서로 후보를 평가합니다.

각 후보에 문헌명, 공개번호/논문 식별자, 확인 가능한 공개일, 직접 URL,
대응 구성 라벨, gap role 커버리지, 대응 이유, 남는 차이를 포함하십시오. 후보는 최대 5개로 제한하십시오."""
    return [
        {
            "axis": "functional",
            "prompt": common + """

[검색 축: 범용 기능]
청구항식 표현과 서수는 제거하고 미커버 구성을 입력, 조건, 처리, 비교, 위치특정,
교정, 출력의 기능 축으로 일반화하십시오. 짧은 동의어 검색식을 여러 개 사용하십시오.""",
        },
        {
            "axis": "patent",
            "prompt": common + """

[검색 축: 특허]
Google Patents, USPTO, KIPRIS, WIPO 등 특허 원문이 나타나도록 영문 중심의
기능·구조 동의어를 사용하십시오. 동일 특허 패밀리는 하나의 후보로 정리하십시오.""",
        },
        {
            "axis": "scholarly",
            "prompt": common + """

[검색 축: 학술논문]
arXiv, CVF, IEEE, ACM, 학회·저널 및 공식 논문 페이지를 중심으로 검색하십시오.
연구 분야의 모델명보다 미커버 처리 메커니즘과 입출력 관계를 우선 검색하십시오.""",
        },
    ]


def search_result_stats(results: List[str]) -> Dict[str, int]:
    joined = "\n".join(results or [])
    urls = {_url.rstrip(".,;") for _url in _URL_RE.findall(joined)}
    identifiers = {_id.replace(" ", "").upper() for _id in _DOC_ID_RE.findall(joined)}
    return {"url_count": len(urls), "identifier_count": len(identifiers)}


def needs_expansion(results: List[str]) -> bool:
    stats = search_result_stats(results)
    return stats["url_count"] < 3 or stats["identifier_count"] < 2


def build_expansion_prompt(
    claim_text: str,
    targets: List[Dict],
    prior_results: List[str],
    additional_query: str = "",
) -> str:
    return f"""[청구항]
{claim_text}

[미커버 구성]
{targets}

[초기 검색 결과]
{chr(10).join(prior_results)[:12000]}

[추가 조건]
{additional_query or "없음"}

초기 결과의 직접 URL 또는 문헌 식별자가 부족합니다. 같은 검색식을 반복하지 말고:
1. 미커버 구성의 gap role을 다시 분류하고,
2. 구현어와 기능어를 서로 다른 조합으로 바꾸며,
3. 기능을 상위개념, 하위 구현, 입력·출력 관계, 결합·선택·제어 관계로 각각 표현하고,
4. 특허 원문 또는 학술 원문 도메인을 대상으로,
5. 기존 결과에 없는 후보만 최대 5개 검색하십시오.
실제 확인한 직접 URL이 없는 후보는 제안하지 마십시오."""


def build_synthesis_prompt(
    claim_text: str,
    targets: List[Dict],
    results: List[Dict[str, str]],
) -> str:
    blocks = "\n\n".join(
        f"### {item['axis']} 검색\n{item['result']}" for item in results
    )
    return f"""[청구항]
{claim_text}

[미커버 구성]
{targets}

[병렬 검색 결과]
{blocks}

웹 검색을 새로 수행하지 말고 위 결과만 정리하십시오.
- URL 또는 식별자가 같은 후보와 특허 패밀리 중복을 제거하십시오.
- 직접 URL이 없거나 검색요약·블로그·뉴스뿐인 후보는 제외하십시오.
- 미커버 구성의 gap role에 직접 대응하는 후보를 먼저 배치하십시오.
- 같은 기술분야라는 이유만으로 후보를 올리지 말고, 빠진 산출값·입력·결합규칙·제어/출력 관계 중 무엇을 원문 근거로 메우는지 표시하십시오.
- 공개일이 불명확하면 불명확하다고 표시하십시오.
- 원문 전체를 정밀 검증한 것처럼 표현하지 마십시오.

## 검색 범위
- 실행한 검색 축과 확장 검색 여부
## 후보 문헌
후보 하나당 아래 형식의 `###` 블록을 정확히 하나씩 사용하십시오. 후보 분류용 `###` 또는 `####` 중간 제목은 만들지 마십시오.
### 후보 1: 문헌명
- **문헌번호(이름)**: 특허는 특허 공개번호, 논문은 논문의 정식 제목
- **공개일**: 확인 가능한 공개일
- **직접 링크**: https://example.com/full/path
- **대응 구성**: 구성 라벨
- **Gap Role 커버리지**: 직접 메우는 기술적 역할
- **대응 이유**: 원문 근거로 메우는 내용
- **남는 차이**: 아직 대응하지 않는 제한

`직접 링크`에는 `[특허번호](URL)` 같은 이름 링크를 쓰지 말고, 복사할 수 있도록 `https://`로 시작하는 전체 URL 문자열을 그대로 표시하십시오.
논문 후보의 `문헌번호(이름)`에는 DOI, 저널 권·호·쪽수 또는 데이터베이스 식별자만 적지 말고 논문의 정식 제목을 적으십시오. DOI는 필요한 경우 `직접 링크` URL에 사용하십시오.
## 검색 결과 평가
- 직접 대응 / 부분 대응 / 대응 불충분
- 추가 확인이 필요한 공개일 또는 원문 근거"""


async def run_adaptive_prior_art_search(
    claim_text: str,
    targets: List[Dict],
    used_documents: List[Dict],
    settings: Settings,
    additional_query: str = "",
    runner: SearchRunner = call_ai,
) -> Dict:
    prompts = build_parallel_search_prompts(
        claim_text, targets, used_documents, additional_query,
    )
    initial = await asyncio.gather(*[
        runner(
            item["prompt"],
            SEARCH_SYSTEM,
            settings,
            agent="compare",
            web_search=True,
        )
        for item in prompts
    ], return_exceptions=True)
    result_items = [
        {"axis": item["axis"], "result": result.strip()}
        for item, result in zip(prompts, initial)
        if isinstance(result, str) and result.strip()
    ]
    expanded = needs_expansion([item["result"] for item in result_items])
    if expanded:
        try:
            expansion = await runner(
                build_expansion_prompt(
                    claim_text,
                    targets,
                    [item["result"] for item in result_items],
                    additional_query,
                ),
                SEARCH_SYSTEM,
                settings,
                agent="compare",
                web_search=True,
            )
        except Exception:
            expansion = ""
        if expansion and expansion.strip():
            result_items.append({"axis": "expanded", "result": expansion.strip()})

    synthesis = await runner(
        build_synthesis_prompt(claim_text, targets, result_items),
        "당신은 검색 결과 정리자입니다. 제공된 결과만 사용하고 새로운 문헌이나 URL을 만들지 마십시오.",
        settings,
        agent="compare",
        web_search=False,
    )
    return {
        "result_md": synthesis.strip(),
        "expanded": expanded,
        "search_axes": [item["axis"] for item in result_items],
        "initial_stats": search_result_stats([item["result"] for item in result_items[:3]]),
        "final_stats": search_result_stats([item["result"] for item in result_items]),
    }
