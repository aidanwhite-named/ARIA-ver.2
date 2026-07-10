import json
from backend.models.schemas import ClaimElement, ElementMatch, ParsedClaim
from backend.services.citation_extractor import _parse_json_array
from backend.services.prompt_loader import load_prompt
from backend.services.report_generator import enforce_phase1_judgment_headers


def test_independent_preamble_is_compared_as_p_limitation():
    claim = ParsedClaim(
        claim_number=1,
        claim_type="independent",
        text="2축 짐벌과 소프트웨어 롤 보정을 이용한 방법에 있어서, 처리하는 단계",
        preamble="2축 짐벌과 소프트웨어 롤 보정을 이용한 방법에 있어서",
        elements=[ClaimElement(label="A", text="처리하는 단계")],
    )

    assert [element.label for element in claim.elements] == ["P", "A"]
    assert claim.elements[0].importance == "5"


def test_missing_limitation_prevents_identical_judgment():
    response = json.dumps([{
        "label": "A",
        "found": True,
        "quote": "a camera captures an image",
        "chunk_id": "[0001]",
        "judgment": "동일",
        "판단_이유": "영상 생성은 대응되지만 기준 해상도는 명시되지 않는다.",
        "directness": "inferred",
        "missing_limitations": ["기준 해상도"],
        "evidence": [],
    }], ensure_ascii=False)

    parsed = _parse_json_array(
        response,
        [ClaimElement(label="A", text="기준 해상도의 원본 영상을 생성")],
    )

    assert parsed[0]["judgment"] == "일부 차이"
    assert parsed[0]["missing_limitations"] == ["기준 해상도"]


def test_report_header_is_clamped_to_structured_judgment():
    original_quote = '(단락 [0001], "The camera captures the image.")'
    report = (
        "### [구성요소 A]\n\n(A) 동일 100%\n\n"
        f"- 인용발명 대응 원문: 카메라는 영상을 촬영한다.\n{original_quote}\n\n"
        "- 판단 이유: 기준 해상도는 확인되지 않는다."
    )
    matches = [
        ElementMatch(
            label="A",
            judgment="일부 차이",
            quote="camera image",
            chunk_id="[0001]",
        )
    ]

    corrected = enforce_phase1_judgment_headers(report, matches)

    assert "(A) 일부차이 87%" in corrected
    assert "(A) 동일 100%" not in corrected
    assert original_quote in corrected


def test_foreign_quote_format_keeps_translation_original_and_paragraph_number():
    prompt = load_prompt("format_phase1_independent.txt", "")

    assert "직역 또는 준직역한 한국어 문장" in prompt
    assert '"quote에 있는 원문 언어 그대로의 발췌"' in prompt
    assert "단락 [XXXX]" in prompt


def test_composite_missing_limitation_caps_to_partial_similarity():
    response = json.dumps([{
        "label": "A",
        "found": True,
        "quote": "processor analyzes image data to determine rotation",
        "chunk_id": "[0098]",
        "judgment": "동일",
        "판단_이유": "이미지 데이터 분석은 대응되나 제2 기초 롤 각도와 제1·제2 기초 롤 각도의 결합 산출은 확인되지 않는다.",
        "directness": "direct",
        "missing_limitations": ["제2 기초 롤 각도", "제1·제2 기초 롤 각도 결합 산출"],
        "evidence": [],
    }], ensure_ascii=False)

    parsed = _parse_json_array(
        response,
        [ClaimElement(label="A", text="제1 기초 롤 각도와 제2 기초 롤 각도에 기초하여 최종 롤 각도를 산출")],
    )

    assert parsed[0]["judgment"] == "일부 유사"
    assert parsed[0]["missing_limitations"] == ["제2 기초 롤 각도", "제1·제2 기초 롤 각도 결합 산출"]


def test_element_match_preserves_coverage_metadata():
    match = ElementMatch(
        label="A",
        found=True,
        quote="processor analyzes image data",
        chunk_id="[0098]",
        judgment="일부 유사",
        directness="inferred",
        missing_limitations=["결합 산출"],
    )

    assert match.directness == "inferred"
    assert match.missing_limitations == ["결합 산출"]
