import json
from backend.models.schemas import ClaimElement, ElementMatch, ParsedClaim
from backend.services.citation_extractor import _parse_json_array
from backend.services.prompt_loader import load_prompt
from backend.services.report_generator import _normalize_report_markdown, enforce_phase1_judgment_headers


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
    assert parsed[0]["llm_judgment"] == "동일"
    assert parsed[0]["judgment_adjusted"] is True
    assert parsed[0]["judgment_adjustment_reason"] == "directness_inferred"
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
    assert '"원문 발췌"' in prompt
    assert "실제번호" in prompt


def test_report_quote_block_removes_redundant_reporting_ending_only():
    report = (
        "### [구성요소]\n\n(A) 실질적동일 92%\n\n"
        "- **인용발명 대응 원문:**\n"
        " 인용발명 1에는 송신기가 헤드 유닛에 신호를 송신한다고 기재되어 있습니다.\n\n"
        "- 판단 이유: 송신기가 기재되어 있습니다."
    )

    normalized = _normalize_report_markdown(report)

    assert "송신기가 헤드 유닛에 신호를 송신한다." in normalized
    assert "송신한다고 기재되어 있습니다" not in normalized
    assert "- 판단 이유: 송신기가 기재되어 있습니다." in normalized


def test_comparison_and_report_prompts_forbid_importing_unrecited_limitations():
    compare_system = load_prompt("system_compare.txt", "")
    report_system = load_prompt("system_report_base.txt", "")
    phase1_prompt = load_prompt("prompt_phase1_main.txt", "")

    assert "청구항에 없는 `주된`" in compare_system
    assert "A가 주된 기준 또는 유일한 기준이어야 한다고 해석하지 않습니다" in compare_system
    assert "청구항보다 좁은 실시형태" in compare_system
    assert "청구항에 없는 `주된`" in report_system
    assert "신규성의 직접 개시 여부와 진보성의 기능적 대응" in report_system
    assert "신규성의 직접성 부족을 그대로 진보성의 결합 곤란으로 전환하지 마십시오" in phase1_prompt


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
    assert parsed[0]["llm_judgment"] == "동일"
    assert parsed[0]["judgment_adjusted"] is True
    assert parsed[0]["judgment_adjustment_reason"] == "multiple_or_composite_missing_limitations"
    assert parsed[0]["missing_limitations"] == ["제2 기초 롤 각도", "제1·제2 기초 롤 각도 결합 산출"]


def test_unadjusted_judgment_keeps_llm_and_final_values_for_audit():
    response = json.dumps([{
        "label": "B",
        "found": True,
        "quote": "the camera array is rotated with horizontal motion",
        "chunk_id": "[0002]",
        "judgment": "실질적 동일",
        "판단_이유": "카메라 배열부가 수평 이동과 함께 회전하는 구성이 직접 대응한다.",
        "directness": "direct",
        "missing_limitations": [],
        "evidence": [],
    }], ensure_ascii=False)

    parsed = _parse_json_array(
        response,
        [ClaimElement(label="B", text="카메라 배열부를 회전 및 이동시키는 스테이지")],
    )

    assert parsed[0]["llm_judgment"] == "실질적 동일"
    assert parsed[0]["judgment"] == "실질적 동일"
    assert parsed[0]["judgment_adjusted"] is False
    assert parsed[0]["judgment_adjustment_reason"] == ""


def test_terminology_difference_phrase_does_not_downgrade_direct_match():
    response = json.dumps([{
        "label": "A",
        "found": True,
        "quote": "Metadata and subtitle text are obtained from the media content.",
        "chunk_id": "[0022]",
        "judgment": "실질적 동일",
        "판단_이유": "영상 자막 등의 텍스트 정보 및 콘텐츠 메타데이터를 획득하는 구성이 용어 차이 외에 실질적으로 동일하게 개시되어 있습니다.",
        "directness": "direct",
        "missing_limitations": [],
        "evidence": [],
    }], ensure_ascii=False)

    parsed = _parse_json_array(
        response,
        [ClaimElement(label="A", text="영상의 텍스트 데이터 및 콘텐츠 메타데이터를 획득")],
    )

    assert parsed[0]["judgment"] == "실질적 동일"
    assert parsed[0]["judgment_adjusted"] is False
    assert parsed[0]["judgment_adjustment_reason"] == ""


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
