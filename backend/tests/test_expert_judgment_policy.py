import json
import pytest
from backend.models.schemas import ClaimElement, ElementMatch, ExtractedDocument, ParsedClaim, PatentChunk
from backend.services.citation_extractor import CompareFailed, _doc_chunks, _parse_json_array
from backend.services.prompt_loader import load_prompt
from backend.services.report_generator import (
    _dedupe_phase1_sections,
    _normalize_report_markdown,
    enforce_phase1_judgment_headers,
)


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


def test_inferred_label_alone_does_not_downgrade_complete_verified_evidence():
    response = json.dumps([{
        "label": "C",
        "found": True,
        "quote": (
            "textual instructions input can be combined with corresponding 3D CAD data, "
            "resulting in a fused spec including technical instructions"
        ),
        "quote_translation": (
            "텍스트 지침 입력은 대응하는 3D CAD 데이터와 결합되어 "
            "기술 지침을 포함하는 융합 사양을 생성할 수 있다."
        ),
        "chunk_id": "D3-P-0100",
        "judgment": "실질적 동일",
        "판단_이유": "사용자 입력과 3D 정보를 융합하여 구조화된 기술 지침을 생성한다.",
        "directness": "inferred",
        "missing_limitations": [],
        "evidence": [],
    }], ensure_ascii=False)

    parsed = _parse_json_array(
        response,
        [ClaimElement(
            label="C",
            text="사용자 입력 및 3D 정보를 종합하여 3D 모델 생성을 위한 구조화된 명령어를 생성",
        )],
    )

    assert parsed[0]["judgment"] == "실질적 동일"
    assert parsed[0]["llm_judgment"] == "실질적 동일"
    assert parsed[0]["judgment_adjusted"] is False
    assert parsed[0]["missing_limitations"] == []


def test_direct_substantially_identical_reason_reconciles_partial_difference_label():
    response = json.dumps([{
        "label": "A",
        "found": True,
        "quote": "The array determines the horizontal and vertical bearing angles.",
        "chunk_id": "[0103]",
        "judgment": "일부 차이",
        "판단_이유": "음향 신호로 음원의 방향 정보를 생성하므로 해당 구성과 실질적으로 동일합니다.",
        "directness": "direct",
        "missing_limitations": [],
        "evidence": [],
    }], ensure_ascii=False)

    parsed = _parse_json_array(
        response,
        [ClaimElement(label="A", text="음향 신호에 기초하여 음원의 방향 정보를 생성")],
    )

    assert parsed[0]["judgment"] == "실질적 동일"
    assert parsed[0]["directness"] == "direct"
    assert parsed[0]["missing_limitations"] == []
    assert parsed[0]["evidence_status"] == "verified"
    assert parsed[0]["llm_judgment"] == "일부 차이"
    assert parsed[0]["judgment_adjustment_reason"] == "reason_label_reconciliation"


@pytest.mark.parametrize("raw_missing", [["없음"], "해당 없음", ["N/A"], ["차이 없음."]])
def test_empty_missing_limitation_markers_do_not_downgrade_judgment(raw_missing):
    response = json.dumps([{
        "label": "A",
        "found": True,
        "quote": "The system can receive an input inserted by a user.",
        "quote_translation": "시스템은 사용자가 입력한 입력을 수신할 수 있다.",
        "chunk_id": "[0097]",
        "judgment": "동일",
        "판단_이유": "사용자 입력을 획득하는 구성과 동일하다.",
        "directness": "direct",
        "missing_limitations": raw_missing,
        "evidence": [],
    }], ensure_ascii=False)

    parsed = _parse_json_array(
        response,
        [ClaimElement(label="A", text="적어도 하나의 사용자 입력을 획득")],
    )

    assert parsed[0]["judgment"] == "동일"
    assert parsed[0]["judgment_adjusted"] is False
    assert parsed[0]["judgment_adjustment_reason"] == ""
    assert parsed[0]["missing_limitations"] == []


def test_true_summary_section_is_excluded_from_comparison_context():
    doc = ExtractedDocument(
        filename="legacy-korean.pdf",
        paragraph_chunks=[
            PatentChunk(
                chunk_id="D1-P-P003",
                section="요약",
                original_text="요약\n발명의 핵심을 간략히 설명한다.",
            )
        ],
    )

    assert _doc_chunks(doc) == []


def test_non_patent_body_is_not_dropped_when_section_detection_labels_it_abstract():
    doc = ExtractedDocument(
        filename="paper.pdf",
        document_type="non_patent",
        paragraph_chunks=[
            PatentChunk(
                chunk_id="D1-P-P003",
                section="Abstract",
                original_text=(
                    "METHOD\nThe multimodal encoder processes image and point cloud "
                    "inputs to generate a CAD command sequence."
                ),
            )
        ],
    )

    chunks = _doc_chunks(doc)

    assert len(chunks) == 1
    assert "CAD command sequence" in chunks[0][1]


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

    assert "(A) 85~89%: 기술 사상 동일, 세부 구현 방식의 단순 변경 🟠 87%" in corrected
    assert "(A) 동일 100%" not in corrected
    assert original_quote in corrected


def test_new_format_report_headers_follow_structured_judgments_for_each_component():
    report = (
        "### (A)\n\n"
        "- 청구항 구성: (A) 고 임피던스 입력\n"
        "- 유사도 평가: 80% 미만: 대응 안됨 ⚪\n"
        "- 판단 이유: 고 임피던스 입력이 직접 개시되어 있음.\n"
        "- 차이점: 없음\n\n"
        "### (B)\n\n"
        "- 청구항 구성: (B) 조정 가능한 부하\n"
        "- 유사도 평가: 95% 이상: 동일 🔵\n"
        "- 판단 이유: 해당 구성이 확인되지 않음.\n"
        "- 차이점: 조정 가능한 부하가 부재함.\n"
    )
    matches = [
        ElementMatch(
            label="A",
            judgment="동일",
            quote="high impedance inputs",
            directness="direct",
        ),
        ElementMatch(
            label="B",
            judgment="대응 없음",
            directness="absent",
        ),
    ]

    corrected = enforce_phase1_judgment_headers(report, matches)

    assert corrected.count("- 유사도 평가: 95% 이상: 동일 🔵") == 1
    assert corrected.count("- 유사도 평가: 80% 미만: 대응 안됨 ⚪") == 1


def test_missing_similarity_line_is_inserted_from_structured_judgment():
    report = (
        "### [구성요소]\n\n"
        "- 청구항 구성: (D) 제1 결과와 제2 출력값에 따라 카메라를 제어\n"
        "- 판단 이유: 일반 카메라 제어만 확인됨.\n"
        "- 차이점: 두 입력의 결합관계가 확인되지 않음.\n"
    )
    matches = [
        ElementMatch(
            label="D",
            judgment="일부 유사",
            quote="camera control",
            directness="inferred",
            missing_limitations=["두 입력의 결합관계"],
        )
    ]

    corrected = enforce_phase1_judgment_headers(report, matches)

    assert corrected.count(
        "- 유사도 평가: 80~84%: 핵심 기능 유사하나 목적/효과에 일부 차이🟡"
    ) == 1
    assert corrected.index("- 청구항 구성: (D)") < corrected.index("- 유사도 평가:")
    assert corrected.index("- 유사도 평가:") < corrected.index("- 판단 이유:")


def test_report_format_keeps_instructions_out_of_output_skeleton():
    system = load_prompt("system_report_base.txt", "")

    for filename in (
        "format_phase1_independent.txt",
        "format_phase1_combo.txt",
        "format_phase1_dependent.txt",
    ):
        output_format = load_prompt(filename, "")
        assert "###" in output_format
        assert "인용발명 대응 및 판단:" in output_format
        assert "병기하십시오" not in output_format
        assert "안내 문구" not in output_format
    assert "[0000]" in system
    assert "95%" in system
    assert "80~84%" in system
    assert "인용발명 1이 구성 대부분에 대응하지만 일부 세부 구성이 결여되고" in system
    assert "`차이점`에 두 문헌의 발췌문과 실제 위치를 한 문장에 함께" in system
    assert "결합으로 해소된 경우" in system

    combo_format = load_prompt("format_phase1_combo.txt", "")
    dependent_format = load_prompt("format_phase1_dependent.txt", "")
    for output_format in (combo_format, dependent_format):
        assert "두 문헌의 발췌문·실제 위치와 보완 판단을 한 문장에 함께" in output_format
        assert "`차이점 참조`" in output_format


def test_specialized_independent_prompts_forbid_summary_sections():
    novelty_prompt = load_prompt("prompt_novelty_rejection.txt", "")
    rejection_impossible_prompt = load_prompt("prompt_rejection_impossible.txt", "")

    for prompt in (novelty_prompt, rejection_impossible_prompt):
        assert "`- 신규성 검토:`" not in prompt
        assert "`- 진보성 검토:`" not in prompt


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


def test_report_quote_block_merges_translation_and_excerpt_into_one_sentence():
    report = (
        "### [구성요소]\n\n"
        "- **인용발명 대응 원문:**\n"
        "**\n"
        "- 번역(인용발명 1): 진폭 제어 회로 및 VGA에 의해 자동 이득 제어(AGC) 루프가 형성된다.\n"
        '- 발췌(인용발명 1): "wherein an automatic gain control (AGC) loop is formed by '
        'the amplitude control circuit and the VGA." (단락 D4-P-1074)\n'
        "- 판단 이유: 대응 관계가 확인됨"
    )

    normalized = _normalize_report_markdown(report)

    assert "**\n" not in normalized
    assert (
        "인용발명 1에는 진폭 제어 회로 및 VGA에 의해 "
        "자동 이득 제어(AGC) 루프가 형성된다. 단락 D4-P-1074 "
        '"wherein an automatic gain control (AGC) loop is formed by '
        'the amplitude control circuit and the VGA."'
    ) in normalized
    assert "번역(인용발명 1)" not in normalized
    assert "발췌(인용발명 1)" not in normalized


def test_comparison_and_report_prompts_forbid_importing_unrecited_limitations():
    compare_system = load_prompt("system_compare.txt", "")
    report_system = load_prompt("system_report_base.txt", "")
    phase1_prompt = load_prompt("prompt_phase1_main.txt", "")
    single_prompt = load_prompt("prompt_compare_single.txt", "")
    hybrid_prompt = load_prompt("prompt_compare_hybrid.txt", "")

    assert "청구항에 없는 `주된`" in compare_system
    assert "A가 주된 기준 또는 유일한 기준이어야 한다고 해석하지 않습니다" in compare_system
    assert "청구항보다 좁은 실시형태" in compare_system
    assert "입력-출력 방향성 규칙" in compare_system
    assert "평면도→3D 모델" in compare_system
    assert "현재 구성 단위 평가 규칙" in compare_system
    assert "후속 구성의 명시적 입력 결합 규칙" in compare_system
    assert "그 두 결과가 실제 입력되고 함께 Z를 좌우하는 연결관계" in compare_system
    assert "인용·번역 충실성 규칙" in compare_system
    assert "청구항에 없는 `주된`" in report_system
    assert "신규성의 직접 개시 여부와 진보성의 기능적 대응" in report_system
    assert "신규성의 직접성 부족을 그대로 진보성의 결합 곤란으로 전환하지 마십시오" in phase1_prompt
    assert "추가 검색·검토가 필요함" in phase1_prompt
    assert "입력→처리→출력" in single_prompt
    assert "앞선 label의 세부 제한을 후속 label에 반복해 중복 차감하지 말고" in single_prompt
    assert "입력→처리→출력" in hybrid_prompt
    assert "번역문에는 현재 인용 원문에 없는 행위나 결과를 추가하지 마십시오" in hybrid_prompt

    independent_format = load_prompt("format_phase1_independent.txt", "")
    combo_format = load_prompt("format_phase1_combo.txt", "")
    assert "종합 분석 요약" not in independent_format
    assert "신규성 검토:" not in independent_format
    assert "진보성 검토:" not in independent_format
    assert "종합 분석 요약" not in combo_format
    assert "신규성 검토:" not in combo_format
    assert "진보성 검토:" not in combo_format


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


def test_single_relational_missing_limitation_is_not_lexically_downgraded():
    response = json.dumps([{
        "label": "C",
        "found": True,
        "quote": "The circuit generates the modulation-driving signal and the demodulation-driving signal.",
        "chunk_id": "[0060]",
        "judgment": "일부 차이",
        "판단_이유": "두 구동 신호는 개시되나 결정된 두 진폭에 각각 연동되는 관계는 확인되지 않는다.",
        "directness": "inferred",
        "missing_limitations": ["결정된 복조 진폭과 변조 진폭에 각각 연동되는 관계"],
        "evidence": [],
    }], ensure_ascii=False)

    parsed = _parse_json_array(
        response,
        [ClaimElement(
            label="C",
            text="복조 진폭 및 변조 진폭에 각각 따라 두 구동 신호를 생성",
        )],
    )

    assert parsed[0]["judgment"] == "일부 차이"
    assert parsed[0]["llm_judgment"] == "일부 차이"
    assert parsed[0]["judgment_adjusted"] is False
    assert parsed[0]["judgment_adjustment_reason"] == ""


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


def _korean_source_doc(*paragraphs: str) -> ExtractedDocument:
    return ExtractedDocument(
        filename="KR-prior.pdf",
        paragraph_chunks=[
            PatentChunk(
                chunk_id=f"D1-P-{index:04d}",
                section="발명의 상세한 설명",
                original_text=text,
            )
            for index, text in enumerate(paragraphs, start=1)
        ],
    )


def test_non_verbatim_model_composition_is_recovered_only_as_partial_evidence():
    source = _korean_source_doc(
        "객체의 이동방향과 속도를 계산하고 카메라의 PAN, TILT 모터를 구동하여 객체를 추적한다."
    )
    response = json.dumps([{
        "label": "D",
        "found": True,
        "quote": "AI 의사결정부의 결정 결과 및 좌표 보정부의 출력값에 따라 PAN, TILT 모터를 구동한다.",
        "chunk_id": "D1-P-0001",
        "judgment": "동일",
        "판단_이유": "두 결과에 따라 카메라를 제어하므로 동일하다.",
        "directness": "direct",
        "missing_limitations": [],
        "evidence": [],
    }], ensure_ascii=False)

    parsed = _parse_json_array(
        response,
        [ClaimElement(
            label="D",
            text="AI 의사결정부의 결정 결과 및 좌표 보정부의 출력값에 따라 카메라 방향을 제어",
        )],
        source_docs=[source],
    )

    assert parsed[0]["found"] is True
    assert parsed[0]["judgment"] == "일부 차이"
    assert parsed[0]["quote"] == source.paragraph_chunks[0].original_text
    assert parsed[0]["directness"] == "inferred"
    assert parsed[0]["evidence_status"] == "recovered_from_cited_chunk"
    assert parsed[0]["technical_judgment"] == "동일"
    assert any("함께 입력" in value for value in parsed[0]["missing_limitations"])


def test_foreign_language_translation_participates_in_compound_relationship_guard():
    source = ExtractedDocument(
        filename="foreign-prior.pdf",
        paragraph_chunks=[
            PatentChunk(
                chunk_id="D1-P-0101",
                section="DETAILED DESCRIPTION",
                original_text=(
                    "The camera and microphone array are independently positioned. "
                    "The device establishes a coordinate system transformation and "
                    "applies a compensation control parameter to calculate pan and tilt angles."
                ),
            )
        ],
    )
    quote = source.paragraph_chunks[0].original_text
    response = json.dumps([{
        "label": "B",
        "found": True,
        "quote": quote,
        "quote_translation": (
            "카메라와 마이크 어레이는 독립적으로 설치되고, 장치는 좌표계 변환 관계를 "
            "설정한 다음 보상 제어 파라미터를 적용하여 팬 및 틸트 각도를 계산한다."
        ),
        "chunk_id": "D1-P-0101",
        "judgment": "실질적 동일",
        "판단_이유": "설치 상태에 맞춘 좌표 변환 및 보상 관계가 직접 기재되어 있다.",
        "directness": "direct",
        "missing_limitations": [],
        "evidence": [],
    }], ensure_ascii=False)

    parsed = _parse_json_array(
        response,
        [ClaimElement(
            label="B",
            text=(
                "음향 위치 정보를 카메라 제어 좌표 정보로 변환하면서 "
                "설치 환경에 따른 오차를 보정하는 좌표 보정부"
            ),
        )],
        source_docs=[source],
    )

    assert parsed[0]["judgment"] == "실질적 동일"


def test_foreign_quote_without_korean_translation_is_rejected():
    source = ExtractedDocument(
        filename="foreign-prior.pdf",
        paragraph_chunks=[
            PatentChunk(
                chunk_id="D1-P-0101",
                original_text="The controller generates a driving signal.",
            )
        ],
    )
    response = json.dumps([{
        "label": "A",
        "found": True,
        "quote": "The controller generates a driving signal.",
        "quote_translation": "",
        "chunk_id": "D1-P-0101",
        "judgment": "동일",
        "판단_이유": "구동 신호 생성이 직접 기재되어 있다.",
        "directness": "direct",
        "missing_limitations": [],
        "evidence": [],
    }], ensure_ascii=False)

    with pytest.raises(CompareFailed, match="quote_translation"):
        _parse_json_array(
            response,
            [ClaimElement(label="A", text="구동 신호를 생성하는 제어기")],
            source_docs=[source],
        )


def test_compound_relationship_guard_downgrades_aria_sound_tracking_overclaims():
    source = _korean_source_doc(
        "음향처리데이터를 토대로 PTZ가 제어되는 회전형 카메라에서 획득한 영상신호를 분석한다.",
        "유의미한 객체가 없는 경우 발생시간, 방향, 영상분석 결과를 저장하고 동일한 상황은 음향이벤트에서 예외 처리한다.",
        "회전형 카메라는 촬영동작이 수행되지 않거나 한 곳을 촬영하거나 랜덤으로 회전하는 상태일 수 있다.",
        "이동방향과 속도를 계산하고 카메라의 PAN, TILT 모터를 구동하여 객체를 추적한다.",
    )
    elements = [
        ClaimElement(
            label="B",
            text="음향 위치 정보를 카메라 제어 좌표 정보로 변환하면서 설치 환경에 따른 오차를 보정하는 좌표 보정부",
        ),
        ClaimElement(
            label="C",
            text="이벤트 특성 정보 및 카메라의 현재 동작 상태를 기초로 카메라의 방향 전환 여부를 결정하는 AI 기반 의사결정부",
        ),
        ClaimElement(
            label="D",
            text="AI 기반 의사결정부의 결정 결과 및 좌표 보정부의 출력값에 따라 카메라의 방향을 제어하는 카메라 제어부",
        ),
    ]
    response = json.dumps([
        {
            "label": "B", "found": True,
            "quote": source.paragraph_chunks[0].original_text,
            "chunk_id": "D1-P-0001", "judgment": "실질적 동일",
            "판단_이유": "음향 데이터로 PTZ를 제어하고 예외 처리한다.",
            "directness": "direct", "missing_limitations": [],
            "evidence": [{
                "limitation": "예외 처리",
                "quote": source.paragraph_chunks[1].original_text,
                "chunk_id": "D1-P-0002",
            }],
        },
        {
            "label": "C", "found": True,
            "quote": source.paragraph_chunks[2].original_text,
            "chunk_id": "D1-P-0003", "judgment": "실질적 동일",
            "판단_이유": "카메라 상태가 기재되어 있다.",
            "directness": "direct", "missing_limitations": [],
            "evidence": [],
        },
        {
            "label": "D", "found": True,
            "quote": source.paragraph_chunks[3].original_text,
            "chunk_id": "D1-P-0004", "judgment": "동일",
            "판단_이유": "카메라 방향 제어가 기재되어 있다.",
            "directness": "direct", "missing_limitations": [],
            "evidence": [],
        },
    ], ensure_ascii=False)

    parsed = _parse_json_array(response, elements, source_docs=[source])
    by_label = {item["label"]: item for item in parsed}

    assert by_label["B"]["judgment"] == "일부 차이"
    assert by_label["C"]["judgment"] == "일부 차이"
    assert by_label["D"]["judgment"] == "일부 차이"
    assert all(by_label[label]["directness"] == "inferred" for label in ("B", "C", "D"))
    assert any("제어 좌표 정보로 변환" in value for value in by_label["B"]["missing_limitations"])
    assert any("현재 동작 상태를 판단 입력" in value for value in by_label["C"]["missing_limitations"])
    assert any("함께 입력" in value for value in by_label["D"]["missing_limitations"])
