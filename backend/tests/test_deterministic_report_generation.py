import unittest
from unittest.mock import patch

from backend.models.schemas import (
    ClaimElement,
    ElementMatch,
    EvidenceSpan,
    ExtractedDocument,
    ParsedClaim,
    PatentChunk,
    Settings,
)
from backend.services.report_generator import (
    _format_citation_location,
    find_unselected_reference_mentions,
    generate_dependent_reports_batch,
    generate_independent_phase1_streaming,
    polish_phase1_summary_text,
)


class DeterministicReportGenerationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.claim = ParsedClaim(
            claim_number=1,
            text="(A) 신호를 수신하고 (B) 신호를 변환하는 장치",
            elements=[
                ClaimElement(label="A", text="신호를 수신하는 수신부"),
                ClaimElement(label="B", text="수신 신호를 변환하는 변환부"),
            ],
        )
        self.document = ExtractedDocument(filename="prior.pdf")
        self.matches = [
            ElementMatch(
                label="A",
                found=True,
                quote="신호를 수신하는 수신기를 포함한다.",
                chunk_id="[0001]",
                judgment="동일",
                directness="direct",
                similarity_reason="수신 기능과 대상이 직접 대응됨.",
            ),
            ElementMatch(
                label="B",
                found=False,
                judgment="대응 없음",
                missing_limitations=["수신 신호의 변환"],
                similarity_reason="변환 기능이 확인되지 않음.",
            ),
        ]
        self.chain = {
            "total": [0],
            "doc_name_mapping": {"0": "인용발명 1"},
            "combination_validity": {
                "coverage_complete": False,
                "remaining_uncovered_labels": ["B"],
            },
        }

    async def test_independent_report_does_not_call_report_llm(self):
        with patch(
            "backend.services.report_generator.call_ai",
            side_effect=AssertionError("report LLM must not be called"),
        ):
            chunks = [
                chunk
                async for chunk in generate_independent_phase1_streaming(
                    self.claim,
                    self.matches,
                    [self.document],
                    self.chain,
                    Settings(),
                )
            ]

        report = "".join(chunks)
        self.assertIn("- 청구항 구성: (A) 신호를 수신하는 수신부", report)
        self.assertIn("수신 신호의 변환", report)
        self.assertIn(
            '인용발명 1 (prior.pdf)에는 "신호를 수신하는 수신기를 포함한다." '
            '(단락 [0001])라는 구성이 기재되어 있으며',
            report,
        )
        self.assertIn("[종합 분석 요약]", report)
        self.assertIn(
            "- 유사점 요약: 인용발명 1 (prior.pdf)은 청구항의 "
            "(A) 신호를 수신하는 수신부와 유사함.",
            report,
        )
        self.assertNotIn("- 진보성 검토:", report)
        self.assertNotIn("- 신규성 검토:", report)
        self.assertNotIn("- 잔여 차이 및 방어 포인트:", report)
        self.assertIn("구성 (B)의 직접·완전한 개시가 확인되지 않아", report)

    async def test_final_judgment_ending_is_completed_before_conclusion_line(self):
        self.matches[0].similarity_reason = (
            "통신 인터페이스를 통해 사용자 입력을 수신하므로 동일합니다."
        )

        report = "".join([
            chunk
            async for chunk in generate_independent_phase1_streaming(
                self.claim,
                self.matches,
                [self.document],
                self.chain,
                Settings(),
            )
        ])

        self.assertIn(
            "통신 인터페이스를 통해 사용자 입력을 수신하므로 동일합니다.\n"
            '  따라서, 청구항의 "신호를 수신하는 수신부" 구성과 대응됩니다.',
            report,
        )
        self.assertNotIn("동일합니다로", report)

    async def test_partial_judgments_use_qualified_conclusion_lines(self):
        expected_by_judgment = {
            "실질적 동일": "구성과 실질적으로 대응됩니다.",
            "일부 차이": "구성과 대체로 대응되나, 일부 세부 구성에는 차이가 있습니다.",
            "일부 유사": "구성과 일부 기능이 유사하게 대응되나, 목적 또는 효과에는 차이가 있습니다.",
            "차이": "구성과 직접 대응된다고 보기 어렵습니다.",
            "대응 없음": "구성과 대응되지 않습니다.",
        }

        for judgment, expected in expected_by_judgment.items():
            with self.subTest(judgment=judgment):
                match = self.matches[0].model_copy(update={"judgment": judgment})
                report = "".join([
                    chunk
                    async for chunk in generate_independent_phase1_streaming(
                        self.claim,
                        [match],
                        [self.document],
                        self.chain,
                        Settings(),
                    )
                ])
                self.assertIn(expected, report)

    async def test_conclusion_collapses_line_breaks_inside_claim_element(self):
        match = self.matches[0].model_copy(
            update={"judgment": "일부 차이"}
        )
        claim = self.claim.model_copy(
            update={
                "elements": [
                    ClaimElement(
                        label="A",
                        text="구조화된 명령어를 기반으로 상기 3D 모델 생성을 위한\nCAD 표현을 생성하는 장치",
                    ),
                    self.claim.elements[1],
                ]
            }
        )

        report = "".join([
            chunk
            async for chunk in generate_independent_phase1_streaming(
                claim,
                [match],
                [self.document],
                self.chain,
                Settings(),
            )
        ])

        self.assertIn(
            '따라서, 청구항의 "구조화된 명령어를 기반으로 상기 3D 모델 생성을 위한 '
            'CAD 표현을 생성하는 장치" 구성과 대체로 대응되나,',
            report,
        )
        self.assertNotIn("위한\nCAD 표현", report)

    async def test_component_separator_is_not_emitted(self):
        report = "".join([
            chunk
            async for chunk in generate_independent_phase1_streaming(
                self.claim,
                self.matches,
                [self.document],
                self.chain,
                Settings(),
            )
        ])

        self.assertNotIn("\n---\n", report)

    async def test_dependent_batch_is_rendered_without_llm(self):
        dependent = self.claim.model_copy(
            update={"claim_number": 2, "claim_type": "dependent", "parent_claim": 1}
        )
        with patch(
            "backend.services.report_generator.call_ai",
            side_effect=AssertionError("report LLM must not be called"),
        ):
            report = await generate_dependent_reports_batch(
                [(dependent, self.matches, self.chain, None)],
                [self.document],
                Settings(),
            )

        self.assertIn("===청구항 2===", report)
        self.assertNotIn("[종합 분석 요약]", report)

    async def test_upload_order_mentions_are_remapped_to_final_reference_names(self):
        for match in self.matches:
            match.cited_invention_index = 2
        self.matches[0].similarity_reason = (
            "인용발명 3은 수신 기능을 직접 개시합니다."
        )
        chain = {
            "total": [2],
            "doc_name_mapping": {
                "0": "인용발명 3",
                "1": "인용발명 2",
                "2": "인용발명 1",
            },
        }
        chunks = [
            chunk
            async for chunk in generate_independent_phase1_streaming(
                self.claim,
                self.matches,
                [self.document, self.document, self.document],
                chain,
                Settings(),
            )
        ]
        report = "".join(chunks)

        self.assertIn("인용발명 1은 수신 기능", report)
        self.assertNotIn("인용발명 3은 수신 기능", report)
        self.assertEqual(find_unselected_reference_mentions(report, chain), [])

    async def test_match_reason_uses_selected_document_when_cached_number_is_ambiguous(self):
        self.matches[0].cited_invention_index = 2
        self.matches[0].similarity_reason = (
            "인용발명 2의 문단에는 수신 기능이 직접 개시되어 있습니다."
        )
        chain = {
            "total": [2, 0, 1],
            "doc_name_mapping": {
                "0": "인용발명 2",
                "1": "인용발명 3",
                "2": "인용발명 1",
            },
        }
        chunks = [
            chunk
            async for chunk in generate_independent_phase1_streaming(
                self.claim,
                self.matches,
                [self.document, self.document, self.document],
                chain,
                Settings(),
            )
        ]
        report = "".join(chunks)

        self.assertIn("인용발명 1의 문단에는 수신 기능", report)
        self.assertNotIn("인용발명 3의 문단에는 수신 기능", report)

    async def test_identical_representative_and_sub_limitation_quotes_are_not_duplicated(self):
        self.matches[0].evidence = [
            EvidenceSpan(
                limitation="신호 수신",
                quote="신호를 수신하는 수신기를 포함한다.",
                chunk_id="[0001]",
            )
        ]
        chunks = [
            chunk
            async for chunk in generate_independent_phase1_streaming(
                self.claim,
                self.matches,
                [self.document],
                self.chain,
                Settings(),
            )
        ]
        report = "".join(chunks)

        self.assertEqual(report.count("신호를 수신하는 수신기를 포함한다."), 1)
        self.assertNotIn("하위 제한별 근거:", report)

    async def test_summary_does_not_merge_different_documents_into_single_reference_novelty(self):
        self.matches[0].cited_invention_index = 2
        self.matches[1].found = True
        self.matches[1].quote = "변환부가 수신 신호를 변환한다."
        self.matches[1].chunk_id = "[0038]"
        self.matches[1].judgment = "실질적 동일"
        self.matches[1].directness = "direct"
        self.matches[1].missing_limitations = []
        self.matches[1].cited_invention_index = 1
        chain = {
            "total": [2, 1],
            "doc_name_mapping": {
                "0": "인용발명 3",
                "1": "인용발명 2",
                "2": "인용발명 1",
            },
            "combination_validity": {
                "coverage_complete": True,
                "remaining_uncovered_labels": [],
            },
        }
        chunks = [
            chunk
            async for chunk in generate_independent_phase1_streaming(
                self.claim,
                self.matches,
                [self.document, self.document, self.document],
                chain,
                Settings(),
            )
        ]
        report = "".join(chunks)

        self.assertIn("[종합 분석 요약]", report)
        self.assertNotIn("- 신규성 검토:", report)
        self.assertNotIn("- 잔여 차이 및 방어 포인트:", report)
        self.assertNotIn("- 진보성 검토:", report)
        self.assertNotIn("단일 인용발명에서 청구항의 모든 필수 구성", report)

    async def test_combination_review_field_and_motivation_dump_are_not_rendered(self):
        self.matches[1].motivation_quote = (
            "문제·개선 필요성에 관한 긴 발췌가 보고서에 나열되면 안 됩니다."
        )
        chain = {
            "total": [0, 1],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
            "combination_validity": {
                "coverage_complete": False,
                "remaining_uncovered_labels": ["B"],
            },
        }
        chunks = [
            chunk
            async for chunk in generate_independent_phase1_streaming(
                self.claim,
                self.matches,
                [self.document, self.document],
                chain,
                Settings(),
            )
        ]
        report = "".join(chunks)

        self.assertNotIn("- 결합 검토:", report)
        self.assertNotIn("문제·개선 필요성에 관한 긴 발췌", report)
        self.assertIn("[종합 분석 요약]", report)
        self.assertIn("현재 문헌 조합만으로 진보성 부정 근거를 구성하기 어려움", report)

    async def test_supplement_review_uses_translation_location_and_original_only(self):
        primary = self.matches[1]
        primary.cited_invention_index = 0
        support = ElementMatch(
            label="B",
            found=True,
            quote=(
                "The device 100 may be applied as a sound producing device which "
                "produces an acoustic sound according to an input (audio) signal S"
            ),
            quote_translation=(
                "장치(100)는 입력 오디오 신호 S에 따라 음향을 생성하는 "
                "음향 생성 장치로 적용될 수 있다."
            ),
            chunk_id="D2-P-0044",
            judgment="동일",
            cited_invention_index=1,
            directness="direct",
            missing_limitations=[],
        )
        chain = {
            "total": [0, 1],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
        }

        chunks = [
            chunk
            async for chunk in generate_independent_phase1_streaming(
                self.claim,
                [primary],
                [self.document, self.document],
                chain,
                Settings(),
                secondary_matches=[support],
            )
        ]
        report = "".join(chunks)

        self.assertIn(
            "인용발명 2 (prior.pdf) - 장치(100)는 입력 오디오 신호 S에 따라 음향을 "
            "생성하는 음향 생성 장치로 적용될 수 있다 [단락 0044] "
            '"The device 100 may be applied as a sound producing device which '
            'produces an acoustic sound according to an input (audio) signal S"',
            report,
        )
        self.assertNotIn("번역:", report)
        self.assertNotIn("발췌:", report)
        self.assertNotIn("판정 동일", report)
        self.assertNotIn("잔여 제한:", report)
        self.assertIn(
            "이는 주 인용발명에서 직접·완전하게 확인되지 않은 "
            '"수신 신호의 변환"을 직접 보완하는 기술적 근거이기 때문이다.',
            report,
        )

    async def test_korean_supplement_uses_excerpt_location_reason_order(self):
        primary = self.matches[1]
        primary.cited_invention_index = 0
        support = ElementMatch(
            label="B",
            found=True,
            quote="수신 신호를 변환하는 변환부를 포함한다.",
            quote_translation="",
            chunk_id="D2-P-0044",
            judgment="동일",
            cited_invention_index=1,
            directness="direct",
            missing_limitations=[],
        )
        chain = {
            "total": [0, 1],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
        }

        chunks = [
            chunk
            async for chunk in generate_independent_phase1_streaming(
                self.claim,
                [primary],
                [self.document, self.document],
                chain,
                Settings(),
                secondary_matches=[support],
            )
        ]
        report = "".join(chunks)

        self.assertIn(
            '인용발명 2 (prior.pdf) - "수신 신호를 변환하는 변환부를 포함한다." '
            '[단락 0044], 이는 주 인용발명에서 직접·완전하게 확인되지 않은 '
            '"수신 신호의 변환"을 직접 보완하는 기술적 근거이기 때문이다.',
            report,
        )


class NonPatentCitationLocationTests(unittest.TestCase):
    """비특허문헌 인용 위치는 청크 ID가 아니라 실제 페이지로 표기해야 한다."""

    def _paper(self) -> ExtractedDocument:
        return ExtractedDocument(
            filename="paper.pdf",
            document_type="non_patent",
            paragraph_chunks=[
                PatentChunk(
                    chunk_id="D1-P-P007",
                    section="METHOD",
                    page_no=4,
                    page_range=[4],
                    original_text="The encoder fuses point cloud and image features.",
                )
            ],
        )

    def test_chunk_id_is_resolved_to_page_number(self):
        match = ElementMatch(label="A", judgment="일부 차이", chunk_id="D1-P-P007")
        self.assertEqual(
            _format_citation_location(match, [self._paper()]), "(본문 4 페이지)"
        )

    def test_unknown_chunk_id_does_not_leak_internal_id(self):
        match = ElementMatch(label="A", judgment="일부 차이", chunk_id="D1-P-P999")
        location = _format_citation_location(match, [self._paper()])
        self.assertNotIn("D1-P-", location)
        self.assertEqual(location, "(본문 위치 미상)")

    def test_page_fallback_chunk_id_keeps_its_page(self):
        match = ElementMatch(label="A", judgment="일부 차이", chunk_id="[P12-C1]")
        doc = ExtractedDocument(filename="paper.pdf", document_type="non_patent")
        self.assertEqual(_format_citation_location(match, [doc]), "(본문 12 페이지)")


if __name__ == "__main__":
    unittest.main()
