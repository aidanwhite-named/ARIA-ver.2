from __future__ import annotations

import asyncio
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from backend.models.schemas import (
    ClaimElement,
    ChatMessage,
    ElementMatch,
    ExtractedDocument,
    ManualClaimRequest,
    ParagraphRecord,
    ParsedClaim,
    PatentChunk,
    Settings,
)
from backend.routers import analyze as analyze_router
from backend.services.ai_engine import (
    _restore_agy_truncated_response,
    _select_agy_response_candidate,
    _transcript_matches_prompt,
)
from backend.services.citation_extractor import (
    CompareFailed,
    _build_hybrid_docs_block,
    _parse_json_array,
    _select_best_matches,
    _shorten_quote,
    analyze_claim_elements_hybrid,
    verify_quotes,
)
from backend.services.citation_chain import (
    CITATION_CHAIN_POLICY_VERSION,
    _apply_conventional_support_policy,
    _conventionality_basis,
    build_citation_chain_from_comparisons,
    get_claim_chain_info,
)
from backend.services.reference_store import (
    save_case_artifacts_sqlite,
    save_reference_entries_sqlite,
)
from backend.services import gap_search
from backend.services.report_generator import (
    _build_phase2_markdown,
    _generate_template_b_phase2,
    _extract_first_json_object,
    _make_phase1_prompt,
    build_rejected_inventions_section,
    generate_dependent_phase2,
    parse_manual_claim_locally,
)


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _protobuf_text_field(field_number: int, text: bytes) -> bytes:
    return _varint((field_number << 3) | 2) + _varint(len(text)) + text


class AgyRecoveryTests(unittest.TestCase):
    def test_windows_prompt_path_matches_decoded_jsonl_content(self):
        marker = r"D:\develope\ARIA ver.2\uploads\_agy_prompts\prompt.txt"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "transcript.jsonl"
            path.write_text(
                json.dumps({"source": "USER_EXPLICIT", "content": f"Prompt file: {marker}"}),
                encoding="utf-8",
            )
            self.assertTrue(_transcript_matches_prompt(path, marker))

    def test_truncated_transcript_response_is_restored_from_conversation_db(self):
        full_response = json.dumps(
            [
                {
                    "label": "A",
                    "found": True,
                    "quote": "original passage " * 30,
                    "chunk_id": "[0036]",
                    "judgment": "실질적 동일",
                    "판단_이유": "대응 내용과 차이를 설명한다. " * 20,
                }
            ],
            ensure_ascii=False,
        )
        truncated = (
            full_response[:180].rstrip()
            + "\n<truncated 512 bytes>\n"
            + full_response[-180:].lstrip()
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            app_dir = Path(temp_dir)
            conversation_id = "conversation-1"
            transcript_path = (
                app_dir
                / "brain"
                / conversation_id
                / ".system_generated"
                / "logs"
                / "transcript.jsonl"
            )
            transcript_path.parent.mkdir(parents=True)
            db_path = app_dir / "conversations" / f"{conversation_id}.db"
            db_path.parent.mkdir(parents=True)

            nested = _protobuf_text_field(1, full_response.encode("utf-8"))
            payload = _protobuf_text_field(20, nested)
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute("CREATE TABLE steps (idx INTEGER, step_payload BLOB)")
                conn.execute("INSERT INTO steps VALUES (?, ?)", (1, payload))
                conn.commit()

            with patch("backend.services.ai_engine._agy_app_data_dir", return_value=app_dir):
                restored = _restore_agy_truncated_response(transcript_path, truncated)

        self.assertEqual(restored, full_response)
        self.assertEqual(json.loads(restored)[0]["label"], "A")


    def test_structured_final_response_wins_over_longer_reasoning(self):
        final_response = json.dumps([{"label": "A", "found": False}], ensure_ascii=False)
        reasoning = "**Analyzing Module Locations**\n" + ("internal reasoning " * 300)
        tool_payload = json.dumps({"CommandLine": "x" * 1000})

        selected = _select_agy_response_candidate(
            [final_response, reasoning, tool_payload, final_response, final_response[:20]]
        )

        self.assertEqual(selected, final_response)
        self.assertIsInstance(json.loads(selected), list)

class ComparisonParsingTests(unittest.TestCase):
    def test_judgments_and_reason_are_normalized_and_selected(self):
        elements = [ClaimElement(label="A", text="sensor"), ClaimElement(label="B", text="controller")]
        response = json.dumps(
            [
                {
                    "label": "(a)",
                    "found": "true",
                    "quote": "An event-based pixel array includes photosensitive devices.",
                    "chunk_id": "[0036]",
                    "judgment": "실질적 동일",
                    "판단_이유": "구조와 기능이 대응한다.",
                },
                {
                    "label": "B",
                    "found": False,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "없음",
                    "similarity_reason": "대응 기재가 없다.",
                },
            ],
            ensure_ascii=False,
        )

        parsed = _parse_json_array(response, elements)
        matches = _select_best_matches(elements, [parsed], 1)

        self.assertEqual([match.judgment for match in matches], ["실질적 동일", "대응 없음"])
        self.assertEqual(matches[0].similarity_reason, "구조와 기능이 대응한다.")

    def test_agy_alias_schema_without_quotes_is_rejected(self):
        elements = [ClaimElement(label="A", text="sensor")]
        response = json.dumps(
            [
                {
                    "doc_index": 0,
                    "claim_element": "A",
                    "found": True,
                    "judgment": "일부 유사",
                    "판단_이유": "관련 문단이 있다.",
                    "quote_start_line": 10,
                    "quote_end_line": 12,
                    "chunk_id": "[0001]",
                }
            ],
            ensure_ascii=False,
        )

        with self.assertRaisesRegex(CompareFailed, "필수 필드 누락"):
            _parse_json_array(response, elements, expected_doc_indices=[0])

    def test_hybrid_matrix_requires_every_document_and_label_pair(self):
        elements = [ClaimElement(label="A", text="sensor")]
        response = json.dumps(
            [
                {
                    "label": "A",
                    "doc_index": 0,
                    "found": False,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "대응 없음",
                    "판단_이유": "대응 기재가 없다.",
                }
            ],
            ensure_ascii=False,
        )

        with self.assertRaisesRegex(CompareFailed, "doc_index=1/label=A"):
            _parse_json_array(response, elements, expected_doc_indices=[0, 1])

    def test_quote_verification_handles_ellipsis_and_rejects_negative_doc_index(self):
        docs = [ExtractedDocument(raw_text="first relevant passage and second relevant passage")]
        valid = ElementMatch(
            label="A",
            quote="first relevant passage ... second relevant passage",
            cited_invention_index=0,
        )
        invalid_index = ElementMatch(
            label="B",
            quote="first relevant passage",
            cited_invention_index=-1,
        )

        results = verify_quotes([valid, invalid_index], docs)

        self.assertIn(results[0]["status"], {"verified", "partial"})
        self.assertEqual(results[1]["status"], "no_doc")

    def test_json_object_extraction_ignores_surrounding_text(self):
        response = 'explanation before\n```json\n{"purpose":"p","effects":"e"}\n```\nafter {"ignored":true}'
        self.assertEqual(
            _extract_first_json_object(response),
            {"purpose": "p", "effects": "e"},
        )

    def test_chat_gap_search_trigger_detects_missing_feature_search_intent(self):
        messages = [
            ChatMessage(
                role="user",
                content="보고서 작성 후 구성대비에 대응없는 구성들을 포함하는 발명을 검색해줄래?",
            )
        ]

        self.assertTrue(analyze_router._should_run_gap_search_from_chat(messages))

    def test_chat_gap_search_trigger_ignores_general_question(self):
        messages = [
            ChatMessage(
                role="user",
                content="구성 C가 왜 대응 없음으로 판단되었는지 설명해줘.",
            )
        ]

        self.assertFalse(analyze_router._should_run_gap_search_from_chat(messages))

    def test_gap_search_verification_merge_keeps_verified_documents(self):
        search_result = {
            "results": [
                {
                    "label": "C",
                    "documents": [
                        {"number": "US123", "title": "doc1", "relevance": "high"},
                        {"number": "US456", "title": "doc2", "relevance": "medium"},
                    ],
                }
            ]
        }
        verification = {
            "results": [
                {
                    "label": "C",
                    "documents": [
                        {
                            "number": "US123",
                            "verification_status": "direct",
                            "confidence": "high",
                            "reason": "직접 개시",
                            "quote": "example quote",
                        },
                        {
                            "number": "US456",
                            "verification_status": "unsupported",
                            "confidence": "low",
                            "reason": "불충분",
                            "quote": "",
                        },
                    ],
                }
            ]
        }

        merged = gap_search._merge_verification(search_result, verification)

        docs = merged["results"][0]["documents"]
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["number"], "US123")
        self.assertEqual(docs[0]["verification_status"], "direct")

    def test_gap_search_http_fallback_returns_patent_candidates(self):
        claim = ParsedClaim(
            claim_number=1,
            text="test claim",
            elements=[ClaimElement(label="C", text="센서 신호를 보정하는 처리부", importance="5")],
            preamble="센서 제어 장치",
        )
        gap_result = {
            "uncovered": [
                {
                    "label": "C",
                    "text": "센서 신호를 보정하는 처리부",
                    "importance": "5",
                    "best_judgment": "없음",
                    "best_doc": "",
                }
            ]
        }

        async def fake_call_ai(*args, **kwargs):
            raise RuntimeError("web search tool unavailable")

        async def fake_search_target_documents(target, field_text):
            return (
                ["site:patents.google.com sensor correction patent"],
                [
                    {
                        "title": "Example patent",
                        "number": "US1234567A",
                        "url": "https://patents.google.com/patent/US1234567A/en",
                        "summary": "fallback result",
                        "relevance": "medium",
                        "source": "http_fallback",
                    }
                ],
            )

        with patch.object(gap_search, "call_ai", side_effect=fake_call_ai), patch.object(
            gap_search,
            "_search_target_documents",
            side_effect=fake_search_target_documents,
        ):
            result = __import__("asyncio").run(
                gap_search.web_search_gap_documents(claim, gap_result, Settings())
            )

        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["results"][0]["label"], "C")
        self.assertEqual(result["results"][0]["documents"][0]["number"], "US1234567A")


    def test_missing_parent_reference_keeps_only_features_after_dependency_phrase(self):
        claim = __import__("asyncio").run(
            parse_manual_claim_locally(
                "제99항에 있어서, 센서가 신호를 검출하는 단계; 검출된 신호를 저장하는 단계.",
                2,
                "dependent",
                None,
            )
        )

        self.assertEqual(claim.parent_claim, 99)
        self.assertEqual(
            [element.text for element in claim.elements],
            ["센서가 신호를 검출하는 단계", "검출된 신호를 저장하는 단계."],
        )
        self.assertTrue(all("제99항" not in element.text for element in claim.elements))


    def test_self_parent_reference_is_not_inferred(self):
        claim = __import__("asyncio").run(
            parse_manual_claim_locally(
                "제1 항에 있어서, 추가 센서 특징.",
                1,
                "dependent",
                None,
            )
        )

        self.assertIsNone(claim.parent_claim)


class ManualClaimRegistrationTests(unittest.TestCase):
    def test_changed_claim_invalidates_its_comparisons_and_job_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uploads = root / "uploads"
            reports = root / "reports"
            cases = root / "cases"
            job_id = "JOB-1"
            job_dir = uploads / job_id
            case_reports = cases / job_id / "reports"
            job_dir.mkdir(parents=True)
            reports.mkdir(parents=True)
            case_reports.mkdir(parents=True)

            claims = [
                ParsedClaim(claim_number=1, text="parent claim").model_dump(),
                ParsedClaim(
                    claim_number=2,
                    claim_type="dependent",
                    parent_claim=1,
                    text="old child claim",
                    elements=[ClaimElement(label="A", text="old feature")],
                ).model_dump(),
            ]
            (job_dir / "claims.json").write_text(
                json.dumps(claims, ensure_ascii=False), encoding="utf-8"
            )
            (job_dir / "comparisons_0.json").write_text(
                json.dumps({"1": [{"label": "A"}], "2": [{"label": "A"}], "_meta": {}}),
                encoding="utf-8",
            )
            for name in ("citation_chain.json", "same_pairs.json", "context.json"):
                (job_dir / name).write_text("{}", encoding="utf-8")
            (reports / f"report_{job_id}_claim1.md").write_text("old", encoding="utf-8")
            (reports / f"report_{job_id}_claim2.md").write_text("old", encoding="utf-8")
            (case_reports / "claim2.md").write_text("old", encoding="utf-8")

            with (
                patch.object(analyze_router, "UPLOADS_DIR", uploads),
                patch.object(analyze_router, "REPORTS_DIR", reports),
                patch.object(analyze_router, "CASES_DIR", cases),
            ):
                result = __import__("asyncio").run(
                    analyze_router.manual_claim(
                        job_id,
                        ManualClaimRequest(
                            claim_text="제1 항에 있어서, 새로운 센서 특징.",
                            claim_number=2,
                            claim_type="dependent",
                        ),
                    )
                )

            cache = json.loads((job_dir / "comparisons_0.json").read_text(encoding="utf-8"))
            self.assertIn("1", cache)
            self.assertNotIn("2", cache)
            self.assertEqual(result["parent_claim"], 1)
            self.assertFalse((job_dir / "citation_chain.json").exists())
            self.assertFalse((job_dir / "context.json").exists())
            self.assertFalse(list(reports.glob(f"report_{job_id}_claim*.*")))
            self.assertFalse((case_reports / "claim2.md").exists())

    def test_enhanced_claim_invalidates_comparison_and_report_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            uploads = root / "uploads"
            reports = root / "reports"
            cases = root / "cases"
            job_id = "JOB-ENHANCE"
            job_dir = uploads / job_id
            job_dir.mkdir(parents=True)
            reports.mkdir(parents=True)

            original = ParsedClaim(
                claim_number=1,
                text="original",
                elements=[ClaimElement(label="A", text="old feature")],
            )
            enhanced = original.model_copy(
                update={"elements": [ClaimElement(label="A", text="new feature")]}
            )
            (job_dir / "claims.json").write_text(
                json.dumps([original.model_dump()], ensure_ascii=False),
                encoding="utf-8",
            )
            (job_dir / "comparisons_0.json").write_text(
                json.dumps({"1": [{"label": "A"}], "_meta": {}}),
                encoding="utf-8",
            )
            (job_dir / "citation_chain.json").write_text("{}", encoding="utf-8")
            (reports / f"report_{job_id}_claim1.md").write_text("old", encoding="utf-8")

            with (
                patch.object(analyze_router, "UPLOADS_DIR", uploads),
                patch.object(analyze_router, "REPORTS_DIR", reports),
                patch.object(analyze_router, "CASES_DIR", cases),
                patch.object(analyze_router, "_load_settings_with_dir", return_value=Settings()),
                patch.object(
                    analyze_router,
                    "enhance_claim_parsing_with_llm",
                    new=AsyncMock(return_value=enhanced),
                ),
            ):
                result = __import__("asyncio").run(analyze_router.enhance_claim(job_id, 1))

            cache = json.loads((job_dir / "comparisons_0.json").read_text(encoding="utf-8"))
            case_claims = json.loads(
                (cases / job_id / "parsed" / "claims.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("1", cache)
            self.assertFalse((job_dir / "citation_chain.json").exists())
            self.assertFalse((reports / f"report_{job_id}_claim1.md").exists())
            self.assertEqual(result["elements"][0]["text"], "new feature")
            self.assertEqual(case_claims[0]["elements"][0]["text"], "new feature")


class IntegratedComparisonTests(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_mode_compares_all_documents_in_one_llm_call(self):
        elements = [ClaimElement(label="A", text="sensor")]
        docs = [
            ExtractedDocument(filename="first.pdf", raw_text="[T1] first sensor passage"),
            ExtractedDocument(filename="second.pdf", raw_text="[T1] second sensor passage"),
        ]
        response = json.dumps(
            [
                {
                    "label": "A",
                    "doc_index": 0,
                    "found": True,
                    "quote": "first sensor passage",
                    "chunk_id": "[T1]",
                    "judgment": "실질적 동일",
                    "판단_이유": "첫 번째 문헌의 센서가 대응한다.",
                },
                {
                    "label": "A",
                    "doc_index": 1,
                    "found": True,
                    "quote": "second sensor passage",
                    "chunk_id": "[T1]",
                    "judgment": "일부 유사",
                    "판단_이유": "두 번째 문헌에도 관련 센서가 있다.",
                },
            ],
            ensure_ascii=False,
        )
        settings = Settings(
            engine="claude",
            comparison_mode="hybrid",
            use_rag_retrieval=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "backend.services.citation_extractor.call_ai",
                new_callable=AsyncMock,
                return_value=response,
            ) as mocked_call:
                await analyze_claim_elements_hybrid(
                    elements,
                    docs,
                    settings,
                    job_dir=temp_dir,
                    claim_number=1,
                )

            mocked_call.assert_awaited_once()
            prompt = mocked_call.await_args.args[0]
            self.assertIn("[doc_index=0] first.pdf", prompt)
            self.assertIn("[doc_index=1] second.pdf", prompt)
            for doc_idx in range(2):
                cache = json.loads(
                    (Path(temp_dir) / f"comparisons_{doc_idx}.json").read_text(encoding="utf-8")
                )
                self.assertEqual(cache["_meta"]["comparison_mode"], "hybrid")
                self.assertIn("1", cache)

    async def test_invalid_hybrid_schema_fails_without_retry(self):
        elements = [ClaimElement(label="A", text="sensor")]
        docs = [
            ExtractedDocument(filename="first.pdf", raw_text="[T1] first sensor passage"),
            ExtractedDocument(filename="second.pdf", raw_text="[T1] second sensor passage"),
        ]
        invalid_response = json.dumps(
            [{"doc_index": 0, "claim_element": "A", "found": True, "judgment": "일부 유사"}],
            ensure_ascii=False,
        )
        settings = Settings(engine="agy", comparison_mode="hybrid", use_rag_retrieval=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "backend.services.citation_extractor.call_ai",
                new_callable=AsyncMock,
                return_value=invalid_response,
            ) as mocked_call:
                with self.assertRaises(CompareFailed):
                    await analyze_claim_elements_hybrid(
                        elements, docs, settings, job_dir=temp_dir, claim_number=1
                    )

            mocked_call.assert_awaited_once()
            self.assertFalse(list(Path(temp_dir).glob("comparisons_*.json")))
    async def test_hybrid_failure_does_not_fall_back_to_per_document(self):
        elements = [ClaimElement(label="A", text="sensor")]
        docs = [
            ExtractedDocument(filename="first.pdf", raw_text="[T1] first sensor passage"),
            ExtractedDocument(filename="second.pdf", raw_text="[T1] unrelated passage"),
        ]
        invalid_response = "internal analysis without a JSON array"
        settings = Settings(engine="agy", comparison_mode="hybrid", use_rag_retrieval=False)

        with patch(
            "backend.services.citation_extractor.call_ai",
            new_callable=AsyncMock,
            return_value=invalid_response,
        ) as mocked_call:
            with self.assertRaises(CompareFailed):
                await analyze_claim_elements_hybrid(elements, docs, settings)

        mocked_call.assert_awaited_once()
    async def test_oversized_hybrid_context_keeps_every_document(self):
        elements = [ClaimElement(label="A", text="needle")]
        docs = [
            ExtractedDocument(filename=f"doc-{idx}.pdf", raw_text=(chr(65 + idx) * 12_000))
            for idx in range(7)
        ]
        settings = Settings(
            engine="claude",
            comparison_mode="hybrid",
            use_rag_retrieval=False,
        )

        block = _build_hybrid_docs_block(docs, elements, settings=settings)

        self.assertEqual(block.count("[doc_index="), len(docs))
        for idx, doc in enumerate(docs):
            self.assertIn(f"[doc_index={idx}] {doc.filename}", block)


class ConventionalSupportPolicyTests(unittest.TestCase):
    @staticmethod
    def _item(label: str, judgment: str, quote: str = "") -> dict:
        return {
            "label": label,
            "judgment": judgment,
            "quote": quote,
            "chunk_id": "[0001]" if quote else "",
        }

    def test_third_reference_is_admitted_only_for_residual_conventional_element(self):
        claim = ParsedClaim(
            claim_number=1,
            text="camera claim",
            elements=[
                ClaimElement(label="A", text="촬상 모듈", importance="5"),
                ClaimElement(label="B", text="적응형 피드백 처리 회로", importance="3"),
                ClaimElement(label="C", text="본체를 이동시키는 바퀴", importance="2"),
            ],
        )
        caches = {
            0: {"1": [
                self._item("A", "동일", "camera"),
                self._item("B", "대응 없음"),
                self._item("C", "대응 없음"),
            ]},
            1: {"1": [
                self._item("A", "일부 유사", "camera field"),
                self._item("B", "실질적 동일", "adaptive feedback circuit"),
                self._item("C", "대응 없음"),
            ]},
            2: {"1": [
                self._item("A", "대응 없음"),
                self._item("B", "대응 없음"),
                self._item("C", "동일", "a wheel attached to the body"),
            ]},
        }
        chains = {"1": {"total": [0, 1], "inherited": [], "added": [0, 1], "parent": None}}
        weights = {("1", "A"): 5, ("1", "B"): 3, ("1", "C"): 2}

        _apply_conventional_support_policy(chains, [claim], caches, 3, weights)

        self.assertEqual(chains["1"]["total"], [0, 1, 2])
        self.assertEqual(chains["1"]["reference_roles"]["2"], "conventional_support")
        self.assertEqual(chains["1"]["conventional_support"]["position"], 3)
        self.assertEqual(chains["1"]["conventional_support"]["labels"], ["C"])

    def test_weak_second_reference_does_not_unlock_third_reference_exception(self):
        claim = ParsedClaim(
            claim_number=1,
            text="camera claim",
            elements=[
                ClaimElement(label="A", text="촬상 모듈", importance="5"),
                ClaimElement(label="B", text="적응형 피드백 처리 회로", importance="3"),
                ClaimElement(label="C", text="바퀴", importance="2"),
            ],
        )
        caches = {
            0: {"1": [
                self._item("A", "동일", "camera"),
                self._item("B", "대응 없음"),
                self._item("C", "대응 없음"),
            ]},
            1: {"1": [
                self._item("A", "일부 유사", "camera field"),
                self._item("B", "일부 유사", "vague feedback"),
                self._item("C", "대응 없음"),
            ]},
            2: {"1": [
                self._item("A", "대응 없음"),
                self._item("B", "대응 없음"),
                self._item("C", "동일", "wheel"),
            ]},
        }
        chains = {"1": {"total": [0, 1], "inherited": [], "added": [0, 1], "parent": None}}
        weights = {("1", "A"): 5, ("1", "B"): 3, ("1", "C"): 2}

        _apply_conventional_support_policy(chains, [claim], caches, 3, weights)

        self.assertEqual(chains["1"]["total"], [0, 2])
        self.assertEqual(chains["1"]["conventional_support"]["position"], 2)
        self.assertNotIn(1, chains["1"]["total"])

    def test_single_reference_uses_common_knowledge_when_document_support_is_weak(self):
        claim = ParsedClaim(
            claim_number=1,
            text="vehicle claim",
            elements=[
                ClaimElement(label="A", text="특수 구동 모듈", importance="5"),
                ClaimElement(label="B", text="바퀴", importance="2"),
            ],
        )
        caches = {
            0: {"1": [self._item("A", "동일", "drive"), self._item("B", "대응 없음")]},
            1: {"1": [self._item("A", "대응 없음"), self._item("B", "일부 유사", "round member")]},
        }
        chains = {"1": {"total": [0, 1], "inherited": [], "added": [0, 1], "parent": None}}
        weights = {("1", "A"): 5, ("1", "B"): 2}

        _apply_conventional_support_policy(chains, [claim], caches, 2, weights)

        self.assertEqual(chains["1"]["total"], [0])
        self.assertEqual(chains["1"]["common_general_knowledge"][0]["label"], "B")

    def test_strong_document_support_is_labeled_as_conventional_evidence(self):
        claim = ParsedClaim(
            claim_number=1,
            text="vehicle claim",
            elements=[
                ClaimElement(label="A", text="특수 구동 모듈", importance="5"),
                ClaimElement(label="B", text="바퀴", importance="2"),
            ],
        )
        caches = {
            0: {"1": [self._item("A", "동일", "drive"), self._item("B", "대응 없음")]},
            1: {"1": [self._item("A", "대응 없음"), self._item("B", "동일", "wheel mounted to body")]},
        }
        chains = {"1": {"total": [0, 1], "inherited": [], "added": [0, 1], "parent": None}}
        weights = {("1", "A"): 5, ("1", "B"): 2}

        _apply_conventional_support_policy(chains, [claim], caches, 2, weights)

        self.assertEqual(chains["1"]["total"], [0, 1])
        self.assertEqual(chains["1"]["reference_roles"]["1"], "conventional_support")
        self.assertEqual(chains["1"]["conventional_support"]["position"], 2)

    def test_specialized_controller_is_not_treated_as_conventional(self):
        element = ClaimElement(
            label="C",
            text="전역 피드백 신호에 기초하여 메모리 저장을 제어하는 제어부",
            importance="2",
        )
        self.assertIsNone(_conventionality_basis(element))

    def test_phase1_prompt_marks_third_document_role(self):
        claim = ParsedClaim(
            claim_number=1,
            text="claim",
            elements=[ClaimElement(label="C", text="바퀴", importance="2")],
        )
        docs = [
            ExtractedDocument(filename="primary.pdf"),
            ExtractedDocument(filename="secondary.pdf"),
            ExtractedDocument(filename="conventional.pdf"),
        ]
        matches = [ElementMatch(label="C", cited_invention_index=2, judgment="동일", quote="wheel")]
        chain_info = {
            "total": [0, 1, 2],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2", "2": "인용발명 3"},
            "combination_rationale": {"label": "공백 보완형", "description": "핵심 차이 보완"},
            "conventional_support": {
                "doc_idx": 2,
                "position": 3,
                "role": "conventional_support",
                "labels": ["C"],
            },
        }

        prompt = _make_phase1_prompt(
            claim,
            matches,
            docs,
            chain_info,
            Settings(),
            combo=True,
            secondary_matches=matches,
        )

        self.assertIn("인용발명 3 - 주지관용 구성 입증자료", prompt)
        self.assertIn("핵심 기술사상 보완 근거로 확대하지 않습니다", prompt)

        phase2 = _build_phase2_markdown(
            "### [구성요소 (C)] 유사도: 동일 95%\n\n- 청구항 구성: 바퀴\n\n"
            "### 종합 분석 요약\n\n- 유사점 요약: 일반 구성\n- 차이점: 없음\n- 결론: 검토 완료",
            1,
            "인용발명 1",
            "인용발명 2",
            "인용발명 3",
            is_combo=True,
            combination_rationale="제3문헌은 주지관용 입증자료로만 사용",
            settings=Settings(),
        )
        self.assertIn("[인용발명 3 - 주지관용 구성 입증자료]", phase2)
        self.assertIn("[구성대비]", phase2)
        self.assertIn("[종합 판단]", phase2)

    def test_combo_phase2_components_use_primary_matches_only(self):
        claim = ParsedClaim(
            claim_number=1,
            text="claim",
            elements=[
                ClaimElement(label="A", text="primary feature", importance="5"),
                ClaimElement(label="B", text="secondary-only feature", importance="3"),
            ],
        )
        docs = [
            ExtractedDocument(filename="primary.pdf"),
            ExtractedDocument(filename="secondary.pdf"),
        ]
        matches = [
            ElementMatch(label="A", cited_invention_index=0, judgment="동일", quote="primary quote", chunk_id="[0001]"),
            ElementMatch(label="B", cited_invention_index=1, judgment="대응 없음", quote="secondary quote", chunk_id="[0002]"),
        ]
        chain_info = {
            "total": [0, 1],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
            "combination_rationale": {"label": "결합", "description": "보조 문헌"},
        }
        phase1_md = (
            "### [구성요소 (A)] 유사도: 동일 95%\n\n- 인용발명 대응 원문: primary quote\n\n"
            "### [구성요소 (B)] 유사도: 대응 없음 0%\n\n- 인용발명 대응 원문: (인용발명 1에서 해당 구성 확인 불가)\n\n"
            "### 종합 분석 요약\n\n- 유사점 요약: A는 직접 대응됨\n- 차이점: B는 인용발명 2에서만 확인됨\n- 결론: 검토 완료"
        )

        phase2 = asyncio.run(
            _generate_template_b_phase2(phase1_md, claim, matches, docs, chain_info, Settings())
        )

        self.assertIn("(A) 동일 95%", phase2)
        self.assertIn("primary quote", phase2)
        self.assertIn("(B) 대응 없음 0%", phase2)
        self.assertNotIn("(A) 동일 (인용발명 1)", phase2)
        self.assertNotIn("secondary quote (인용발명 2)", phase2)
        self.assertLess(phase2.index("[구성대비]"), phase2.index("[종합 판단]"))

    def test_combo_phase2_keeps_difference_separate_from_combination_rationale(self):
        claim = ParsedClaim(
            claim_number=10,
            text="claim",
            elements=[ClaimElement(label="A", text="feature", importance="5")],
        )
        docs = [
            ExtractedDocument(filename="primary.pdf"),
            ExtractedDocument(filename="secondary.pdf"),
        ]
        chain_info = {
            "total": [0, 1],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
            "combination_rationale": {"label": "결합 논리", "description": "보조 설명"},
        }
        phase1_md = (
            "### [구성요소 (A)] 유사도: 일부 유사 80%\n\n"
            "- 인용발명 대응 원문: primary quote\n\n"
            "### 종합 분석 요약\n\n"
            "- 유사점 요약: 일부 구성이 대응됨\n"
            "- 차이점: [[차이점 1]] 추가 구성 확인 필요\n"
            "- 결합 논리 및 차이점 요약: [차이점 1] 인용발명 2의 보조 구성이 추가로 필요함\n"
            "- 결론: 검토 필요"
        )

        phase2 = asyncio.run(
            _generate_template_b_phase2(phase1_md, claim, [], docs, chain_info, Settings())
        )

        self.assertIn("[차이점]", phase2)
        self.assertIn("[차이점 1] 추가 구성 확인 필요", phase2)
        self.assertNotIn("[결합 논리]", phase2)
        self.assertNotIn("인용발명 2의 보조 구성이 추가로 필요함", phase2)
        self.assertNotIn("[[차이점 1]]", phase2)

    def test_second_conventional_document_gets_limited_rationale(self):
        chain_data = {
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
            "chains": {
                "1": {
                    "total": [0, 1],
                    "conventional_support": {
                        "doc_idx": 1,
                        "position": 2,
                        "role": "conventional_support",
                        "labels": ["B"],
                    },
                }
            },
        }

        info = get_claim_chain_info(chain_data, 1)

        self.assertEqual(info["combination_rationale_type"], "conventional_support")
        self.assertEqual(info["combination_rationale"]["label"], "주지관용 구성 문헌 보강형")

    def test_full_chain_build_persists_exceptional_third_reference_role(self):
        claim = ParsedClaim(
            claim_number=1,
            text="camera claim",
            elements=[
                ClaimElement(label="A", text="특수 촬상 모듈", importance="5"),
                ClaimElement(label="B", text="적응형 피드백 처리 회로", importance="3"),
                ClaimElement(label="C", text="바퀴", importance="2"),
            ],
        )
        caches = [
            {"1": [
                self._item("A", "동일", "special camera"),
                self._item("B", "대응 없음"),
                self._item("C", "대응 없음"),
            ]},
            {"1": [
                self._item("A", "대응 없음"),
                self._item("B", "동일", "adaptive feedback"),
                self._item("C", "대응 없음"),
            ]},
            {"1": [
                self._item("A", "대응 없음"),
                self._item("B", "대응 없음"),
                self._item("C", "동일", "wheel"),
            ]},
        ]
        docs = [ExtractedDocument(filename=f"doc-{idx}.pdf") for idx in range(3)]

        with tempfile.TemporaryDirectory() as temp_dir:
            for idx, cache in enumerate(caches):
                (Path(temp_dir) / f"comparisons_{idx}.json").write_text(
                    json.dumps(cache, ensure_ascii=False),
                    encoding="utf-8",
                )
            result = build_citation_chain_from_comparisons(temp_dir, [claim], docs)

        chain = result["chains"]["1"]
        self.assertEqual(len(chain["total"]), 3)
        self.assertEqual(chain["reference_roles"][str(chain["total"][2])], "conventional_support")
        self.assertEqual(result["policy_version"], CITATION_CHAIN_POLICY_VERSION)

    def test_related_document_without_gap_evidence_is_not_adopted_as_secondary(self):
        claim = ParsedClaim(
            claim_number=1,
            text="image sensor claim",
            elements=[
                ClaimElement(label="A", text="특수 촬상 모듈", importance="5"),
                ClaimElement(
                    label="D",
                    text="포토 다이오드가 신호 배선을 통해 로우 드라이버에 연결된 회로",
                    importance="2",
                ),
            ],
        )
        caches = [
            {"1": [
                self._item("A", "실질적 동일", "autofocus image sensor"),
                self._item("D", "대응 없음"),
            ]},
            {"1": [
                self._item("A", "실질적 동일", "related image sensor"),
                self._item("D", "대응 없음"),
            ]},
            {"1": [
                self._item("A", "일부 유사", "pixel array"),
                self._item("D", "대응 없음"),
            ]},
        ]
        docs = [ExtractedDocument(filename=f"doc-{idx}.pdf") for idx in range(3)]

        with tempfile.TemporaryDirectory() as temp_dir:
            for idx, cache in enumerate(caches):
                (Path(temp_dir) / f"comparisons_{idx}.json").write_text(
                    json.dumps(cache, ensure_ascii=False),
                    encoding="utf-8",
                )
            result = build_citation_chain_from_comparisons(temp_dir, [claim], docs)

        self.assertEqual(result["chains"]["1"]["total"], [result["primary_inv_idx"]])
        self.assertEqual(result["combination_rationale_type"], "insufficient_support")
        self.assertEqual(result["confidence"]["1"]["uncovered_labels"], ["D"])


class RejectedInventionsSectionTests(unittest.TestCase):
    def test_rejected_inventions_are_rendered_as_related_a_summary(self):
        claim = ParsedClaim(
            claim_number=1,
            text="vehicle claim",
            elements=[
                ClaimElement(label="A", text="sensor module", importance="5"),
                ClaimElement(label="B", text="control unit", importance="4"),
                ClaimElement(label="C", text="display", importance="3"),
            ],
        )
        docs = [
            ExtractedDocument(filename="primary.pdf"),
            ExtractedDocument(filename="related-a.pdf"),
        ]
        chain_info = {
            "total": [0],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
        }
        cache = {
            "1": [
                {
                    "label": "A",
                    "found": True,
                    "quote": "sensor arranged on a vehicle body",
                    "chunk_id": "[0001]",
                    "judgment": "동일",
                    "similarity_reason": "차량 본체에 센서를 배치하는 구성은 청구항과 동일합니다.",
                },
                {
                    "label": "B",
                    "found": True,
                    "quote": "controller transmits a control signal",
                    "chunk_id": "[0002]",
                    "judgment": "일부 유사",
                    "similarity_reason": "제어 신호를 생성하는 점은 유사하지만 세부 제어 방식은 다릅니다.",
                },
                {
                    "label": "C",
                    "found": False,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "대응 없음",
                    "similarity_reason": "",
                },
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "comparisons_1.json").write_text(
                json.dumps(cache, ensure_ascii=False),
                encoding="utf-8",
            )
            result = build_rejected_inventions_section(claim, docs, chain_info, temp_dir)

        self.assertIn("## 관련도 A 인용발명", result)
        self.assertIn("인용발명 2", result)
        self.assertIn("(A) 차량 본체에 센서를 배치하는 구성은 청구항과 동일합니다. (sensor arranged on a vehicle body [0001])", result)
        self.assertIn("(B) 제어 신호를 생성하는 점은 유사하지만 세부 제어 방식은 다릅니다. (controller transmits a control signal [0002])", result)
        self.assertIn("차이점: (C) 구성은 이 인용발명에서 직접 확인되지 않아 최종 채택에서 제외되었습니다.", result)

    def test_rejected_inventions_section_groups_remaining_docs_under_single_a_heading(self):
        claim = ParsedClaim(
            claim_number=1,
            claim_type="independent",
            text="청구항 1. 장치.",
            elements=[
                ClaimElement(label="A", text="sensor module", importance="5"),
            ],
        )
        docs = [
            ExtractedDocument(filename="primary.pdf"),
            ExtractedDocument(filename="secondary.pdf"),
            ExtractedDocument(filename="tertiary.pdf"),
        ]
        chain_info = {
            "total": [0],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2", "2": "인용발명 3"},
        }
        cache1 = {
            "1": [
                {
                    "label": "A",
                    "found": True,
                    "quote": "sensor arrangement",
                    "chunk_id": "[0001]",
                    "judgment": "동일",
                    "similarity_reason": "센서 배치 구성이 청구항과 동일합니다.",
                }
            ]
        }
        cache2 = {
            "1": [
                {
                    "label": "A",
                    "found": True,
                    "quote": "auxiliary sensor layout",
                    "chunk_id": "[0002]",
                    "judgment": "실질적 동일",
                    "similarity_reason": "보조 센서 배치 구성이 청구항과 실질적으로 동일합니다.",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "comparisons_1.json").write_text(
                json.dumps(cache1, ensure_ascii=False),
                encoding="utf-8",
            )
            (Path(temp_dir) / "comparisons_2.json").write_text(
                json.dumps(cache2, ensure_ascii=False),
                encoding="utf-8",
            )
            result = build_rejected_inventions_section(claim, docs, chain_info, temp_dir)

        self.assertEqual(result.count("## 관련도 A 인용발명"), 1)
        self.assertNotIn("## 관련도 B 인용발명", result)
        self.assertNotIn("## 관련도 C 인용발명", result)
        self.assertIn("**인용발명 2** (secondary.pdf)", result)
        self.assertIn("**인용발명 3** (tertiary.pdf)", result)
        self.assertEqual(result.count("**인용발명 "), 2)

    def test_rejected_inventions_section_groups_multiple_missing_labels_into_one_difference_line(self):
        claim = ParsedClaim(
            claim_number=1,
            claim_type="independent",
            text="청구항 1. 장치.",
            elements=[
                ClaimElement(label="A", text="sensor module", importance="5"),
                ClaimElement(label="B", text="controller", importance="4"),
                ClaimElement(label="C", text="display", importance="3"),
            ],
        )
        docs = [
            ExtractedDocument(filename="primary.pdf"),
            ExtractedDocument(filename="secondary.pdf"),
        ]
        chain_info = {
            "total": [0],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
        }
        cache = {
            "1": [
                {
                    "label": "A",
                    "found": True,
                    "quote": "sensor arrangement",
                    "chunk_id": "[0001]",
                    "judgment": "동일",
                    "similarity_reason": "센서 배치 구성이 청구항과 동일합니다.",
                },
                {
                    "label": "B",
                    "found": False,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "대응 없음",
                    "similarity_reason": "",
                },
                {
                    "label": "C",
                    "found": False,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "대응 없음",
                    "similarity_reason": "",
                },
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "comparisons_1.json").write_text(
                json.dumps(cache, ensure_ascii=False),
                encoding="utf-8",
            )
            result = build_rejected_inventions_section(claim, docs, chain_info, temp_dir)

        self.assertIn("차이점: (B), (C) 구성은 이 인용발명에서 직접 확인되지 않아 최종 채택에서 제외되었습니다.", result)


class DependentCitationChainPolicyTests(unittest.TestCase):
    @staticmethod
    def _item(label: str, judgment: str, quote: str = "") -> dict:
        return {
            "label": label,
            "found": bool(quote),
            "quote": quote,
            "chunk_id": "[0001]" if quote else "",
            "judgment": judgment,
        }

    def _write_caches_and_build(self, claims, caches):
        docs = [ExtractedDocument(filename=f"doc-{idx}.pdf") for idx in range(len(caches))]
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx, cache in enumerate(caches):
                (Path(temp_dir) / f"comparisons_{idx}.json").write_text(
                    json.dumps(cache, ensure_ascii=False),
                    encoding="utf-8",
                )
            return build_citation_chain_from_comparisons(temp_dir, claims, docs)

    def test_missing_parent_claim_uses_only_child_feature_evidence(self):
        claims = [
            ParsedClaim(
                claim_number=2,
                claim_type="dependent",
                parent_claim=99,
                text="제99항에 있어서, 추가 센서 특징",
                elements=[ClaimElement(label="A", text="추가 센서 특징", importance="5")],
            )
        ]
        caches = [
            {"2": [self._item("A", "동일", "matching sensor feature")]},
            {"2": [self._item("A", "대응 없음")]},
        ]

        result = self._write_caches_and_build(claims, caches)
        chain = result["chains"]["2"]

        self.assertFalse(chain["parent_available"])
        self.assertEqual(chain["inherited"], [])
        self.assertEqual(chain["added"], [0])
        self.assertEqual(chain["total"], [0])
        self.assertTrue(chain["coverage_complete"])

    def test_nested_and_sibling_claims_inherit_parent_and_add_one_reference(self):
        claims = [
            ParsedClaim(
                claim_number=1,
                text="independent claim",
                elements=[
                    ClaimElement(label="IA", text="특수 광학 모듈", importance="5"),
                    ClaimElement(label="IB", text="적응형 처리 회로", importance="5"),
                ],
            ),
            ParsedClaim(
                claim_number=2,
                claim_type="dependent",
                parent_claim=1,
                text="claim 2",
                elements=[ClaimElement(label="C", text="추가 구성 C", importance="5")],
            ),
            ParsedClaim(
                claim_number=3,
                claim_type="dependent",
                parent_claim=2,
                text="claim 3",
                elements=[ClaimElement(label="D", text="추가 구성 D", importance="5")],
            ),
            ParsedClaim(
                claim_number=4,
                claim_type="dependent",
                parent_claim=1,
                text="claim 4",
                elements=[ClaimElement(label="E", text="추가 구성 E", importance="5")],
            ),
        ]
        no_dep = {
            "2": [self._item("C", "대응 없음")],
            "3": [self._item("D", "대응 없음")],
            "4": [self._item("E", "대응 없음")],
        }
        caches = [
            {
                "1": [self._item("IA", "동일", "optical"), self._item("IB", "대응 없음")],
                **no_dep,
            },
            {
                "1": [self._item("IA", "대응 없음"), self._item("IB", "동일", "adaptive")],
                **no_dep,
            },
            {
                "1": [self._item("IA", "대응 없음"), self._item("IB", "대응 없음")],
                "2": [self._item("C", "동일", "feature C")],
                "3": [self._item("D", "대응 없음")],
                "4": [self._item("E", "대응 없음")],
            },
            {
                "1": [self._item("IA", "대응 없음"), self._item("IB", "대응 없음")],
                "2": [self._item("C", "대응 없음")],
                "3": [self._item("D", "동일", "feature D")],
                "4": [self._item("E", "대응 없음")],
            },
            {
                "1": [self._item("IA", "대응 없음"), self._item("IB", "대응 없음")],
                "2": [self._item("C", "대응 없음")],
                "3": [self._item("D", "대응 없음")],
                "4": [self._item("E", "동일", "feature E")],
            },
        ]

        result = self._write_caches_and_build(claims, caches)

        independent_total = result["chains"]["1"]["total"]
        self.assertEqual(set(independent_total), {0, 1})
        self.assertEqual(result["chains"]["2"]["total"], independent_total + [2])
        self.assertEqual(result["chains"]["2"]["added"], [2])
        self.assertEqual(result["chains"]["3"]["total"], independent_total + [2, 3])
        self.assertEqual(result["chains"]["3"]["added"], [3])
        self.assertEqual(result["chains"]["4"]["total"], independent_total + [4])
        self.assertEqual(result["chains"]["4"]["added"], [4])
        mapping = result["doc_name_mapping"]
        self.assertEqual(mapping[str(independent_total[0])], "인용발명 1")
        self.assertEqual(mapping[str(independent_total[1])], "인용발명 2")
        self.assertEqual(mapping["2"], "인용발명 3")
        self.assertEqual(mapping["3"], "인용발명 4")
        self.assertEqual(mapping["4"], "인용발명 5")

    def test_two_new_references_are_not_combined_for_one_dependent_claim(self):
        claims = [
            ParsedClaim(
                claim_number=1,
                text="independent claim",
                elements=[
                    ClaimElement(label="IA", text="특수 광학 모듈", importance="5"),
                    ClaimElement(label="IB", text="적응형 처리 회로", importance="5"),
                ],
            ),
            ParsedClaim(
                claim_number=2,
                claim_type="dependent",
                parent_claim=1,
                text="claim 2",
                elements=[
                    ClaimElement(label="C", text="추가 구성 C", importance="5"),
                    ClaimElement(label="D", text="추가 구성 D", importance="5"),
                ],
            ),
        ]
        caches = [
            {
                "1": [self._item("IA", "동일", "optical"), self._item("IB", "대응 없음")],
                "2": [self._item("C", "대응 없음"), self._item("D", "대응 없음")],
            },
            {
                "1": [self._item("IA", "대응 없음"), self._item("IB", "동일", "adaptive")],
                "2": [self._item("C", "대응 없음"), self._item("D", "대응 없음")],
            },
            {
                "1": [self._item("IA", "대응 없음"), self._item("IB", "대응 없음")],
                "2": [self._item("C", "동일", "feature C"), self._item("D", "대응 없음")],
            },
            {
                "1": [self._item("IA", "대응 없음"), self._item("IB", "대응 없음")],
                "2": [self._item("C", "대응 없음"), self._item("D", "동일", "feature D")],
            },
        ]

        result = self._write_caches_and_build(claims, caches)
        chain = result["chains"]["2"]
        independent_total = result["chains"]["1"]["total"]

        self.assertEqual(set(independent_total), {0, 1})
        self.assertEqual(chain["inherited"], independent_total)
        self.assertEqual(chain["added"], [])
        self.assertEqual(chain["total"], independent_total)
        self.assertFalse(chain["coverage_complete"])
        self.assertEqual(chain["uncovered_labels"], ["C", "D"])

    def test_phase2_does_not_reject_when_one_new_reference_cannot_cover_the_claim(self):
        claim = ParsedClaim(
            claim_number=2,
            claim_type="dependent",
            parent_claim=1,
            text="claim 2",
            elements=[ClaimElement(label="C", text="추가 구성 C")],
        )
        chain_info = {
            "inherited": [0, 1],
            "added": [],
            "total": [0, 1],
            "coverage_complete": False,
            "uncovered_labels": ["C"],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
        }

        report = generate_dependent_phase2("### 청구항 2\n\n추가 구성을 검토했습니다.", claim, chain_info, Settings())

        self.assertIn("새 인용발명 1개만 추가", report)
        self.assertIn("쉽게 발명할 수 있다고 보기 어렵습니다", report)


class ConsistencyRegressionTests(unittest.TestCase):
    def test_quote_limit_applies_even_when_input_already_contains_ellipsis(self):
        quote = "A" * 180 + " ... " + "B" * 220
        shortened = _shorten_quote(quote)

        self.assertLessEqual(len(shortened), 350)
        self.assertIn(" ... ", shortened)

    def test_dependent_phase2_preserves_explicit_phase1_conclusion(self):
        claim = ParsedClaim(
            claim_number=2,
            claim_type="dependent",
            parent_claim=1,
            text="claim 2",
            elements=[ClaimElement(label="A", text="additional feature")],
        )
        chain_info = {
            "inherited": [0],
            "added": [],
            "total": [0],
            "parent_available": True,
            "coverage_complete": True,
            "doc_name_mapping": {"0": "인용발명 1"},
        }
        phase1 = (
            "### [추가 구성 (A)] 유사도: 일부 차이 60%\n\n"
            "### 종합 분석 요약\n\n"
            "- 결론: 남은 조건 차이에 관해서는 추가 근거가 필요합니다."
        )

        report = generate_dependent_phase2(phase1, claim, chain_info, Settings())

        self.assertIn("추가 근거가 필요합니다", report)
        self.assertNotIn("쉽게 발명할 수 있습니다", report)

    def test_dependent_phase2_does_not_decide_full_claim_without_parent(self):
        claim = ParsedClaim(
            claim_number=5,
            claim_type="dependent",
            parent_claim=4,
            text="claim 5",
            elements=[ClaimElement(label="A", text="additional feature")],
        )
        chain_info = {
            "inherited": [],
            "added": [0],
            "total": [0],
            "parent_available": False,
            "coverage_complete": True,
            "doc_name_mapping": {"0": "인용발명 1"},
        }

        report = generate_dependent_phase2(
            "### [추가 구성 (A)]\n\n- 결론: 대응됩니다.",
            claim,
            chain_info,
            Settings(),
        )

        self.assertIn("청구항 전체의 거절 근거 구성 가능 여부", report)
        self.assertIn("판단할 수 없습니다", report)

    def test_reference_store_replaces_all_rows_for_same_claim_scope(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir)
            first = [
                {
                    "publication_no": "DOC-1",
                    "title": "first",
                    "used_in_case": "CASE-1",
                    "claim_number": 1,
                    "role": "primary_reference",
                },
                {
                    "publication_no": "DOC-2",
                    "title": "second",
                    "used_in_case": "CASE-1",
                    "claim_number": 1,
                    "role": "secondary_reference",
                },
            ]
            replacement = [
                {
                    "publication_no": "DOC-1",
                    "title": "first",
                    "used_in_case": "CASE-1",
                    "claim_number": 1,
                    "role": "primary_reference",
                }
            ]

            save_reference_entries_sqlite(case_dir, first)
            save_reference_entries_sqlite(case_dir, replacement)

            with closing(sqlite3.connect(case_dir / "reference.sqlite")) as conn:
                rows = conn.execute(
                    "SELECT publication_no FROM reference_entries "
                    "WHERE used_in_case = ? AND claim_number = ?",
                    ("CASE-1", 1),
                ).fetchall()
            self.assertEqual(rows, [("DOC-1",)])

    def test_reference_store_uses_parent_doc_id_for_cached_children(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            case_dir = Path(temp_dir)
            doc = ExtractedDocument(
                doc_index=0,
                doc_id="D1",
                filename="prior.pdf",
                paragraph_records=[
                    ParagraphRecord(
                        doc_id="stale-doc",
                        paragraph_no="[0001]",
                        original_text="paragraph",
                        normalized_text="paragraph",
                    )
                ],
                paragraph_chunks=[
                    PatentChunk(
                        chunk_id="chunk-1",
                        doc_id="stale-doc",
                        paragraph_no="[0001]",
                        original_text="paragraph",
                        normalized_text="paragraph",
                    )
                ],
            )

            save_case_artifacts_sqlite(case_dir, [doc], [])

            with closing(sqlite3.connect(case_dir / "reference.sqlite")) as conn:
                paragraph_doc_ids = conn.execute(
                    "SELECT doc_id FROM paragraphs"
                ).fetchall()
                chunk_doc_ids = conn.execute(
                    "SELECT doc_id FROM chunks"
                ).fetchall()

            self.assertEqual(paragraph_doc_ids, [("D1",)])
            self.assertEqual(chunk_doc_ids, [("D1",)])


if __name__ == "__main__":
    unittest.main()
