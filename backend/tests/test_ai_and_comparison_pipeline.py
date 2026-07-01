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
    _comparison_safe_elements,
    _parse_json_array,
    _retrieval_query_elements,
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
    _filter_summary_diff_by_component_judgments,
    _normalize_difference_section,
    _format_component_comparison,
    _generate_template_b_phase2,
    _extract_first_json_object,
    _make_phase1_b_prompt,
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


class RetrievalQueryExpansionTests(unittest.TestCase):
    def test_mode_value_dependent_claim_expands_search_terms(self):
        elements = [
            ClaimElement(
                label="A",
                text="상기 특정 요청 패킷은, 상기 인밴드 모드를 표시하는 제1값, 상기 아웃밴드 모드를 표시하는 제2값 및 상기 혼용 모드를 표시하는 제3값 중 어느 하나의 값을 포함하는, 무선전력 수신장치.",
            )
        ]

        expanded = _retrieval_query_elements(elements)

        self.assertEqual(expanded[0].label, "A")
        self.assertIn("specific request packet", expanded[0].text)
        self.assertIn("first value second value third value", expanded[0].text)
        self.assertIn("mixed mode hybrid mode", expanded[0].text)


class ComparisonParsingTests(unittest.TestCase):
    def test_comparison_safe_elements_relabels_placeholder_labels(self):
        elements = [
            ClaimElement(label="_", text="processor"),
            ClaimElement(label="_", text="memory"),
        ]

        safe = _comparison_safe_elements(elements)
        response = json.dumps(
            [
                {
                    "label": "A",
                    "doc_index": 0,
                    "found": False,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "대응 없음",
                    "similarity_reason": "not disclosed",
                },
                {
                    "label": "B",
                    "doc_index": 0,
                    "found": False,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "대응 없음",
                    "similarity_reason": "not disclosed",
                },
            ],
            ensure_ascii=False,
        )

        self.assertEqual([element.label for element in safe], ["A", "B"])
        parsed = _parse_json_array(response, safe, expected_doc_indices=[0])
        self.assertEqual([item["label"] for item in parsed], ["A", "B"])

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


class BatchStatusHeartbeatTests(unittest.IsolatedAsyncioTestCase):
    async def test_heartbeat_timeout_does_not_cancel_work(self):
        async def slow_work():
            await asyncio.sleep(0.03)
            return "done"

        with patch.object(analyze_router, "_update_dependent_batch_status") as update_status:
            result = await analyze_router._await_with_batch_status_heartbeat(
                slow_work(),
                job_id="job-1",
                claim_numbers=[2],
                started_at="2026-06-29 00:00:00",
                stage="waiting_for_batch_llm",
                message_builder=lambda elapsed: f"working {elapsed}",
                reports_ready_getter=lambda: 0,
                interval=0.01,
            )

        self.assertEqual(result, "done")
        self.assertGreaterEqual(update_status.call_count, 1)


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
            "### [구성요소]\n\n(C) 실질적동일 95%\n\n- 청구항 구성: 바퀴\n\n"
            "### 종합 분석 요약\n\n- 유사점 요약: 일반 구성\n- 차이점: 없음\n- 결론: 검토 완료",
            1,
            "인용발명 1",
            "인용발명 2",
            "인용발명 3",
            is_combo=True,
            combination_rationale="제3문헌은 주지관용 입증자료로만 사용",
            chain_info=chain_info,
            settings=Settings(),
        )
        self.assertIn("[인용발명 1과 2의 결합 및 주지관용(진보성)]", phase2)
        self.assertIn("[구성대비]", phase2)
        self.assertNotIn("[구성요소]", phase2)
        self.assertIn("[종합 판단]", phase2)

    def test_single_phase2_header_changes_with_common_knowledge(self):
        base_phase1 = (
            "### [구성요소]\n\n(A) 동일 100%\n\n- 청구항 구성: 제어부\n\n"
            "### 종합 분석 요약\n\n- 유사점 요약: 직접 대응\n- 차이점: 없음\n- 결론: 검토 완료"
        )

        novelty_phase2 = _build_phase2_markdown(
            base_phase1,
            1,
            "인용발명 1",
            settings=Settings(),
        )
        self.assertIn("[인용발명 단독(신규성)]", novelty_phase2)

        inventive_phase2 = _build_phase2_markdown(
            base_phase1,
            1,
            "인용발명 1",
            chain_info={
                "common_general_knowledge": [{"label": "B"}],
            },
            settings=Settings(),
        )
        self.assertIn("[인용발명 1 + 주지관용(진보성)]", inventive_phase2)

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
            "### [구성요소]\n\n(A) 동일 100%\n\n- 인용발명 대응 원문: primary quote\n\n"
            "### [구성요소]\n\n(B) 차이 0%\n\n- 인용발명 대응 원문: (인용발명 1에서 해당 구성 확인 불가)\n\n"
            "### 종합 분석 요약\n\n- 유사점 요약: A는 직접 대응됨\n- 차이점: B는 인용발명 2에서만 확인됨\n- 결론: 검토 완료"
        )

        phase2 = asyncio.run(
            _generate_template_b_phase2(phase1_md, claim, matches, docs, chain_info, Settings())
        )

        self.assertIn("동일 100%", phase2)
        self.assertIn("primary quote", phase2)
        self.assertIn("(단락 [0001])", phase2)
        self.assertIn("차이 0%", phase2)
        self.assertIn("(A) 동일", phase2)
        self.assertNotIn("secondary quote (인용발명 2)", phase2)
        self.assertLess(phase2.index("[구성대비]"), phase2.index("[종합 판단]"))

    def test_combo_phase2_preserves_translated_phase1_quote_block(self):
        claim = ParsedClaim(
            claim_number=1,
            text="claim",
            elements=[ClaimElement(label="A", text="sensor", importance="5")],
        )
        docs = [
            ExtractedDocument(filename="primary.pdf"),
            ExtractedDocument(filename="secondary.pdf"),
        ]
        matches = [
            ElementMatch(
                label="A",
                cited_invention_index=0,
                judgment="실질적 동일",
                quote="a sensor that detects pressure",
                chunk_id="[0001]",
            ),
        ]
        chain_info = {
            "total": [0, 1],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
        }
        phase1_md = (
            "### [구성요소]\n\n(A) 실질적동일 90%\n\n"
            "- 인용발명 대응 원문: 압력을 검출하는 센서\n"
            "(단락 [0001], \"a sensor that detects pressure\")\n\n"
            "### 종합 분석 요약\n\n"
            "- 유사점 요약: 핵심 구성이 대응됨\n"
            "- 차이점: 없음\n"
            "- 결론: 검토 완료"
        )

        phase2 = asyncio.run(
            _generate_template_b_phase2(phase1_md, claim, matches, docs, chain_info, Settings())
        )

        self.assertIn("실질적 동일 90%\n\n압력을 검출하는 센서", phase2)
        self.assertIn('(단락 [0001], "a sensor that detects pressure")', phase2)
        self.assertNotIn("\n\na sensor that detects pressure\n\n[종합 판단]", phase2)

    def test_combo_phase2_prefers_phase1_quote_even_when_raw_match_is_no_match(self):
        claim = ParsedClaim(
            claim_number=1,
            text="claim",
            elements=[ClaimElement(label="A", text="sensor", importance="5")],
        )
        docs = [ExtractedDocument(filename="primary.pdf"), ExtractedDocument(filename="secondary.pdf")]
        matches = [
            ElementMatch(
                label="A",
                cited_invention_index=0,
                judgment="대응 없음",
                quote="",
                chunk_id="[0001]",
            ),
        ]
        phase1_md = (
            "### [구성요소]\n\n(A) 실질적동일 90%\n\n"
            "- **인용발명** 대응 원문: 압력을 검출하는 센서\n"
            "(문단 [0001], \"a sensor that detects pressure\")\n\n"
            "### 종합 분석 요약\n\n"
            "- 유사점 요약: 핵심 구성이 대응됨\n"
            "- 차이점: 없음\n"
            "- 결론: 검토 완료"
        )

        phase2 = asyncio.run(
            _generate_template_b_phase2(
                phase1_md,
                claim,
                matches,
                docs,
                {"total": [0, 1], "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"}},
                Settings(),
            )
        )

        self.assertIn("실질적 동일 90%", phase2)
        self.assertIn("압력을 검출하는 센서", phase2)
        self.assertIn('(문단 [0001], "a sensor that detects pressure")', phase2)
        self.assertNotIn("(인용발명 1에서 해당 구성 확인 불가)", phase2)

    def test_combo_component_comparison_uses_primary_reference_per_component(self):
        docs = [
            ExtractedDocument(filename="primary.pdf"),
            ExtractedDocument(filename="secondary.pdf"),
        ]
        matches = [
            ElementMatch(label="A", cited_invention_index=0, judgment="일부 유사", quote="primary quote", chunk_id="[0001]"),
            ElementMatch(label="A", cited_invention_index=1, judgment="실질적 동일", quote="secondary quote", chunk_id="[0002]"),
        ]

        result = _format_component_comparison(
            matches,
            docs,
            primary_idx=0,
            combo=True,
            secondary_matches=matches,
            total_invs=[0, 1],
        )

        self.assertIn("- 인용발명 1: 일부 유사", result)
        self.assertIn("primary quote", result)
        self.assertNotIn("secondary quote", result)

    def test_combo_phase1_prompt_requires_primary_gap_before_secondary_evidence(self):
        prompt = _make_phase1_b_prompt(
            ParsedClaim(
                claim_number=1,
                text="claim",
                elements=[ClaimElement(label="B", text="controller", importance="5")],
            ),
            [ElementMatch(label="B", cited_invention_index=0, judgment="일부 차이", quote="primary quote", chunk_id="[0001]")],
            [ExtractedDocument(filename="primary.pdf"), ExtractedDocument(filename="secondary.pdf")],
            {
                "total": [0, 1],
                "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
            },
            Settings(),
            secondary_matches=[
                ElementMatch(label="B", cited_invention_index=1, judgment="실질적 동일", quote="secondary quote", chunk_id="[0002]")
            ],
        )

        self.assertIn("인용발명 1에는 [명시되지 않은 부분]이 명시되어 있지 않아 차이가 있으나", prompt)
        self.assertIn("인용발명 2 발췌가 청구항의 부족한 제한에 대응되는 이유", prompt)
        self.assertIn("인용발명 2의 직접 대응 여부와 보완 범위는 종합 분석 요약의 차이점에서만 작성합니다.", prompt)

    def test_combo_phase2_removes_diff_for_substantially_identical_component(self):
        claim = ParsedClaim(
            claim_number=11,
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
        }
        phase1_md = (
            "### [구성요소]\n\n(A) 일부유사 85%\n\n"
            "- 인용발명 대응 원문: primary quote\n\n"
            "### 종합 분석 요약\n\n"
            "- 유사점 요약: 핵심 구성이 실질적으로 대응됨\n"
            "- 차이점: (A) 표현 차이는 있으나 인용발명 1에 직접 개시되어 있지 않은 세부 문언은 인용발명 2에서 보완됩니다.\n"
            "  다만 실질적 동일한 기술수단으로 판단됩니다.\n"
            "- 결론: 검토 완료"
        )

        phase2 = asyncio.run(
            _generate_template_b_phase2(phase1_md, claim, [], docs, chain_info, Settings())
        )

        self.assertIn("[차이점]", phase2)
        self.assertIn("인용발명 2에서 보완됩니다", phase2)

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
            "### [구성요소]\n\n(A) 일부차이 80%\n\n"
            "- 인용발명 대응 원문: primary quote\n\n"
            "### 종합 분석 요약\n\n"
            "- 유사점 요약: 일부 구성이 대응됨\n"
            "- 차이점: [[차이점 1]] 추가 구성 확인 필요\n"
            "- 결론: 검토 필요"
        )

        phase2 = asyncio.run(
            _generate_template_b_phase2(phase1_md, claim, [], docs, chain_info, Settings())
        )

        self.assertIn("[차이점]", phase2)
        self.assertIn("추가 구성 확인 필요", phase2)
        self.assertNotIn("[결합 논리]", phase2)
        self.assertNotIn("[[차이점 1]]", phase2)
        self.assertNotIn("[차이점 1]", phase2)

    def test_combo_phase2_normalizes_difference_into_phase2_sentence_and_conclusion_lines(self):
        claim = ParsedClaim(
            claim_number=10,
            text="claim",
            elements=[ClaimElement(label="B", text="feature", importance="5")],
        )
        docs = [
            ExtractedDocument(filename="primary.pdf"),
            ExtractedDocument(filename="secondary.pdf"),
        ]
        chain_info = {
            "total": [0, 1],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
        }
        phase1_md = (
            "### [구성요소]\n\n(B) 차이 70%\n\n"
            "- 인용발명 대응 원문: primary quote\n\n"
            "### 종합 분석 요약\n\n"
            "- 유사점 요약: 일부 구성이 대응됨\n"
            "- 차이점: (B) 청구항의 센서 출력값에 따라 보정값을 갱신하는 구성은 인용발명 1에 직접 개시되어 있지 않으나,\n"
            "  인용발명 2에는 센서 측정값을 이용하여 보정 파라미터를 갱신\n"
            "  (단락 [0034], \"the correction parameter is updated using the sensed value\") 하는 구성이 기재되어 있으며\n"
            "  이는 입력값 변화에 따라 보정값을 다시 설정하는 제어 로직을 제시하므로, 청구항의 해당 차이 부분에 대응되는 내용으로 볼 수 있습니다.\n"
            "  다만 갱신 조건의 구체성은 청구항보다 제한적으로 개시되어 있다.\n"
            "- 결론: 검토 필요"
        )

        phase2 = asyncio.run(
            _generate_template_b_phase2(phase1_md, claim, [], docs, chain_info, Settings())
        )

        self.assertIn(
            '청구항의 센서 출력값에 따라 보정값을 갱신하는 구성에 대해 '
            '인용발명 2에는 "센서 측정값을 이용하여 보정 파라미터를 갱신"이라는 내용이 기재되어 있으며'
            '(단락 [0034], "the correction parameter is updated using the sensed value"), 이는 '
            '입력값 변화에 따라 보정값을 다시 설정하는 제어 로직을 제시하므로, 청구항의 해당 차이 부분에 대응된다고 볼 수 있습니다.',
            phase2,
        )
        self.assertNotIn("구성 (B)의", phase2)

    def test_combo_phase2_converts_generic_positive_conclusion_to_plain_difference_when_reason_is_negative(self):
        phase2 = _build_phase2_markdown(
            "### [구성요소]\n\n(B) 차이 70%\n\n"
            "### 종합 분석 요약\n\n"
            "- 유사점 요약: 일부 구성이 대응됨\n"
            "- 차이점: (B) 뎁스 모드를 포함하는 세 가지 모드의 제어 회로를 구비하는 이미지 처리 장치는 인용발명 1에 직접 개시되어 있지 않으나, "
            "인용발명 2에는 이미지 처리 장치가 기재되어 있으며 (문단 [0042]) 이는 일부 모드 제어만 설명할 뿐 청구항과 같이 세 가지 모드를 모두 명시하고 있지 않습니다. "
            "따라서 추가 검토를 통해 거절 근거를 구성할 수 있습니다.\n"
            "- 결론: 검토 필요",
            1,
            "인용발명 1",
            "인용발명 2",
            is_combo=True,
            settings=Settings(),
        )

        self.assertIn(
            "뎁스 모드를 포함하는 세 가지 모드의 제어 회로를 구비하는 이미지 처리 장치에 대해서는 명시되어 있지 않다는 점에서 차이가 있습니다.",
            phase2,
        )
        self.assertNotIn("추가 검토를 통해 거절 근거를 구성할 수 있습니다", phase2)

    def test_combo_phase2_converts_missing_disclosure_to_non_obviousness_style_difference(self):
        phase2 = _build_phase2_markdown(
            "### [구성요소]\n\n(C) 차이 65%\n\n"
            "### 종합 분석 요약\n\n"
            "- 유사점 요약: 일부 구성이 대응됨\n"
            "- 차이점: (C) 적외선 픽셀이 포함된 패턴에 최적화된 합산 처리는 인용발명 1에 직접 개시되어 있지 않으나, "
            "인용발명 2에는 공백 보완 방식이 기재되어 있으며 (문단 [0051]) 이는 공백 보완만 설명할 뿐 적외선 픽셀이 포함된 패턴에 최적화된 합산 처리에 대해서는 구체적 개시가 없습니다. "
            "따라서 용이하게 도출될 수 있을 것으로 판단됩니다.\n"
            "- 결론: 검토 필요",
            1,
            "인용발명 1",
            "인용발명 2",
            is_combo=True,
            settings=Settings(),
        )

        self.assertIn(
            "다만 적외선 픽셀이 포함된 패턴에 최적화된 합산 처리에 대해서는 구체적 개시가 없다는 점에서 용이하게 도출하기 어렵다고 보여집니다.",
            phase2,
        )
        self.assertNotIn("용이하게 도출될 수 있을 것으로 판단됩니다", phase2)

    def test_combo_phase2_preserves_phase1_limiting_reason_before_therefore(self):
        phase2 = _build_phase2_markdown(
            "### [구성요소]\n\n(B) 일부차이 75%\n\n"
            "### 종합 분석 요약\n\n"
            "- 유사점 요약: 일부 통신 모드가 대응됨\n"
            "- 차이점: (B) 인밴드 통신과 아웃밴드 통신을 함께 사용하는 혼용 모드는 인용발명 1에 직접 개시되어 있지 않으나, "
            "인용발명 2에는 아웃밴드 방식이 일치하면 아웃밴드로 반환하고 일치하지 않으면 제1 통신 방식으로 반환하는 내용이 기재되어 있으며 (단락 [0009]) "
            "이는 아웃밴드 통신의 일치성 여부에 따라 인밴드 또는 아웃밴드를 전환하여 이용하는 제어 방식에 대응됩니다. "
            "다만 이는 동시 병행 사용이라기보다는 택일적 전환에 가까운 측면이 있어 청구항의 혼용 모드와 완전히 동일하지는 않습니다. "
            "따라서 인용발명 2의 개시 기법을 결합하더라도 혼용 모드 구성을 직접 도출하기에는 한계가 있어 추가 근거가 필요합니다.\n"
            "- 결론: 검토 필요",
            1,
            "인용발명 1",
            "인용발명 2",
            is_combo=True,
            settings=Settings(),
        )

        self.assertIn("택일적 전환에 가까운 측면이 있어 청구항의 혼용 모드와 완전히 동일하지는 않습니다.", phase2)
        self.assertIn("따라서 인용발명 2의 개시 기법을 결합하더라도 혼용 모드 구성을 직접 도출하기에는 한계가 있어 추가 근거가 필요합니다.", phase2)
        self.assertLess(
            phase2.index("택일적 전환에 가까운 측면"),
            phase2.index("따라서 인용발명 2의 개시 기법"),
        )

    def test_combo_phase2_adds_space_after_difference_label(self):
        normalized = _normalize_difference_section("(A)?붿껌?쒖옄 媛?낅꽦???꾪빐 ?쒓났?쒕? 蹂댁젙?⑸땲??")

        self.assertEqual("?붿껌?쒖옄 媛?낅꽦???꾪빐 ?쒓났?쒕? 蹂댁젙?⑸땲??", normalized)

    def test_phase2_builds_judgment_fallback_when_phase1_summary_is_missing(self):
        phase1_md = (
            "### [구성요소]\n\n"
            "(A) 실질적동일 95%\n\n"
            "- 인용발명 대응 원문: primary quote\n\n"
            "- 판단 이유: 인용발명 1의 배열 구조가 청구항과 실질적으로 동일합니다.\n\n"
            "### [구성요소]\n\n"
            "(B) 일부유사 60%\n\n"
            "- 인용발명 대응 원문: secondary quote\n\n"
            "- 판단 이유: 세부 모드 동작 회로는 직접 개시되지 않아 추가 보완이 필요합니다. 기능적 취지는 일부 대응됩니다."
        )

        phase2 = _build_phase2_markdown(
            phase1_md,
            13,
            "인용발명 1",
            "인용발명 2",
            is_combo=True,
            settings=Settings(),
        )

        self.assertNotIn("직접 작성하십시오", phase2)
        self.assertIn("구성요소 (A)는 인용발명과 직접 대응되거나 실질적으로 동일한 구성이 확인됩니다.", phase2)
        self.assertIn("세부 모드 동작 회로는 직접 개시되지 않아 추가 보완이 필요합니다.", phase2)
        self.assertIn("구성요소 (B)에서 남는 차이가 있어 추가 보완 근거나 결합 논리 검토가 필요합니다.", phase2)

    def test_filter_summary_diff_drops_substantially_identical_without_space_variant(self):
        filtered = _filter_summary_diff_by_component_judgments(
            "(A) 문언 차이만 있습니다.\n\n(B) 실제 차이가 남습니다.",
            [
                {"label": "A", "judgment": "실질적동일"},
                {"label": "B", "judgment": "차이"},
            ],
        )

        self.assertNotIn("(A)", filtered)
        self.assertIn("(B) 실제 차이가 남습니다.", filtered)

    def test_combo_phase2_splits_multiple_labeled_differences_without_blank_lines(self):
        normalized = _normalize_difference_section(
            "(A) 첫 번째 차이입니다.\n"
            "다만 보완 필요합니다.\n"
            "(B) 두 번째 차이입니다.\n"
            "또한 추가 검토가 필요합니다.\n"
            "(C) 세 번째 차이입니다."
        )

        self.assertEqual(
            "첫 번째 차이입니다.\n"
            "다만 보완 필요합니다.\n\n"
            "두 번째 차이입니다.\n"
            "또한 추가 검토가 필요합니다.\n\n"
            "세 번째 차이입니다.",
            normalized,
        )

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

    def test_secondary_selection_prefers_broader_consistent_gap_coverage_over_single_strong_point(self):
        claim = ParsedClaim(
            claim_number=1,
            text="image sensor claim",
            elements=[
                ClaimElement(label="A", text="infrared pixel array", importance="5"),
                ClaimElement(label="B", text="mode control circuit", importance="3"),
                ClaimElement(label="C", text="luminance-based remosaic processing", importance="3"),
                ClaimElement(label="D", text="low-light binning control", importance="3"),
                ClaimElement(label="E", text="depth map generation", importance="5"),
            ],
        )
        caches = [
            {"1": [
                self._item("A", "\ub3d9\uc77c", "infrared pixel array"),
                self._item("B", "\ub300\uc751 \uc5c6\uc74c"),
                self._item("C", "\ub300\uc751 \uc5c6\uc74c"),
                self._item("D", "\ub300\uc751 \uc5c6\uc74c"),
                self._item("E", "\ub300\uc751 \uc5c6\uc74c"),
            ]},
            {"1": [
                self._item("A", "\ub300\uc751 \uc5c6\uc74c"),
                self._item("B", "\ub300\uc751 \uc5c6\uc74c"),
                self._item("C", "\ub300\uc751 \uc5c6\uc74c"),
                self._item("D", "\ub300\uc751 \uc5c6\uc74c"),
                self._item("E", "\ub3d9\uc77c", "depth map generation"),
            ]},
            {"1": [
                self._item("A", "\ub300\uc751 \uc5c6\uc74c"),
                self._item("B", "\uc77c\ubd80 \ucc28\uc774", "mode control circuit"),
                self._item("C", "\uc77c\ubd80 \ucc28\uc774", "luminance adaptive remosaic processing"),
                self._item("D", "\uc77c\ubd80 \ucc28\uc774", "low-light binning control"),
                self._item("E", "\ub300\uc751 \uc5c6\uc74c"),
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

        self.assertEqual(result["chains"]["1"]["total"], [0, 2])
        self.assertGreater(
            result["secondary_candidate_details"]["2"]["residual_breadth"],
            result["secondary_candidate_details"]["1"]["residual_breadth"],
        )
        self.assertGreater(
            result["secondary_candidate_details"]["1"]["single_feature_dominance_penalty"],
            0,
        )


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

    def test_rejected_inventions_section_keeps_similar_items_when_later_labels_are_missing(self):
        claim = ParsedClaim(
            claim_number=1,
            claim_type="independent",
            text="청구항 1. 장치.",
            elements=[
                ClaimElement(label="A", text="sensor module", importance="5"),
                ClaimElement(label="B", text="controller", importance="4"),
                ClaimElement(label="C", text="mode selector", importance="3"),
                ClaimElement(label="D", text="display", importance="3"),
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
                    "quote": "sensor arrangement",
                    "chunk_id": "[0001]",
                    "judgment": "실질적동일",
                    "similarity_reason": "센서 모듈은 청구항 구성과 실질적으로 대응됩니다.",
                },
                {
                    "label": "B",
                    "found": True,
                    "quote": "controller sends a signal",
                    "chunk_id": "[0002]",
                    "judgment": "일부 유사",
                    "similarity_reason": "제어부가 신호를 송신하는 점은 일부 유사합니다.",
                },
                {
                    "label": "C",
                    "found": False,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "대응없음",
                    "similarity_reason": "",
                },
                {
                    "label": "D",
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

        self.assertIn("(A) 센서 모듈은 청구항 구성과 실질적으로 대응됩니다. (sensor arrangement [0001])", result)
        self.assertIn("(B) 제어부가 신호를 송신하는 점은 일부 유사합니다. (controller sends a signal [0002])", result)
        self.assertIn("차이점: (C), (D) 구성은 이 인용발명에서 직접 확인되지 않아 최종 채택에서 제외되었습니다.", result)
        self.assertNotIn("청구항과 직접 대응되는 구성은 확인되지 않았습니다.", result)

    def test_rejected_inventions_section_uses_legacy_reason_and_evidence_for_similar_items(self):
        claim = ParsedClaim(
            claim_number=1,
            claim_type="independent",
            text="청구항 1. 장치.",
            elements=[
                ClaimElement(label="A", text="sensor module", importance="5"),
                ClaimElement(label="B", text="controller", importance="4"),
                ClaimElement(label="C", text="display", importance="3"),
                ClaimElement(label="Z", text="terminal module", importance="3"),
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
                    "quote": "sensor arrangement",
                    "chunk_id": "[0001]",
                    "judgment": "실질적 동일",
                    "판단_이유": "센서 모듈은 청구항 구성과 실질적으로 대응됩니다.",
                },
                {
                    "label": "B",
                    "found": True,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "일부 유사",
                    "판단_이유": "제어부가 신호를 송신하는 점은 일부 유사합니다.",
                    "evidence": [
                        {
                            "limitation": "controller",
                            "quote": "controller sends a signal",
                            "chunk_id": "[0002]",
                        }
                    ],
                },
                {
                    "label": "C",
                    "found": False,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "대응 없음",
                    "similarity_reason": "",
                },
                {
                    "label": "Z",
                    "found": True,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "일부 유사",
                    "판단_이유": "단말 모듈이 통신 신호를 처리하는 점은 일부 유사합니다.",
                    "evidence": [
                        {
                            "limitation": "terminal module",
                            "quote": "terminal module processes a communication signal",
                            "chunk_id": "[0026]",
                        }
                    ],
                },
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "comparisons_1.json").write_text(
                json.dumps(cache, ensure_ascii=False),
                encoding="utf-8",
            )
            result = build_rejected_inventions_section(claim, docs, chain_info, temp_dir)

        self.assertIn("(A) 센서 모듈은 청구항 구성과 실질적으로 대응됩니다. (sensor arrangement [0001])", result)
        self.assertIn("(B) 제어부가 신호를 송신하는 점은 일부 유사합니다. (controller sends a signal [0002])", result)
        self.assertIn("(Z) 단말 모듈이 통신 신호를 처리하는 점은 일부 유사합니다. (terminal module processes a communication signal [0026])", result)
        self.assertIn("차이점: (C) 구성은 이 인용발명에서 직접 확인되지 않아 최종 채택에서 제외되었습니다.", result)

    def test_rejected_inventions_section_uses_difference_reason_and_quote_when_present(self):
        claim = ParsedClaim(
            claim_number=1,
            claim_type="independent",
            text="청구항 1. 장치.",
            elements=[
                ClaimElement(label="A", text="sensor module", importance="5"),
                ClaimElement(label="B", text="controller", importance="4"),
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
                    "quote": "controller only forwards a preset signal",
                    "chunk_id": "[0007]",
                    "judgment": "대응 없음",
                    "similarity_reason": "제어 신호 전달은 보이나 청구항의 제어부 판단 로직은 직접 개시되어 있지 않습니다.",
                },
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "comparisons_1.json").write_text(
                json.dumps(cache, ensure_ascii=False),
                encoding="utf-8",
            )
            result = build_rejected_inventions_section(claim, docs, chain_info, temp_dir)

        self.assertIn(
            "차이점: (B) 제어 신호 전달은 보이나 청구항의 제어부 판단 로직은 직접 개시되어 있지 않습니다. (controller only forwards a preset signal [0007])",
            result,
        )

    def test_rejected_inventions_section_prefixes_dependent_claim_summary(self):
        claim = ParsedClaim(
            claim_number=3,
            claim_type="dependent",
            parent_claim=2,
            text="제2항에 있어서,상기 특정 요청 패킷은, 상기 인밴드 모드를 표시하는 제1값, 상기 아웃밴드 모드를 표시하는 제2값 및 상기 혼용 모드를 표시하는 제3값 중 어느 하나의 값을 포함하는, 무선전력 수신장치.",
            elements=[
                ClaimElement(
                    label="A",
                    text="상기 특정 요청 패킷은, 상기 인밴드 모드를 표시하는 제1값, 상기 아웃밴드 모드를 표시하는 제2값 및 상기 혼용 모드를 표시하는 제3값 중 어느 하나의 값을 포함하는, 무선전력 수신장치.",
                    importance="5",
                ),
            ],
        )
        docs = [
            ExtractedDocument(filename="primary.pdf"),
            ExtractedDocument(filename="related-a.pdf"),
        ]
        chain_info = {
            "total": [0],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 3"},
        }
        cache = {
            "3": [
                {
                    "label": "A",
                    "found": False,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "대응 없음",
                    "판단_이유": "인밴드, 아웃밴드, 혼용 모드를 지시하는 값이나 이를 선택하는 특정 요청 패킷에 대한 기재가 확인되지 않습니다.",
                }
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "comparisons_1.json").write_text(
                json.dumps(cache, ensure_ascii=False),
                encoding="utf-8",
            )
            result = build_rejected_inventions_section(claim, docs, chain_info, temp_dir)

        self.assertIn(
            "청구항의 특정 요청 패킷은, 상기 인밴드 모드를 표시하는 제1값, 상기 아웃밴드 모드를 표시하는 제2값 및 상기 혼용 모드를 표시하는 제3값 중 어느 하나의 값을 포함하는 구성은",
            result,
        )
        self.assertIn("차이점: (A)", result)


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
        self.assertIn("청구항 2의 추가 구성에 대해서는", report)
        self.assertNotIn("청구항 2의 C에 대해서는", report)


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
            "### [추가 구성]\n\n(A) 차이 60%\n\n"
            "### 종합 분석 요약\n\n"
            "- 결론: 남은 조건 차이에 관해서는 추가 근거가 필요합니다."
        )

        report = generate_dependent_phase2(phase1, claim, chain_info, Settings())

        self.assertIn("추가 근거가 필요합니다", report)
        self.assertNotIn("쉽게 발명할 수 있습니다", report)

    def test_dependent_phase2_additional_configuration_contains_quote_only(self):
        claim = ParsedClaim(
            claim_number=2,
            claim_type="dependent",
            parent_claim=1,
            text="제1항에 있어서, 압력 센서를 더 포함하는 장치",
            elements=[ClaimElement(label="A", text="압력 센서를 더 포함하는 구성")],
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
            "### [추가 구성]\n\n"
            "(A) 실질적동일 90%\n\n"
            "- 청구항 추가 구성: 압력 센서를 더 포함하는 구성\n\n"
            "- 인용발명 대응 원문: 압력 센서를 포함한다(단락 [0010])\n\n"
            "- 인용발명 대응 부분 요약: 압력 센서가 요약됩니다.\n\n"
            "- 판단 이유: 청구항의 센서와 대응됩니다.\n\n"
            "### 종합 분석 요약\n\n"
            "- 차이점: 없음\n"
            "- 결론: 추가 구성은 인용발명 1에 의해 실질적으로 동일하게 확인됩니다."
        )

        report = generate_dependent_phase2(phase1, claim, chain_info, Settings())
        additional = report.split("### 종합 분석 요약", 1)[0]

        self.assertIn("[추가 구성]", report)
        self.assertIn("- 인용발명 대응 원문:", additional)
        self.assertIn("압력 센서를 포함한다(단락 [0010])", additional)
        self.assertNotIn("인용발명 대응 부분 요약", additional)
        self.assertNotIn("판단 이유", additional)
        self.assertIn("[결론]", report)
        self.assertIn("청구항 2의 구성에 대해 인용발명 1에는 위 인용발명 대응 원문과 같은 내용이 기재되어 있으며", report)
        self.assertIn("이는 청구항 2의 압력 센서를 더 포함하는 구성에 실질적으로 동일하게 대응될 수 있습니다.", report)
        self.assertIn("따라서 추가 구성은 인용발명 1에 의해 실질적으로 동일하게 확인됩니다.", report)
        self.assertNotIn("[판단 이유]", report)
        self.assertNotIn("청구항의 센서와 대응됩니다.", report)

    def test_dependent_phase2_bold_quote_label_does_not_leak_into_claim_core(self):
        claim = ParsedClaim(
            claim_number=3,
            claim_type="dependent",
            parent_claim=2,
            text="제2항에 있어서, 특정 요청 패킷이 모드 표시 값을 포함하는 장치",
            elements=[ClaimElement(label="A", text="특정 요청 패킷이 모드 표시 값을 포함하는 구성")],
        )
        chain_info = {
            "inherited": [0, 1],
            "added": [],
            "total": [0, 1],
            "parent_available": True,
            "coverage_complete": False,
            "uncovered_labels": ["A"],
            "doc_name_mapping": {"0": "인용발명 1", "1": "인용발명 2"},
        }
        phase1 = (
            "### [추가 구성]\n\n"
            "(A) 일부차이 70%\n\n"
            "- 청구항 추가 구성: 특정 요청 패킷이 모드 표시 값을 포함하는 구성\n\n"
            "- **인용발명** 대응 원문:\n"
            "가정해보자, PRX는 BLE, NFC 및 강화된 인밴드 통신 방식을 지원한다\n\n"
            "- 판단 이유: 패킷 내 표시 값은 직접 개시되지 않습니다.\n\n"
            "### 종합 분석 요약\n\n"
            "- 차이점: 패킷 내 파라미터 정보가 명시되어 있지 않아 차이가 있습니다.\n"
            "- 결론: 추가 근거가 필요합니다."
        )

        report = generate_dependent_phase2(phase1, claim, chain_info, Settings())
        conclusion = report.split("[결론]", 1)[1].split("따라서", 1)[0]
        claim_side = conclusion.split("이는 청구항 3의", 1)[1]

        self.assertIn("특정 요청 패킷이 모드 표시 값을 포함하는 구성과 차이가 있습니다.", claim_side)
        self.assertNotIn("인용발명", claim_side)
        self.assertNotIn("가정해보자", claim_side)

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
