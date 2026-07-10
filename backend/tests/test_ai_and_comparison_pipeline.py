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
    _claim_keywords,
    _comparison_safe_elements,
    _parse_json_array,
    _select_best_matches,
    _shorten_quote,
    analyze_claim_elements_hybrid,
    normalize_label,
    verify_quotes,
)
from backend.services.citation_chain import (
    CITATION_CHAIN_POLICY_VERSION,
    _apply_conventional_support_policy,
    _conventionality_basis,
    _score_prior_cache,
    build_citation_chain_from_comparisons,
    get_claim_chain_info,
)
from backend.services.reference_store import (
    save_case_artifacts_sqlite,
    save_reference_entries_sqlite,
)
from backend.services.report_generator import (
    _format_component_comparison,
    _dedupe_phase1_sections,
    _extract_first_json_object,
    _make_phase1_b_prompt,
    _make_phase1_prompt,
    build_rejected_inventions_section,
    format_rejection_basis_header,
    generate_dependent_report,
    parse_manual_claim_locally,
    polish_phase1_summary_text,
)


class RejectionBasisHeaderTests(unittest.TestCase):
    def test_single_reference_with_remaining_differences_is_not_labeled_novelty(self):
        header = format_rejection_basis_header(
            "인용발명 1",
            is_novelty=False,
        )
        self.assertEqual(header, "[인용발명 1 단독(진보성 검토)]")

    def test_single_reference_with_all_identical_elements_is_labeled_novelty(self):
        header = format_rejection_basis_header(
            "인용발명 1",
            is_novelty=True,
        )
        self.assertEqual(header, "[인용발명 1 단독(신규성)]")


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

    def test_preamble_label_variants_are_normalized(self):
        self.assertEqual(normalize_label("P"), "P")
        self.assertEqual(normalize_label("(P)"), "P")
        self.assertEqual(normalize_label("[PREAMBLE]"), "P")
        self.assertEqual(normalize_label("(P) 전제부"), "P")

    def test_hybrid_matrix_accepts_parenthesized_preamble_labels(self):
        elements = [ClaimElement(label="P", text="image processing apparatus")]
        response = json.dumps(
            [
                {
                    "label": "(P)",
                    "doc_index": doc_index,
                    "found": False,
                    "quote": "",
                    "chunk_id": "",
                    "judgment": "대응 없음",
                    "판단_이유": "대응 기재가 없다.",
                }
                for doc_index in range(3)
            ],
            ensure_ascii=False,
        )

        parsed = _parse_json_array(response, elements, expected_doc_indices=[0, 1, 2])

        self.assertEqual(
            [(item["doc_index"], item["label"]) for item in parsed],
            [(0, "P"), (1, "P"), (2, "P")],
        )

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

    def test_parse_json_array_dedupes_repeated_document_label_pairs(self):
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
                    "판단_이유": "첫 번째 중복 항목",
                },
                {
                    "label": "A",
                    "doc_index": 0,
                    "found": True,
                    "quote": "pressure sensor",
                    "chunk_id": "[0010]",
                    "judgment": "실질적 동일",
                    "판단_이유": "더 강한 중복 항목",
                },
            ],
            ensure_ascii=False,
        )

        parsed = _parse_json_array(response, elements, expected_doc_indices=[0])

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["judgment"], "실질적 동일")
        self.assertEqual(parsed[0]["quote"], "pressure sensor")

    def test_hybrid_grouped_document_results_are_flattened(self):
        elements = [ClaimElement(label="A", text="sensor")]
        response = json.dumps(
            [
                {
                    "doc_index": 0,
                    "results": [
                        {
                            "claim_element": "A",
                            "found": True,
                            "quote": "pressure sensor",
                            "chunk_id": "[0010]",
                            "judgment": "일부 유사",
                            "판단 이유": "센서 기능은 관련되나 세부 제어 제한은 확인되지 않는다.",
                        }
                    ],
                },
                {
                    "doc_index": 1,
                    "results": [
                        {
                            "claim_element": "A",
                            "found": False,
                            "quote": "",
                            "chunk_id": "",
                            "judgment": "대응 없음",
                            "reason": "대응 기재가 없다.",
                        }
                    ],
                },
            ],
            ensure_ascii=False,
        )

        parsed = _parse_json_array(response, elements, expected_doc_indices=[0, 1])

        self.assertEqual([(item["doc_index"], item["label"]) for item in parsed], [(0, "A"), (1, "A")])
        self.assertEqual(parsed[0]["판단_이유"], "센서 기능은 관련되나 세부 제어 제한은 확인되지 않는다.")

    def test_hybrid_object_response_with_evidence_is_not_parsed_as_evidence_array(self):
        elements = [ClaimElement(label="A", text="sensor")]
        response = json.dumps(
            {
                "comparisons": [
                    {
                        "label": "A",
                        "doc_index": 0,
                        "found": True,
                        "quote": "pressure sensor",
                        "chunk_id": "[0010]",
                        "judgment": "일부 유사",
                        "판단_이유": "센서 기능은 관련되나 제어 조건은 확인되지 않는다.",
                        "evidence": [
                            {"limitation": "센서", "quote": "pressure sensor", "chunk_id": "[0010]"}
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        )

        parsed = _parse_json_array(response, elements, expected_doc_indices=[0])

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["label"], "A")
        self.assertEqual(parsed[0]["evidence"][0]["limitation"], "센서")

    def test_hybrid_document_object_with_label_keys_is_flattened(self):
        elements = [ClaimElement(label="A", text="sensor"), ClaimElement(label="B", text="controller")]
        response = json.dumps(
            [
                {
                    "doc_index": 0,
                    "A": {
                        "found": False,
                        "quote": "",
                        "chunk_id": "",
                        "judgment": "대응 없음",
                        "reason": "대응 기재가 없다.",
                    },
                    "B": {
                        "found": True,
                        "quote": "controller",
                        "chunk_id": "[0020]",
                        "judgment": "일부 차이",
                        "판단 이유": "제어부는 관련되나 세부 알고리즘은 확인되지 않는다.",
                    },
                }
            ],
            ensure_ascii=False,
        )

        parsed = _parse_json_array(response, elements, expected_doc_indices=[0])

        self.assertEqual([(item["doc_index"], item["label"]) for item in parsed], [(0, "A"), (0, "B")])
        self.assertEqual(parsed[1]["판단_이유"], "제어부는 관련되나 세부 알고리즘은 확인되지 않는다.")

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

    async def test_invalid_hybrid_schema_falls_back_to_per_document(self):
        elements = [ClaimElement(label="A", text="sensor")]
        docs = [
            ExtractedDocument(filename="first.pdf", raw_text="[T1] first sensor passage"),
            ExtractedDocument(filename="second.pdf", raw_text="[T1] second sensor passage"),
        ]
        invalid_response = "internal analysis without a JSON array"
        per_doc_responses = [
            json.dumps(
                [
                    {
                        "label": "A",
                        "found": True,
                        "quote": "first sensor passage",
                        "chunk_id": "[T1]",
                        "judgment": "\uc2e4\uc9c8\uc801 \ub3d9\uc77c",
                        "similarity_reason": "first document discloses the sensor.",
                    }
                ],
                ensure_ascii=False,
            ),
            json.dumps(
                [
                    {
                        "label": "A",
                        "found": False,
                        "quote": "",
                        "chunk_id": "",
                        "judgment": "\ub300\uc751 \uc5c6\uc74c",
                        "similarity_reason": "second document has no corresponding feature.",
                    }
                ],
                ensure_ascii=False,
            ),
        ]
        settings = Settings(engine="agy", comparison_mode="hybrid", use_rag_retrieval=False)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "backend.services.citation_extractor.call_ai",
                new_callable=AsyncMock,
                side_effect=[invalid_response, *per_doc_responses],
            ) as mocked_call:
                matches = await analyze_claim_elements_hybrid(
                    elements, docs, settings, job_dir=temp_dir, claim_number=1
                )

            self.assertEqual(mocked_call.await_count, 3)
            self.assertTrue(matches[0].found)
            self.assertEqual(matches[0].cited_invention_index, 0)
            for doc_idx in range(2):
                cache = json.loads(
                    (Path(temp_dir) / f"comparisons_{doc_idx}.json").read_text(encoding="utf-8")
                )
                self.assertIn("1", cache)

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

    async def test_mixed_mode_compacts_long_documents_by_default(self):
        elements = [ClaimElement(label="A", text="needle")]
        docs = [
            ExtractedDocument(filename="doc-0.pdf", raw_text=("A" * 70_000) + " needle"),
            ExtractedDocument(filename="doc-1.pdf", raw_text=("B" * 70_000) + " needle"),
        ]
        settings = Settings(engine="agy", comparison_mode="mixed")

        block = _build_hybrid_docs_block(docs, elements, settings=settings)

        self.assertEqual(block.count("[doc_index="), len(docs))
        self.assertLess(len(block), 90_000)
        self.assertLess(len(block), sum(len(doc.raw_text) for doc in docs))


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


class DependentReportValidationTests(unittest.TestCase):
    def test_dedupe_keeps_unlabeled_single_dependent_section(self):
        report = (
            "### [추가 구성]\n"
            "실질적동일 85%\n\n"
            "- 청구항 추가 구성: 압력 센서를 포함하는 구성\n"
            "- 판단 이유: 인용발명의 압력 센서와 대응됩니다.\n\n"
            "[종합분석요약]\n"
            "- 결론: 추가 구성은 인용발명에 의해 확인됩니다."
        )

        result = _dedupe_phase1_sections(report)

        self.assertIn("실질적동일 85%", result)
        self.assertIn("[종합분석요약]", result)

    def test_header_only_dependent_report_is_not_substantive(self):
        report = "[인용발명 1 단독(신규성)]\n\n[구성대비]\n\n### 청구항 3"

        self.assertFalse(analyze_router._has_substantive_dependent_report(report, 3))

    def test_dependent_report_with_body_is_substantive(self):
        report = "### 청구항 3\n\n### [추가 구성]\n실질적동일 85%\n\n판단 이유: 대응 근거가 확인됩니다."

        self.assertTrue(analyze_router._has_substantive_dependent_report(report, 3))


class DependentReportGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_report_prompt_handles_uncovered_chain_without_added_doc(self):
        claim = ParsedClaim(
            claim_number=7,
            claim_type="dependent",
            parent_claim=1,
            text="제1항에 있어서, 압력 센서를 더 포함하는 장치",
            elements=[ClaimElement(label="A", text="압력 센서를 더 포함하는 구성")],
        )
        matches = [
            ElementMatch(
                label="A",
                cited_invention_index=0,
                judgment="차이",
                quote="pressure sensor",
                chunk_id="[0001]",
            )
        ]
        docs = [ExtractedDocument(filename="prior.pdf")]
        chain_info = {
            "inherited": [0],
            "added": [],
            "total": [0],
            "coverage_complete": False,
            "doc_name_mapping": {"0": "인용발명 1"},
        }

        with patch(
            "backend.services.report_generator.call_ai",
            new=AsyncMock(return_value="ok"),
        ) as mocked_call:
            result = await generate_dependent_report(
                claim,
                matches,
                docs,
                chain_info,
                Settings(),
            )

        self.assertEqual(result, "ok")
        prompt = mocked_call.await_args.args[0]
        self.assertIn("단일 추가 문헌 커버 상태: 일부 추가 구성 미대응", prompt)
        self.assertIn("`청구항 추가 구성:`에는 종속항 원문의 `제~항에 있어서,` 문구를 포함", prompt)


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

        self.assertIn("인용발명 1에서 확인되지 않는 하위 제한", prompt)
        self.assertIn("보강 후 실질적인 차이가 남는 경우", prompt)
        self.assertIn("외국어 문헌의 괄호 안 따옴표 원문은 반드시 해당 외국어 원문 그대로", prompt)
        self.assertIn("인용발명 2의 직접 대응 여부와 보완 범위는 종합 분석 요약의 차이점에서만 작성합니다.", prompt)

    def test_phase1_summary_polish_removes_repeated_fallback_phrasing(self):
        raw = (
            "손실은 다수의 방향에서 자세 변형된 3d 휴먼모델을 렌더링한 3d 이미지들과 "
            "다수의 방향에서 gt 3d 휴먼모델을 렌더링한 2d 이미지들 간의 차이인 구성에 대해 "
            "인용발명 1에는 3d 이미지와 2d 이미지 간의 렌더링 차이를 손실로 산출하는 구성이 "
            "명시되어 있지 않아 차이가 있습니다.\n"
            "다만 이 구성에 대한 직접적인 대응 관계를 보완할 보조 인용발명은 기재되어 있지 않습니다.\n"
            "이는 3d 이미지와 2d 이미지 간의 차이를 손실로 계산하는 세부 처리 조건이 부재함을 의미합니다.\n"
            "다만 인용발명 1로도 확인되지 않는 하위 제한은 충족하지 못합니다.\n"
            "따라서 인용발명 1의 이미지 간 인지적 손실을 다방향 렌더링 이미지 간의 차이로 설계 변경하는 것은 "
            "통상의 기술자가 추가 근거 없이 용이하게 도출할 수 있는지 검토가 필요합니다."
        )

        polished = polish_phase1_summary_text(raw)

        self.assertIn("인용발명도 확인되지 않습니다.", polished)
        self.assertIn("세부 처리 조건이 부재하고", polished)
        self.assertIn("추가 문헌 근거 없이 통상의 기술자가 용이하게 도출할 수 있는지는 별도로 검토해야 합니다.", polished)
        self.assertNotIn("다만 이 구성에 대한", polished)
        self.assertNotIn("다만 인용발명 1로도", polished)


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

    def test_quoted_difference_is_kept_as_secondary_evidence_for_primary_gap(self):
        claim = ParsedClaim(
            claim_number=1,
            text="visual text generation claim",
            elements=[
                ClaimElement(label="A", text="프롬프트를 레이아웃 조건으로 변환", importance="5"),
                ClaimElement(label="B", text="OCR 오류 영역 식별", importance="3"),
            ],
        )
        caches = [
            {"1": [
                self._item("A", "대응 없음"),
                self._item("B", "동일", "OCR mismatch detection"),
            ]},
            {"1": [
                self._item("A", "차이", "layout of keywords extracted from text prompts"),
                self._item("B", "대응 없음"),
            ]},
        ]
        docs = [ExtractedDocument(filename=f"doc-{idx}.pdf") for idx in range(2)]

        with tempfile.TemporaryDirectory() as temp_dir:
            for idx, cache in enumerate(caches):
                (Path(temp_dir) / f"comparisons_{idx}.json").write_text(
                    json.dumps(cache, ensure_ascii=False),
                    encoding="utf-8",
                )
            result = build_citation_chain_from_comparisons(temp_dir, [claim], docs)

        self.assertEqual(result["chains"]["1"]["total"], [0, 1])
        self.assertEqual(result["secondary_reason"], "support")
        matrix = result["gap_evidence_matrix"]["1"]["elements"]
        self.assertEqual(matrix[0]["label"], "A")
        self.assertEqual(matrix[0]["candidate_evidence"][0]["doc_idx"], 1)
        self.assertIn("layout of keywords", matrix[0]["candidate_evidence"][0]["quote"])

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

    def test_conditioned_selection_keywords_prioritize_generic_structure(self):
        elements = [
            ClaimElement(
                label="B",
                text="입력 데이터의 유형을 고려하여 제1 처리 서비스 또는 제2 변환 서비스 중 적어도 하나를 수행하는 단계",
                importance="3",
            )
        ]

        keywords = _claim_keywords(elements)

        self.assertLess(keywords.index("선택"), keywords.index("데이터의"))
        self.assertLess(keywords.index("분기"), keywords.index("처리"))
        self.assertIn("condition", keywords)

    def test_conditioned_selection_element_is_core_weighted_for_primary_score(self):
        claim = ParsedClaim(
            claim_number=1,
            claim_type="independent",
            text="방법",
            elements=[
                ClaimElement(label="A", text="입력 데이터를 수신하는 단계", importance="3"),
                ClaimElement(
                    label="B",
                    text="입력 데이터의 유형을 고려하여 제1 처리 서비스 또는 제2 변환 서비스 중 적어도 하나를 수행하는 단계",
                    importance="3",
                ),
            ],
        )
        cache = {
            "1": [
                {"label": "A", "judgment": "대응 없음"},
                {"label": "B", "judgment": "일부 차이"},
            ]
        }

        _score, _match_count, detail = _score_prior_cache(cache, [claim])

        # B의 명시 importance는 3이지만 조건 기반 선택식이므로 핵심 가중치(4)로 승격된다.
        self.assertAlmostEqual(detail["core_coverage"], 0.55, places=2)

    def test_local_claim_parser_marks_conditioned_selection_as_core(self):
        parsed = asyncio.run(parse_manual_claim_locally(
            "처리 장치가 (A) 입력 데이터를 수신하는 단계 (B) 입력 데이터의 유형을 고려하여 제1 처리 서비스 또는 제2 변환 서비스 중 적어도 하나를 수행하는 단계 및 (C) 결과를 출력하는 단계를 포함하는 방법",
            1,
            "independent",
            None,
        ))

        importance_by_label = {element.label: element.importance for element in parsed.elements}
        self.assertEqual(importance_by_label["B"], "5")

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

    def test_only_one_partial_reference_is_added_for_one_dependent_claim(self):
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
        self.assertEqual(chain["added"], [2])
        self.assertEqual(chain["total"], independent_total + [2])
        self.assertFalse(chain["coverage_complete"])
        self.assertEqual(chain["uncovered_labels"], ["D"])

    def test_dependent_claim_keeps_quoted_difference_as_partial_added_reference(self):
        claims = [
            ParsedClaim(
                claim_number=1,
                text="independent claim",
                elements=[
                    ClaimElement(label="IA", text="잠재 포인트 클라우드", importance="5"),
                    ClaimElement(label="IB", text="대상 식별 임베딩", importance="5"),
                ],
            ),
            ParsedClaim(
                claim_number=4,
                claim_type="dependent",
                parent_claim=1,
                text="claim 4",
                elements=[
                    ClaimElement(
                        label="A",
                        text="잠재 데이터 표현과 대상 식별 임베딩을 입력으로 하는 제1 처리 모듈 출력 및 제2 처리 모듈 출력",
                        importance="5",
                    )
                ],
            ),
        ]
        caches = [
            {
                "1": [self._item("IA", "동일", "latent representation"), self._item("IB", "대응 없음")],
                "4": [self._item("A", "대응 없음")],
            },
            {
                "1": [self._item("IA", "대응 없음"), self._item("IB", "동일", "target embedding")],
                "4": [self._item("A", "대응 없음")],
            },
            {
                "1": [self._item("IA", "대응 없음"), self._item("IB", "대응 없음")],
                "4": [self._item("A", "차이", "the processing model accepts structured input data")],
            },
        ]

        result = self._write_caches_and_build(claims, caches)
        chain = result["chains"]["4"]
        independent_total = result["chains"]["1"]["total"]

        self.assertEqual(chain["inherited"], independent_total)
        self.assertEqual(chain["added"], [2])
        self.assertEqual(chain["total"], independent_total + [2])
        self.assertFalse(chain["coverage_complete"])
        self.assertEqual(chain["uncovered_labels"], ["A"])


class ConsistencyRegressionTests(unittest.TestCase):
    def test_quote_limit_applies_even_when_input_already_contains_ellipsis(self):
        quote = "A" * 180 + " ... " + "B" * 220
        shortened = _shorten_quote(quote)

        self.assertLessEqual(len(shortened), 350)
        self.assertIn(" ... ", shortened)


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
