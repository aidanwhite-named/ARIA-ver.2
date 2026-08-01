import unittest

from backend.services.pdf_extractor import (
    _build_enriched_document,
    _extract_claims,
    _extract_paragraphs,
    _extract_publication_no,
    _looks_like_patent,
)
from backend.services.citation_extractor import _doc_chunks


class PdfExtractorParagraphTests(unittest.TestCase):
    def test_wo_pct_front_matter_is_recognized_as_patent_without_numbered_paragraphs(self):
        raw_text = "\n".join([
            "WO 2021/068061 A1",
            "WIPO PCT",
            "(12) INTERNATIONAL APPLICATION PUBLISHED UNDER THE PATENT COOPERATION TREATY (PCT)",
            "PCT/CA2020/051337",
            "SYSTEM AND METHOD FOR GENERATING 3D MODELS",
            "The system extracts relevant data and generates a 3-dimensional model.",
        ])

        self.assertEqual(_extract_paragraphs(raw_text), {})
        self.assertTrue(_looks_like_patent(raw_text))

    def test_unnumbered_wo_keeps_detailed_description_and_excludes_claims(self):
        raw_text = "\n".join([
            "WO 2021/068061 A1",
            "Summary",
            "A short summary of the invention.",
            "Detailed Description of Certain Aspects",
            "The extraction module applies computer vision to a floorplan.",
            "The generation module creates a 3D model from the extracted data.",
            "Claims",
            "1. A system comprising an extraction module.",
        ])
        doc = _build_enriched_document(
            paragraphs={},
            pages={"1": raw_text},
            claims={"1": "A system comprising an extraction module."},
            raw_text=raw_text,
            filename="WO2021068061A1.pdf",
            doc_index=0,
            pdf_path="WO2021068061A1.pdf",
            doc_type="patent",
        )

        comparison_text = "\n".join(text for _, text in _doc_chunks(doc))

        self.assertIn("computer vision", comparison_text)
        self.assertIn("creates a 3D model", comparison_text)
        self.assertNotIn("A system comprising an extraction module", comparison_text)

    def test_detected_claims_are_a_patent_classification_signal(self):
        raw_text = "\n".join([
            "What is claimed is:",
            "1. A system comprising a processor.",
            "2. The system of claim 1, further comprising a memory.",
        ])

        claims = _extract_claims(raw_text, "patent")

        self.assertEqual(set(claims), {"1", "2"})
        self.assertTrue(claims)

    def test_legacy_korean_patent_without_numbered_paragraphs_keeps_description_sections(self):
        raw_text = "\n".join([
            "(12) 공개특허공보(A)",
            "요약",
            "이동체 제어 장치의 요약 내용",
            "발명의 상세한 설명",
            "발명의 목적",
            "발명이 속하는 기술 및 그 분야의 종래기술",
            "종래 기술 설명",
            "발명이 이루고자 하는 기술적 과제",
            "해결 과제 설명",
            "발명의 구성 및 작용",
            "다중 마이크로 음원의 방향을 추정하고 위치를 보정한다.",
            "청구의 범위",
            "청구항1. 이동체 제어 장치",
        ])

        doc = _build_enriched_document(
            paragraphs={},
            pages={"1": raw_text},
            claims={"1": "이동체 제어 장치"},
            raw_text=raw_text,
            filename="KR20060086231A.pdf",
            doc_index=0,
            pdf_path="KR20060086231A.pdf",
            doc_type="patent",
        )

        sections = {chunk.section for chunk in doc.paragraph_chunks}
        comparison_text = "\n".join(text for _, text in _doc_chunks(doc))
        self.assertEqual(doc.document_type, "patent")
        self.assertIn("발명의 구성 및 작용", sections)
        self.assertNotIn("청구의 범위", sections)
        self.assertNotIn("이동체 제어 장치의 요약 내용", comparison_text)
        self.assertIn("다중 마이크로 음원의 방향을 추정", comparison_text)

    def test_extracts_zero_padded_paragraphs(self):
        text = "\n".join([
            "[0001]",
            "첫 번째 단락",
            "[0002]",
            "두 번째 단락",
        ])

        paragraphs = _extract_paragraphs(text)

        self.assertEqual(paragraphs, {
            "[0001]": "첫 번째 단락",
            "[0002]": "두 번째 단락",
        })

    def test_extracts_short_wo_style_paragraphs_when_dense(self):
        text = "\n".join([
            "WO 2020/085614",
            "[1]",
            "명세서",
            "[2]",
            "기술분야",
            "[3]",
            "배경기술",
            "[4]",
            "추가 설명",
            "[5]",
            "마지막 설명",
        ])

        paragraphs = _extract_paragraphs(text)

        self.assertEqual(list(paragraphs.keys())[:5], ["[1]", "[2]", "[3]", "[4]", "[5]"])
        self.assertEqual(paragraphs["[3]"], "배경기술")

    def test_extracts_short_wo_style_paragraphs_with_broken_brackets(self):
        text = "\n".join([
            "WO 2020/085614",
            "[6[",
            "여섯 번째 단락",
            "[7]",
            "일곱 번째 단락",
            "[8]",
            "여덟 번째 단락",
            "[91",
            "아홉 번째 단락",
            "[10]",
            "열 번째 단락",
        ])

        paragraphs = _extract_paragraphs(text)

        self.assertEqual(list(paragraphs.keys())[:5], ["[6]", "[7]", "[8]", "[9]", "[10]"])
        self.assertEqual(paragraphs["[9]"], "아홉 번째 단락")

    def test_ignores_sparse_short_bracket_numbers(self):
        text = "\n".join([
            "참고문헌 [1] 및 [3]을 검토하였다.",
            "도 2의 구성은 다음과 같다.",
            "표 5는 측정값이다.",
        ])

        paragraphs = _extract_paragraphs(text)

        self.assertEqual(paragraphs, {})

    def test_academic_paper_with_trailing_references_treated_as_non_patent(self):
        long_body = "\n\n".join([
            "I. INTRODUCTION\n" + ("This is intro content. " * 50),
            "II. METHODS\n" + ("This is methods content. " * 50),
            "III. RESULTS\n" + ("This is results content. " * 50),
            "IV. DISCUSSION\n" + ("This is discussion content. " * 50),
        ]) * 3  # ~12,000 chars
        refs = "\n".join([f"[{i}] Author {i}, Title {i}, 2020." for i in range(1, 15)])  # ~400 chars
        raw_text = long_body + "\n\nREFERENCES\n" + refs

        paragraphs = _extract_paragraphs(raw_text)
        self.assertEqual(paragraphs, {}, "Academic paper with low coverage of bracketed refs should return empty paragraphs")

        from backend.services.pdf_extractor import _build_enriched_document
        doc = _build_enriched_document(
            paragraphs=paragraphs,
            pages={"1": raw_text[:5000], "2": raw_text[5000:]},
            claims={},
            raw_text=raw_text,
            filename="paper.pdf",
            doc_index=1,
            pdf_path="paper.pdf",
            doc_type="non_patent",
        )

        self.assertEqual(doc.document_type, "non_patent")
        self.assertGreater(len(doc.group_chunks), 5)
        self.assertTrue(any("methods content" in c.original_text for c in doc.group_chunks))
        # Ensure references section chunk is excluded from group_chunks
        self.assertFalse(any("REFERENCES" in c.section and "Author 1" in c.original_text for c in doc.group_chunks))

    def test_non_patent_chunks_exclude_summary_and_use_overlap(self):
        from backend.services.pdf_extractor import _build_enriched_document

        abstract = "ABSTRACT\n" + ("Abstract-only disclosure. " * 80)
        method_paragraphs = [
            f"Method paragraph {idx}. " + ("technical operation and result. " * 20)
            for idx in range(1, 9)
        ]
        methods = "II. METHODS\n" + "\n\n".join(method_paragraphs)
        references = "REFERENCES\n[1] Author, unrelated title, 2020."
        raw_text = "\n\n".join([abstract, methods, references])

        doc = _build_enriched_document(
            paragraphs={},
            pages={"1": raw_text},
            claims={},
            raw_text=raw_text,
            filename="dense-paper.pdf",
            doc_index=0,
            pdf_path="dense-paper.pdf",
            doc_type="non_patent",
        )

        chunk_texts = [chunk.original_text for chunk in doc.paragraph_chunks]
        self.assertGreater(len(chunk_texts), 1)
        self.assertFalse(any("Abstract-only disclosure" in text for text in chunk_texts))
        self.assertFalse(any("unrelated title" in text for text in chunk_texts))
        self.assertTrue(
            any(
                set(chunk_texts[idx].splitlines()) & set(chunk_texts[idx + 1].splitlines())
                for idx in range(len(chunk_texts) - 1)
            )
        )

class PublicationRecognitionTests(unittest.TestCase):
    """KR 이외의 공보(WO/EP/JP/CN/US 등록)도 특허문헌으로 인식해야 한다."""

    CASES = {
        "WO": (
            "(12) INTERNATIONAL APPLICATION PUBLISHED UNDER THE PATENT COOPERATION TREATY\n"
            "(19) World Intellectual Property Organization\n"
            "(10) International Publication Number WO 2023/123456 A1\n"
            "(21) International Application Number: PCT/KR2022/012345",
            "WO2023/123456A1",
        ),
        "WO_compact": (
            "WO2023123456A1 International Bureau (54) Title: SOMETHING",
            "WO2023123456A1",
        ),
        "EP": ("EUROPEAN PATENT APPLICATION EP 3 456 789 A1", "EP3456789A1"),
        "JP": ("公開特許公報 (11)特許出願公開番号 特開2023-123456", "特開2023-123456"),
        "CN": ("发明专利申请 (10)申请公布号 CN 115123456 A", "CN115123456A"),
        "US_granted": ("United States Patent US 11,123,456 B2", "US11,123,456"),
        "US_pub": (
            "United States Patent Application Publication US 2024/0394445 A1",
            "US2024/0394445",
        ),
        "KR_bare": (
            "대한민국특허청 공개특허공보 공개번호 10-2020-0123456",
            "10-2020-0123456",
        ),
    }

    def test_publication_office_front_pages_are_recognized(self):
        for name, (raw_text, expected_no) in self.CASES.items():
            with self.subTest(office=name):
                self.assertTrue(_looks_like_patent(raw_text))
                self.assertEqual(
                    _extract_publication_no(raw_text, f"{name}.pdf"), expected_no
                )

    def test_paper_citing_a_patent_number_is_not_treated_as_patent(self):
        raw_text = (
            "CAD-MLLM: Unifying Multimodality-Conditioned CAD Generation\n"
            "Abstract—This paper aims to design a unified system.\n"
            "arXiv:2411.04954"
        )
        self.assertFalse(_looks_like_patent(raw_text))


class NonPatentSectionDetectionTests(unittest.TestCase):
    def test_bibliography_author_initials_are_not_treated_as_section_headings(self):
        """`X. Xu, ...` 같은 저자 이니셜이 로마숫자 표제로 잡히면 참고문헌이
        본문으로 취급되고 진짜 본문은 앞 섹션에 흡수되어 사라진다."""
        body = "I. INTRODUCTION\n" + ("Intro sentence about the encoder. " * 40)
        refs = "\n".join([
            "REFERENCES",
            "X. Xu, J. Lambourne, and P. Jayaraman, “Brepgen,” 2024.",
            "V. Khalidov, P. Fernandez, and D. Haziza, “Dinov2,” 2023.",
            "I. Loshchilov and F. Hutter, “Decoupled weight decay,” 2019.",
        ])
        doc = _build_enriched_document(
            paragraphs={},
            pages={"1": body},
            claims={},
            raw_text=body + "\n\n" + refs,
            filename="paper.pdf",
            doc_index=0,
            pdf_path="paper.pdf",
            doc_type="non_patent",
        )

        sections = {rec.section for rec in doc.paragraph_records}
        self.assertNotIn("X. Xu, J. Lambourne, and P. Jayaraman, “Brepgen,” 2024.", sections)
        comparison_text = "\n".join(text for _, text in _doc_chunks(doc))
        self.assertIn("Intro sentence about the encoder", comparison_text)
        self.assertNotIn("Decoupled weight decay", comparison_text)

    def test_body_survives_when_section_detection_labels_everything_abstract(self):
        """섹션 표제를 하나도 찾지 못해 본문 전체가 초록으로 묶여도,
        앞부분 상한을 넘는 분량은 본문으로 남아야 한다."""
        raw_text = "Abstract\n" + "\n\n".join(
            f"Body paragraph {idx}: the encoder fuses point cloud and image features. " * 12
            for idx in range(1, 20)
        )
        doc = _build_enriched_document(
            paragraphs={},
            pages={"1": raw_text},
            claims={},
            raw_text=raw_text,
            filename="paper.pdf",
            doc_index=0,
            pdf_path="paper.pdf",
            doc_type="non_patent",
        )

        kept = [rec for rec in doc.paragraph_records if not rec.chunk_excluded]
        self.assertTrue(kept, "본문이 통째로 제외되면 안 된다")
        self.assertTrue(any("Body paragraph 19" in rec.original_text for rec in kept))
        self.assertTrue(_doc_chunks(doc))


if __name__ == "__main__":
    unittest.main()
