import unittest

from backend.services.pdf_extractor import _extract_paragraphs


class PdfExtractorParagraphTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
