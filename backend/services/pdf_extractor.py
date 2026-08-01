"""
PDF 텍스트 추출기
우선순위: PyMuPDF → OpenDataLoader-pdf(폴백)
"""
from __future__ import annotations
import re
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional

from backend.models.schemas import (
    ExtractedDocument,
    PageLayout,
    PageTextBlock,
    PageTextLine,
    PageTextSpan,
    ParagraphRecord,
    PatentChunk,
)

logger = logging.getLogger(__name__)
PDF_EXTRACTOR_SCHEMA_VERSION = 5
# 공보 서지면(발행국·공개번호·명칭)이 들어 있는 선두 구간 길이
_BIBLIOGRAPHIC_HEAD_CHARS = 5_000
# 논문 초록의 현실적인 상한. 섹션 검출이 실패해 본문 전체가 초록으로 묶이는 경우
# 본문까지 통째로 비교 대상에서 제외되는 사고를 막는 안전장치다.
_NON_PATENT_ABSTRACT_MAX_CHARS = 3_000

# 단락번호 패턴:
# - 일반 특허: [0001], 【0001】, (0001)
# - WO/PCT 일부 문헌: [1], [2]처럼 1~3자리 번호
# - OCR 오류: [6[, [91 처럼 닫는 괄호가 깨진 경우
_PARA_PATTERN = re.compile(r"[\[【\(]\s*(0\d{3})\s*[\]】\)]")
_SHORT_PARA_PATTERN = re.compile(r"[\[【\(]\s*([1-9]\d{0,2})\s*[\]】\)\[]?")
# OCR 텍스트 레이어에서 대괄호가 유실되는 경우([0006] → "0006 FIG. ...")가 있어,
# 괄호 매칭이 전무하면 줄 시작의 맨숫자 단락번호로 재시도한다.
# bare 폴백은 기존처럼 0 시작 4자리만 허용해 과검출을 막는다.
_PARA_PATTERN_BARE = re.compile(r"^\s*(0\d{3})(?:\.|\s|$)")
# 청구항 섹션 시작 패턴 (한국 특허 다양한 포맷 모두 지원)
_KR_CLAIMS_START = re.compile(
    r"【\s*(?:특허)?청구(?:의)?\s*범위\s*】"   # 【청구범위】/【청구의 범위】/【특허청구(의) 범위】
    r"|^\s*(?:특허)?청구(?:의)?\s*범위\s*$"    # 줄 단독: 청구범위 / 청구의 범위 / 특허청구범위
    r"|\[\s*(?:특허)?청구(?:의)?\s*범위\s*\]",  # [청구범위] / [특허청구범위]
    re.MULTILINE,
)
_KR_CLAIM_ITEM = re.compile(r"청구항\s*(\d+)")
# 미국특허 청구항 섹션
_US_CLAIMS_START = re.compile(r"^CLAIMS\s*$|^What is claimed", re.MULTILINE | re.IGNORECASE)
_US_CLAIM_ITEM = re.compile(r"^\s*(\d+)\.\s", re.MULTILINE)
_SECTION_HEADINGS = [
    "기술분야",
    "배경기술",
    "해결하려는 과제",
    "과제의 해결수단",
    "발명의 효과",
    "도면의 간단한 설명",
    "발명을 실시하기 위한 구체적인 내용",
    "발명의 실시를 위한 형태",
    "실시예",
    "청구의 범위",
    "청구범위",
    "특허청구범위",
    "CLAIMS",
    "BACKGROUND",
    "SUMMARY",
    "DETAILED DESCRIPTION",
]
_GROUP_BOUNDARY_RE = re.compile(
    r"(제\s*\d+\s*실시예|변형예|다른\s*실시예|도\s*\d+\s*(?:은|는|을|를)?|"
    r"이하에서는|한편|상기와\s*같이\s*구성된|S\d{3}|"
    r"기술분야|배경기술|해결하려는\s*과제|과제의\s*해결수단|발명의\s*효과|"
    r"도면의\s*간단한\s*설명|발명을\s*실시하기\s*위한\s*구체적인\s*내용)",
    re.IGNORECASE,
)
_CLAIM_SECTION_RE = re.compile(r"(?:특허)?청구(?:의)?\s*범위|^CLAIMS$", re.IGNORECASE)
# 공개번호 인식은 KR/US 뿐 아니라 WO(PCT)·EP·JP·CN 공보까지 포함한다.
# 하나라도 빠지면 해당 문헌의 공개번호가 파일명으로 대체되어 인용 이력이 깨진다.
_PUBLICATION_RE = re.compile(
    r"\b("
    r"WO\s*\d{4}\s*/\s*\d{4,6}(?:\s*A\d?)?"           # WO 2023/123456 A1
    r"|WO\s*\d{10,11}(?:\s*A\d?)?"                    # WO2023123456A1
    r"|PCT/[A-Z]{2}\s*\d{4}/\d{4,8}"                  # PCT/KR2022/012345
    r"|EP\s*\d\s?\d{3}\s?\d{3}(?:\s*[AB]\d?)?"        # EP 3 456 789 A1
    r"|KR\s*\d{2}-?\d{4}-?\d{7}"                      # KR 10-2020-0123456
    r"|(?:10|20|30)-\d{4}-\d{7}"                      # 10-2020-0123456 (KIPO 공개/출원)
    r"|(?:10|20|30)-\d{7}"                            # 10-2345678 (KIPO 등록)
    r"|US\s*\d{4}/\d{7}"                              # US 2024/0394445
    r"|US\s*\d{1,2},\d{3},\d{3}"                      # US 11,123,456
    r"|US\s*\d{7,}"                                   # US20240394445
    r"|(?:特開|特表|特願|特許)\s*\d{4}-\d{6}"          # 特開2023-123456
    r"|JP\s*\d{4}-?\d{6}"                             # JP2023-123456
    r"|CN\s*\d{9,12}\s*[A-Z]\b"                       # CN 115123456 A
    r")",
    re.IGNORECASE,
)
# 논문 섹션 표제 패턴.
#
# 표제로 인정하는 형태는 (1) 로마숫자/아라비아숫자 번호 + 대문자 표제어,
# (2) 번호 없는 정형 표제어 두 가지뿐이다. 예전 패턴은 `[I|V|X]+\.\s+[^\n]+`
# 처럼 문자클래스 안에 `|`가 들어가 있어 참고문헌의 저자 이니셜
# ("X. Xu, K. Willis...", "V. Khalidov, ...")까지 섹션 표제로 잡았고,
# 그 결과 참고문헌 목록이 본문으로 취급되는 반면 진짜 본문은 앞 섹션(초록)에
# 흡수되어 통째로 제외되는 문제가 있었다. 표제어를 열거형으로 고정해 막는다.
_NON_PATENT_SECTION_WORDS = (
    r"INTRODUCTION|RELATED\s+WORKS?|METHODS?|METHODOLOGY|APPROACH|PRELIMINARIES"
    r"|MATERIALS?(?:\s+AND\s+METHODS?)?|MODEL|ARCHITECTURE|IMPLEMENTATION(?:\s+DETAILS?)?"
    r"|DATASETS?|EXPERIMENTS?(?:\s+AND\s+RESULTS?)?|EVALUATION|ABLATION(?:\s+STUD(?:Y|IES))?"
    r"|RESULTS?|ANALYSIS|DISCUSSION|LIMITATIONS?|FUTURE\s+WORK|CONCLUSIONS?"
    r"|BACKGROUND(?:\s+OF\s+THE\s+INVENTION)?|APPENDI(?:X|CES)|SUPPLEMENTARY(?:\s+MATERIALS?)?"
)
# 영문 표제어는 번호가 붙은 경우에만 대소문자를 무시한다("3.1 Model Architecture").
# 번호가 없으면 대문자 표기일 때만 표제로 본다. 그렇지 않으면 본문 중간의
# "model ...", "methods ..." 같은 줄이 매번 새 섹션 경계로 잡혀 섹션이 산산조각 난다.
_NON_PATENT_HEADING_RE = re.compile(
    r"^[ \t]*(?:\(\s*\d+\s*\)\s*)?("
    # 번호가 붙은 표제: "I. INTRODUCTION", "3.1 Experiments", "IV EVALUATION"
    r"(?:[IVXL]{1,5}|\d+(?:\.\d+)*)\.?[ \t]+(?i:" + _NON_PATENT_SECTION_WORDS + r")"
    # 번호 없는 표제: 대문자로 시작하면서 그 줄에 단독으로 놓인 경우만 인정한다.
    # 소문자로 시작하면 본문 문장("model is trained ..."), 뒤에 다른 글자가
    # 이어지면 참고문헌 항목("MODEL CARD.md")이므로 표제가 아니다.
    r"|(?=[A-Z])(?i:" + _NON_PATENT_SECTION_WORDS + r"|ABSTRACT|SUMMARY|CLAIMS"
    r"|REFERENCES|BIBLIOGRAPHY|ACKNOWLEDG(?:E)?MENTS?"
    r"|FIELD(?:\s+OF\s+THE\s+INVENTION)?"
    r")(?=[ \t]*[:.]?[ \t]*$)"
    r"|(?i:\d*\s*DETAILED\s+DESCRIPTION(?:\s+OF\s+[^\n]+)?)"
    r"|초록|요약|서론|관련\s*연구|제안\s*방법|실험|결과|고찰|결론|참고문헌|감사의?\s*글|사사"
    r"|도면의\s*간단한\s*설명|발명의\s*상세한\s*설명|발명의\s*목적"
    r"|발명이\s*속하는\s*기술\s*및\s*그\s*분야의\s*종래기술"
    r"|발명이\s*이루고자\s*하는\s*기술적\s*과제|발명의\s*구성\s*및\s*작용"
    r"|(?:특허)?청구(?:의)?\s*범위"
    r")\b",
    re.MULTILINE,
)
_LEGACY_PATENT_RE = re.compile(
    r"(?:대한민국\s*특허청|\(\s*12\s*\)\s*공개특허공보|공개특허\s*10-\d{4}-\d{7}|"
    r"발명의\s*상세한\s*설명|청구의\s*범위)",
    re.IGNORECASE,
)
_INTERNATIONAL_PATENT_RE = re.compile(
    r"(?:"
    # WO / PCT (국제공개공보)
    r"\bWO\s*\d{4}\s*/\s*\d{4,6}\s*[A-Z]\d?\b|"
    r"\bWO\s*\d{10,11}\s*[A-Z]\d?\b|"
    r"\bPCT/[A-Z]{2}\s*\d{4}/\d{4,8}\b|"
    r"\bWIPO\s+PCT\b|"
    r"\bWorld\s+Intellectual\s+Property\s+Organization\b|"
    r"INTERNATIONAL\s+(?:APPLICATION|PUBLICATION)\s+(?:PUBLISHED\s+UNDER\s+THE\s+PATENT\s+COOPERATION\s+TREATY|NUMBER)|"
    # EP (유럽특허공보)
    r"\bEUROPEAN\s+PATENT\s+(?:APPLICATION|SPECIFICATION|BULLETIN)\b|"
    r"\bEuropean\s+Patent\s+Office\b|"
    # US (공개/등록공보)
    r"\bUnited\s+States\s+Patent(?:\s+Application\s+Publication)?\b|"
    r"\bPatent\s+Application\s+Publication\b|"
    # JP (일본공보)
    r"(?:公開特許公報|特許公報|公表特許公報|特開\s*\d{4}-\d{6})|"
    # CN (중국공보)
    r"(?:发明专利申请|发明专利说明书|申请公布号|授权公告号)"
    r")",
    re.IGNORECASE,
)



def extract(pdf_path: str, doc_index: int = 0) -> ExtractedDocument:
    filename = Path(pdf_path).name
    try:
        return _extract_pymupdf(pdf_path, doc_index, filename)
    except Exception as exc:
        logger.warning(f"PyMuPDF failed for {filename}, trying OpenDataLoader-pdf fallback: {exc}")

    doc = _try_opendataloader(pdf_path, doc_index, filename)
    if doc is not None:
        return doc
    raise RuntimeError(
        f"{filename}: PyMuPDF와 OpenDataLoader-pdf 모두 PDF 텍스트 추출에 실패했습니다."
    )


def _looks_like_patent(raw_text: str) -> bool:
    """공보 서지사항으로 특허문헌 여부를 판정한다.

    KR/US/WO/EP/JP/CN 공보는 모두 첫 페이지(서지면)에 발행국·공개번호 표시가
    있으므로 국제 공보 표지는 앞부분에서만 찾는다. 논문 본문이 특허를 인용하며
    공개번호를 언급하는 경우를 특허문헌으로 오인하지 않기 위한 제한이다.
    """
    text = raw_text or ""
    return bool(
        _LEGACY_PATENT_RE.search(text)
        or _INTERNATIONAL_PATENT_RE.search(text[:_BIBLIOGRAPHIC_HEAD_CHARS])
    )


# ---------------------------------------------------------------------------
# OpenDataLoader-pdf (2차 fallback)
# ---------------------------------------------------------------------------

def _try_opendataloader(pdf_path: str, doc_index: int, filename: str) -> Optional[ExtractedDocument]:
    try:
        import opendataloader_pdf  # type: ignore
        import time

        # ODL ignores output_dir and writes to its own ESTsoft temp dir.
        # Record mtime before conversion so we can find the newly created file.
        estsoft_base = Path(r"C:\Users\Public\Documents\ESTsoft\CreatorTemp")
        stem = Path(pdf_path).stem
        before_ts = time.time()

        opendataloader_pdf.convert(
            input_path=[pdf_path],
            output_dir=str(estsoft_base),  # hint (ignored by ODL, but pass anyway)
            format="json",
        )

        # Find the most recently created JSON whose stem matches the PDF stem.
        json_file: Optional[Path] = None
        if estsoft_base.exists():
            candidates = [
                p for p in estsoft_base.rglob("*.json")
                if p.stem == stem and p.stat().st_mtime >= before_ts - 5
            ]
            if candidates:
                json_file = max(candidates, key=lambda p: p.stat().st_mtime)

        if json_file is None:
            logger.warning(f"opendataloader-pdf: JSON not found in {estsoft_base} for {filename}")
            return None

        logger.info(f"opendataloader-pdf: found JSON at {json_file}")
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        return _parse_odl_json(data, doc_index, filename, pdf_path)
    except ImportError:
        logger.info("opendataloader-pdf not installed, skipping")
        return None
    except Exception as e:
        logger.warning(f"opendataloader-pdf error: {e}")
        return None


def _parse_odl_json(data: dict | list, doc_index: int, filename: str, pdf_path: str) -> ExtractedDocument:
    # ODL JSON structure: top-level dict with "kids" list.
    # Each kid: {type, "page number", content, ...}
    # list type kids have nested "list items" each with their own content.
    if isinstance(data, list):
        elements = data
    else:
        elements = data.get("kids", data.get("elements", data.get("content", [])))

    pages: Dict[str, str] = {}
    raw_lines: List[str] = []

    _SKIP_TYPES = {"image", "header", "footer"}

    def _collect(elem: dict) -> None:
        etype = elem.get("type", "")
        if etype in _SKIP_TYPES:
            return
        page = str(elem.get("page number", elem.get("page_number", elem.get("page", 1))))
        text = elem.get("content", elem.get("text", ""))
        if text and text.strip():
            pages.setdefault(page, "")
            pages[page] += text + "\n"
            raw_lines.append(text)
        # Recurse into nested list items
        for sub in elem.get("list items", elem.get("kids", [])):
            if isinstance(sub, dict):
                _collect(sub)

    for elem in elements:
        if isinstance(elem, dict):
            _collect(elem)

    raw_text = "\n".join(raw_lines)
    paragraphs = _extract_paragraphs(raw_text)
    claims = _extract_claims(raw_text, "patent")
    doc_type = "patent" if paragraphs or claims or _looks_like_patent(raw_text) else "non_patent"

    enriched = _build_enriched_document(
        paragraphs=paragraphs,
        pages=pages,
        claims=claims,
        raw_text=raw_text,
        filename=filename,
        doc_index=doc_index,
        pdf_path=pdf_path,
        doc_type=doc_type,
    )
    return enriched


# ---------------------------------------------------------------------------
# PyMuPDF (1차)
# ---------------------------------------------------------------------------

def _extract_pymupdf(pdf_path: str, doc_index: int, filename: str) -> ExtractedDocument:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError("PyMuPDF(fitz)가 설치되어 있지 않습니다. pip install pymupdf")

    doc = fitz.open(pdf_path)
    pages: Dict[str, str] = {}
    page_layouts: List[PageLayout] = []
    raw_lines: List[str] = []

    for i, page in enumerate(doc, start=1):
        page_dict = page.get_text("dict")
        text = page.get_text("text")
        if text:
            pages[str(i)] = text
            raw_lines.append(text)
        page_layouts.append(_parse_pymupdf_page_layout(i, page, page_dict, text or ""))

    doc.close()

    # 스캔 PDF 감지 (페이지당 평균 100자 미만)
    if pages:
        avg_len = sum(len(v) for v in pages.values()) / len(pages)
        if avg_len < 100:
            raise ValueError(
                f"{filename}: 스캔 이미지 PDF로 판단됩니다 (페이지당 평균 {avg_len:.0f}자). "
                "텍스트 레이어가 포함된 PDF를 사용하세요."
            )

    raw_text = "\n".join(raw_lines)
    paragraphs = _extract_paragraphs(raw_text)
    claims = _extract_claims(raw_text, "patent")
    doc_type = "patent" if paragraphs or claims or _looks_like_patent(raw_text) else "non_patent"

    return _build_enriched_document(
        paragraphs=paragraphs,
        pages=pages,
        claims=claims,
        raw_text=raw_text,
        filename=filename,
        doc_index=doc_index,
        pdf_path=pdf_path,
        doc_type=doc_type,
        page_layouts=page_layouts,
    )


# ---------------------------------------------------------------------------
# 공통 파싱 로직
# ---------------------------------------------------------------------------

def _bbox(value) -> List[float]:
    if not value:
        return []
    return [float(v) for v in value[:4]]


def _parse_pymupdf_page_layout(page_no: int, page, page_dict: dict, page_text: str) -> PageLayout:
    blocks: List[PageTextBlock] = []
    for block in page_dict.get("blocks", []) or []:
        block_no = int(block.get("number", len(blocks)))
        lines: List[PageTextLine] = []
        block_text_parts: List[str] = []
        for line in block.get("lines", []) or []:
            spans: List[PageTextSpan] = []
            line_text_parts: List[str] = []
            for span in line.get("spans", []) or []:
                span_text = span.get("text", "")
                if span_text:
                    line_text_parts.append(span_text)
                spans.append(PageTextSpan(
                    text=span_text,
                    bbox=_bbox(span.get("bbox")),
                    font=span.get("font", ""),
                    size=span.get("size"),
                    flags=span.get("flags"),
                    color=span.get("color"),
                ))
            line_text = "".join(line_text_parts)
            if line_text:
                block_text_parts.append(line_text)
            lines.append(PageTextLine(
                bbox=_bbox(line.get("bbox")),
                spans=spans,
            ))
        blocks.append(PageTextBlock(
            block_no=block_no,
            block_type=int(block.get("type", 0)),
            bbox=_bbox(block.get("bbox")),
            text="\n".join(block_text_parts).strip(),
            lines=lines,
        ))
    rect = page.rect
    return PageLayout(
        page_no=page_no,
        width=float(rect.width),
        height=float(rect.height),
        rotation=int(page.rotation or 0),
        text=page_text,
        blocks=blocks,
    )

def _extract_paragraphs(text: str) -> Dict[str, str]:
    """[XXXX] 단락번호 기준으로 텍스트를 분리하여 dict 반환"""
    paragraphs = _split_paragraphs(text, _PARA_PATTERN)
    if not paragraphs:
        short_start = _find_dense_short_paragraph_start(text)
        if short_start is not None:
            short_text = _rewrite_dense_short_paragraph_markers(text[short_start:])
            paragraphs = _split_paragraphs(short_text, _SHORT_PARA_PATTERN)
    if not paragraphs:
        bare = _split_paragraphs(text, _PARA_PATTERN_BARE)
        # 숫자 데이터 줄을 단락번호로 오인하지 않도록 충분히 많을 때만 채택
        if len(bare) >= 5:
            paragraphs = bare

    # 텍스트 커버리지 검증:
    # 파싱된 단락의 전체 길이에 비해 원문 텍스트가 매우 큰 경우(예: 논문 하단 참고문헌만 오인 추출)
    # 단락 파싱을 무효화하고 비특허(non_patent) 문헌 전용 파이프라인으로 넘긴다.
    if paragraphs:
        total_para_len = sum(len(v) for v in paragraphs.values())
        if len(text) >= 5000 and (total_para_len / len(text)) < 0.35:
            logger.warning(
                f"단락 파싱 커버리지 부족 ({total_para_len}/{len(text)} = {total_para_len/len(text):.1%}). "
                "특허 단락 오인으로 판단하여 비특허(non_patent) 문헌 파이프라인으로 전환합니다."
            )
            paragraphs = {}

    return paragraphs


def _split_paragraphs(text: str, pattern: re.Pattern) -> Dict[str, str]:
    paragraphs: Dict[str, str] = {}
    lines = text.splitlines()
    current_key: Optional[str] = None
    current_buf: List[str] = []

    for line in lines:
        m = pattern.search(line)
        if m:
            if current_key is not None:
                paragraphs[current_key] = "\n".join(current_buf).strip()
            current_key = f"[{m.group(1)}]"
            # 단락번호 이후 내용이 같은 줄에 있을 수 있음
            after = line[m.end():].strip()
            current_buf = [after] if after else []
        elif current_key is not None:
            current_buf.append(line)

    if current_key is not None and current_buf:
        paragraphs[current_key] = "\n".join(current_buf).strip()

    return paragraphs


def _normalize_short_marker(raw_marker: str, prev_num: Optional[int], next_num: Optional[int]) -> Optional[int]:
    try:
        value = int(raw_marker)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    if prev_num is not None and next_num is not None and value > 9:
        trimmed = int(str(value)[:-1]) if len(str(value)) > 1 else value
        if trimmed == prev_num + 1 and trimmed == next_num - 1:
            return trimmed
    return value


def _find_dense_short_paragraph_start(text: str) -> Optional[int]:
    matches = list(re.finditer(r"^[ \t]*[\[【\(]\s*([1-9]\d{0,2})\s*[\]】\)\[]?[ \t]*$", text or "", re.MULTILINE))
    if len(matches) < 5:
        return None
    normalized: List[tuple[int, int]] = []
    raw_values = [m.group(1) for m in matches]
    for idx, match in enumerate(matches):
        prev_num = int(raw_values[idx - 1]) if idx > 0 and raw_values[idx - 1].isdigit() else None
        next_num = int(raw_values[idx + 1]) if idx + 1 < len(raw_values) and raw_values[idx + 1].isdigit() else None
        value = _normalize_short_marker(raw_values[idx], prev_num, next_num)
        if value is not None:
            normalized.append((match.start(), value))
    consecutive = 1
    run_start = 0
    for idx in range(1, len(normalized)):
        if normalized[idx][1] == normalized[idx - 1][1] + 1:
            consecutive += 1
            if consecutive >= 5:
                start_pos = normalized[run_start][0]
                if len(text) >= 10000 and start_pos > len(text) * 0.3:
                    return None
                return start_pos
        else:
            consecutive = 1
            run_start = idx
    return None


def _rewrite_dense_short_paragraph_markers(text: str) -> str:
    lines = text.splitlines()
    marker_indexes: List[int] = []
    raw_values: List[str] = []
    for idx, line in enumerate(lines):
        match = re.match(r"^[ \t]*[\[【\(]\s*([1-9]\d{0,2})\s*[\]】\)\[]?[ \t]*$", line)
        if match:
            marker_indexes.append(idx)
            raw_values.append(match.group(1))

    replacements: Dict[int, int] = {}
    for idx, line_index in enumerate(marker_indexes):
        prev_num = int(raw_values[idx - 1]) if idx > 0 and raw_values[idx - 1].isdigit() else None
        next_num = int(raw_values[idx + 1]) if idx + 1 < len(raw_values) and raw_values[idx + 1].isdigit() else None
        value = _normalize_short_marker(raw_values[idx], prev_num, next_num)
        if value is not None:
            replacements[line_index] = value

    for line_index, value in replacements.items():
        lines[line_index] = f"[{value}]"
    return "\n".join(lines)


def _extract_claims(text: str, doc_type: str) -> Dict[str, str]:
    """청구항 섹션 파싱 (한국어 / 미국특허 모두 지원)"""
    claims: Dict[str, str] = {}

    # 한국어 특허 청구항
    kr_match = _KR_CLAIMS_START.search(text)
    if kr_match:
        claims_section = text[kr_match.end():]
        _parse_kr_claims(claims_section, claims)
        if claims:
            return claims

    # 미국특허 청구항 (CLAIMS / What is claimed)
    us_match = _US_CLAIMS_START.search(text)
    if us_match:
        claims_section = text[us_match.end():]
        _parse_us_claims(claims_section, claims)
        if claims:
            return claims

    # 폴백: "청구항 N" 패턴으로 직접 검색
    _parse_kr_claims(text, claims)
    return claims


def _normalize_text(text: str) -> str:
    """검색용 텍스트. 보고서 인용에는 사용하지 않는다."""
    normalized = re.sub(r"[\[\]【】()（）,.;:;\"'“”‘’<>]", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _hash_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _clean_para_no(key: str) -> str:
    m = re.search(r"0\d{3}", key or "")
    return m.group(0) if m else (key or "").strip("[]")


def _extract_publication_no(text: str, filename: str) -> str:
    for source in (text[:_BIBLIOGRAPHIC_HEAD_CHARS], filename):
        m = _PUBLICATION_RE.search(source or "")
        if m:
            return re.sub(r"\s+", "", m.group(1)).upper()
    return Path(filename).stem


def _extract_title(text: str, filename: str) -> str:
    patterns = [
        r"발명의\s*명칭\s*[:：]?\s*([^\n]+)",
        r"\(54\)\s*(?:Title|발명의\s*명칭)\s*[:：]?\s*([^\n]+)",
        r"Title\s*[:：]\s*([^\n]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            title = m.group(1).strip()
            if 2 <= len(title) <= 160:
                return title
    return Path(filename).stem


# 1페이지 상단 상용구(저널 머리글·공개번호·발행일)는 제목 후보에서 제외한다.
_TITLE_BOILERPLATE_RE = re.compile(
    r"JOURNAL\s+OF|PROCEEDINGS|TRANSACTIONS\s+ON|CONFERENCE\s+ON|VOL\.\s*\d"
    r"|^\s*(?:arXiv|doi|DOI|https?://)|PREPRINT|SUBMITTED\s+TO"
    r"|^\s*(?:[A-Z]{2}\s*)?\d{4}/\d{7}|^\s*\(\s*\d{2}\s*\)"
    r"|^\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*\d{1,2},\s*\d{4}\s*$",
    re.IGNORECASE,
)


def _extract_title_from_layout(page_layouts: Optional[List[PageLayout]], fallback: str) -> str:
    """1페이지 최대 글꼴 줄에서 비특허문헌 제목을 추출한다.

    논문에는 특허의 `발명의 명칭`에 해당하는 표제가 없어 예전에는 파일명이
    그대로 제목이 되었다. 논문 제목은 본문보다 뚜렷하게 큰 글꼴로 조판되므로,
    저널 머리글·arXiv 식별자 같은 상용구를 걸러내고 최대 글꼴 줄만 이어 붙인다.
    공보는 제목과 본문의 글꼴 크기 차이가 거의 없어 이 방법을 쓰지 않고,
    `_extract_publication_no`가 뽑는 공개번호를 식별자로 사용한다.
    """
    if not page_layouts:
        return fallback
    first = page_layouts[0]
    sized: List[tuple[float, str]] = []
    for block in first.blocks or []:
        for line in block.lines or []:
            spans = [span for span in (line.spans or []) if (span.text or "").strip()]
            if not spans:
                continue
            size = max(float(span.size or 0) for span in spans)
            line_text = "".join(span.text or "" for span in spans).strip()
            if line_text:
                sized.append((size, line_text))
    if not sized:
        return fallback

    candidates = [
        (size, line) for size, line in sized
        if len(line) >= 4 and not _TITLE_BOILERPLATE_RE.search(line)
    ]
    if not candidates:
        return fallback
    max_size = max(size for size, _ in candidates)
    title_lines = [line for size, line in candidates if size >= max_size - 0.5]
    title = re.sub(r"\s+", " ", " ".join(title_lines)).strip()
    return title if 8 <= len(title) <= 200 else fallback


def _find_page_no(pages: Dict[str, str], para_no: str, para_text: str) -> Optional[int]:
    # Strip leading marker prefix like [1] or [0001]
    clean_text = re.sub(rf"^\[\s*{re.escape(para_no)}\s*\]\s*", "", para_text or "").strip()
    needle = clean_text[:40]
    if needle:
        for page_key, page_text in pages.items():
            if needle in (page_text or ""):
                try:
                    return int(page_key)
                except ValueError:
                    pass

    marker_variants = [f"[{para_no}]", f"【{para_no}】", f"({para_no})"]
    if len(para_no) > 2:
        marker_variants.append(para_no)

    for page_key, page_text in pages.items():
        page = page_text or ""
        if any(marker in page for marker in marker_variants):
            try:
                return int(page_key)
            except ValueError:
                return None
    return None


def _section_positions(text: str) -> List[tuple[int, str]]:
    positions: List[tuple[int, str]] = []
    for heading in _SECTION_HEADINGS:
        for m in re.finditer(re.escape(heading), text, re.IGNORECASE):
            positions.append((m.start(), heading))
    return sorted(positions, key=lambda x: x[0])


def _section_for_paragraph(raw_text: str, para_no: str, positions: List[tuple[int, str]]) -> str:
    marker = re.search(rf"[\[【\(]\s*{re.escape(para_no)}\s*[\]】\)]", raw_text)
    idx = marker.start() if marker else -1
    section = ""
    for pos, heading in positions:
        if idx >= 0 and pos <= idx:
            section = heading
        elif idx >= 0:
            break
    return section


def _reference_signs(text: str) -> List[str]:
    signs = re.findall(r"\((\d{2,4}[a-zA-Z]?)\)", text or "")
    seen: set[str] = set()
    result: List[str] = []
    for sign in signs:
        if sign not in seen:
            seen.add(sign)
            result.append(sign)
    return result[:30]


def _figure_no(text: str) -> Optional[str]:
    m = re.search(r"도\s*\d+[A-Za-z가-힣]?", text or "")
    return m.group(0).replace(" ", "") if m else None


def _is_claim_paragraph(section: str, text: str) -> bool:
    if _CLAIM_SECTION_RE.search(section or ""):
        return True
    return bool(re.match(r"^\s*(청구항\s*)?\d+\s*\.", text or ""))


def _build_paragraph_records(
    paragraphs: Dict[str, str],
    pages: Dict[str, str],
    raw_text: str,
    filename: str,
    doc_index: int,
    publication_no: str,
    title: str,
) -> List[ParagraphRecord]:
    doc_id = f"D{doc_index + 1}"
    positions = _section_positions(raw_text)
    records: List[ParagraphRecord] = []
    for key, body in paragraphs.items():
        para_no = _clean_para_no(key)
        original_text = f"[{para_no}] {(body or '').strip()}".strip()
        section = _section_for_paragraph(raw_text, para_no, positions)
        excluded = _is_claim_paragraph(section, body)
        records.append(ParagraphRecord(
            doc_id=doc_id,
            publication_no=publication_no,
            title=title,
            page_no=_find_page_no(pages, para_no, body),
            section=section,
            paragraph_no=para_no,
            claim_no=None,
            figure_no=_figure_no(body),
            reference_signs=_reference_signs(body),
            original_text=original_text,
            normalized_text=_normalize_text(original_text),
            text_hash=_hash_text(original_text),
            chunk_excluded=excluded,
            exclusion_reason="prior_claim" if excluded else "",
        ))
    return records


def _paragraph_chunks(records: List[ParagraphRecord]) -> List[PatentChunk]:
    chunks: List[PatentChunk] = []
    for rec in records:
        if rec.chunk_excluded:
            continue
        chunks.append(PatentChunk(
            chunk_type="paragraph",
            chunk_id=f"{rec.doc_id}-P-{rec.paragraph_no}",
            doc_id=rec.doc_id,
            publication_no=rec.publication_no,
            title=rec.title,
            section=rec.section,
            paragraph_no=rec.paragraph_no,
            paragraph_range=[rec.paragraph_no],
            page_no=rec.page_no,
            page_range=[rec.page_no] if rec.page_no is not None else [],
            original_text=rec.original_text,
            normalized_text=rec.normalized_text,
            text_hash=rec.text_hash,
            source="description",
        ))
    return chunks


def _group_label(section: str, text: str) -> str:
    source = f"{section} {text}"
    if re.search(r"효과", source):
        return "EFFECT"
    if re.search(r"과제|문제|목적", source):
        return "PROBLEM"
    if re.search(r"해결수단|수단|구성", source):
        return "SOLUTION"
    if re.search(r"도\s*\d+|도면", source):
        return "DRAWING"
    if re.search(r"S\d{3}|흐름|제어", source, re.IGNORECASE):
        return "CONTROL-FLOW"
    if re.search(r"실시예|구체적인\s*내용|DETAILED", source, re.IGNORECASE):
        return "DETAIL-EMBODIMENT"
    return "SUMMARY"


def _group_chunks(records: List[ParagraphRecord]) -> List[PatentChunk]:
    groups: List[List[ParagraphRecord]] = []
    current: List[ParagraphRecord] = []
    current_section = ""

    for rec in records:
        if rec.chunk_excluded:
            continue
        boundary = False
        if current and rec.section and rec.section != current_section:
            boundary = True
        if current and _GROUP_BOUNDARY_RE.search(rec.original_text):
            boundary = True
        if current and len(current) >= 5:
            boundary = True
        if boundary:
            groups.append(current)
            current = []
        current.append(rec)
        current_section = rec.section or current_section
    if current:
        groups.append(current)

    chunks: List[PatentChunk] = []
    counters: Dict[str, int] = {}
    for group in groups:
        first = group[0]
        label = _group_label(first.section, " ".join(r.original_text for r in group[:2]))
        counters[label] = counters.get(label, 0) + 1
        paras = [r.paragraph_no for r in group]
        pages = sorted({r.page_no for r in group if r.page_no is not None})
        original = "\n".join(r.original_text for r in group)
        chunks.append(PatentChunk(
            chunk_type="group",
            chunk_id=f"{first.doc_id}-{label}-{counters[label]:03d}",
            doc_id=first.doc_id,
            publication_no=first.publication_no,
            title=first.title,
            section=first.section,
            paragraph_range=paras,
            page_range=pages,
            original_text=original,
            normalized_text=_normalize_text(original),
            text_hash=_hash_text(original),
            source="description",
        ))
    return chunks


def _is_references_section(section_name: str) -> bool:
    return bool(re.search(r"REFERENCES|BIBLIOGRAPHY|참고문헌", section_name or "", re.IGNORECASE))


def _is_non_patent_summary_section(section_name: str) -> bool:
    return bool(re.search(r"^\s*(?:ABSTRACT|SUMMARY|초록|요약)\b", section_name or "", re.IGNORECASE))


def _is_non_patent_back_matter_section(section_name: str) -> bool:
    return bool(
        re.search(
            r"^\s*(?:ACKNOWLEDG(?:E)?MENTS?|감사의?\s*글|사사)\b",
            section_name or "",
            re.IGNORECASE,
        )
    )


def _dense_non_patent_blocks(
    section_text: str,
    *,
    target_chars: int = 2_000,
    overlap_chars: int = 400,
) -> List[str]:
    """Build paragraph-aware non-patent chunks with a small trailing overlap."""
    paragraphs = [
        block.strip()
        for block in re.split(r"\n\s*\n", section_text or "")
        if block.strip()
    ]
    if not paragraphs:
        return []

    blocks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0
    for paragraph in paragraphs:
        if buffer and buffer_len + len(paragraph) + 2 > target_chars:
            blocks.append("\n\n".join(buffer))
            overlap: List[str] = []
            overlap_len = 0
            for previous in reversed(buffer):
                overlap.insert(0, previous)
                overlap_len += len(previous) + 2
                if overlap_len >= overlap_chars:
                    break
            buffer = overlap
            buffer_len = sum(len(item) + 2 for item in buffer)

        # 긴 단일 문단은 문장 경계를 우선해 조밀한 고정 길이 블록으로 나눈다.
        if len(paragraph) > target_chars and not buffer:
            sentences = re.split(r"(?<=[.!?。！？])\s+", paragraph)
            for sentence in sentences:
                if buffer and buffer_len + len(sentence) + 1 > target_chars:
                    blocks.append(" ".join(buffer))
                    tail = blocks[-1][-overlap_chars:].strip()
                    buffer = [tail] if tail else []
                    buffer_len = len(tail)
                buffer.append(sentence)
                buffer_len += len(sentence) + 1
            continue

        buffer.append(paragraph)
        buffer_len += len(paragraph) + 2

    if buffer:
        candidate = "\n\n".join(buffer).strip()
        if candidate and (not blocks or candidate != blocks[-1]):
            blocks.append(candidate)
    return blocks


def _build_non_patent_records_and_chunks(
    raw_text: str,
    pages: Dict[str, str],
    filename: str,
    doc_index: int,
    publication_no: str,
    title: str,
) -> tuple[List[ParagraphRecord], List[PatentChunk], List[PatentChunk]]:
    doc_id = f"D{doc_index + 1}"
    records: List[ParagraphRecord] = []
    para_chunks: List[PatentChunk] = []
    group_chunks: List[PatentChunk] = []

    matches = list(_NON_PATENT_HEADING_RE.finditer(raw_text or ""))
    sections: List[tuple[str, str]] = []

    if matches:
        if matches[0].start() > 0:
            preamble = raw_text[:matches[0].start()].strip()
            if preamble:
                sections.append(("ABSTRACT", preamble))
        for idx, match in enumerate(matches):
            sec_name = match.group(1).strip().replace("\n", " ")
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)
            sec_text = raw_text[start:end].strip()
            if sec_text:
                sections.append((sec_name, sec_text))
    else:
        blocks = [b.strip() for b in (raw_text or "").split("\n\n") if b.strip()]
        curr_buf: List[str] = []
        curr_len = 0
        sec_idx = 1
        for block in blocks:
            curr_buf.append(block)
            curr_len += len(block)
            if curr_len >= 1000:
                sections.append((f"SECTION-{sec_idx:02d}", "\n\n".join(curr_buf)))
                curr_buf = []
                curr_len = 0
                sec_idx += 1
        if curr_buf:
            sections.append((f"SECTION-{sec_idx:02d}", "\n\n".join(curr_buf)))

    para_idx = 1
    group_idx = 1

    for sec_name, sec_text in sections:
        is_ref = _is_references_section(sec_name)
        is_summary = _is_non_patent_summary_section(sec_name)
        is_back_matter = _is_non_patent_back_matter_section(sec_name)
        is_claim = bool(_CLAIM_SECTION_RE.search(sec_name or ""))
        sub_blocks = _dense_non_patent_blocks(sec_text)
        summary_chars = 0

        for sub in sub_blocks:
            sub = sub.strip()
            if not sub:
                continue
            # 초록/요약은 앞부분만 제외한다. 섹션 검출이 실패해 본문 전체가
            # 초록 하나로 묶이면 예전에는 문헌이 통째로 비교 대상에서
            # 빠졌으므로, 상한을 넘어선 분량은 본문으로 되돌린다.
            summary_excluded = is_summary and summary_chars < _NON_PATENT_ABSTRACT_MAX_CHARS
            if is_summary:
                summary_chars += len(sub)
            is_excluded = is_ref or summary_excluded or is_back_matter or is_claim
            para_no = f"P{para_idx:03d}"
            page_no = _find_page_no(pages, para_no, sub)

            rec = ParagraphRecord(
                doc_id=doc_id,
                publication_no=publication_no,
                title=title,
                page_no=page_no,
                section=sec_name,
                paragraph_no=para_no,
                claim_no=None,
                figure_no=_figure_no(sub),
                reference_signs=_reference_signs(sub),
                original_text=sub,
                normalized_text=_normalize_text(sub),
                text_hash=_hash_text(sub),
                chunk_excluded=is_excluded,
                exclusion_reason=(
                    "references"
                    if is_ref
                    else "summary"
                    if summary_excluded
                    else "back_matter"
                    if is_back_matter
                    else "prior_claim"
                    if is_claim
                    else ""
                ),
            )
            records.append(rec)

            if not is_excluded:
                p_chunk = PatentChunk(
                    chunk_type="paragraph",
                    chunk_id=f"{doc_id}-P-{para_no}",
                    doc_id=doc_id,
                    publication_no=publication_no,
                    title=title,
                    section=sec_name,
                    paragraph_no=para_no,
                    paragraph_range=[para_no],
                    page_no=page_no,
                    page_range=[page_no] if page_no is not None else [],
                    original_text=sub,
                    normalized_text=_normalize_text(sub),
                    text_hash=_hash_text(sub),
                    source="description",
                )
                para_chunks.append(p_chunk)

                g_label = _group_label(sec_name, sub)
                g_chunk = PatentChunk(
                    chunk_type="group",
                    chunk_id=f"{doc_id}-{g_label}-{group_idx:03d}",
                    doc_id=doc_id,
                    publication_no=publication_no,
                    title=title,
                    section=sec_name,
                    paragraph_range=[para_no],
                    page_range=[page_no] if page_no is not None else [],
                    original_text=sub,
                    normalized_text=_normalize_text(sub),
                    text_hash=_hash_text(sub),
                    source="description",
                )
                group_chunks.append(g_chunk)
                group_idx += 1

            para_idx += 1

    return records, para_chunks, group_chunks


def _build_enriched_document(
    paragraphs: Dict[str, str],
    pages: Dict[str, str],
    claims: Dict[str, str],
    raw_text: str,
    filename: str,
    doc_index: int,
    pdf_path: str,
    doc_type: str,
    page_layouts: Optional[List[PageLayout]] = None,
) -> ExtractedDocument:
    publication_no = _extract_publication_no(raw_text, filename)
    title = _extract_title(raw_text, filename)
    if doc_type == "non_patent" and title == Path(filename).stem:
        title = _extract_title_from_layout(page_layouts, title)

    if doc_type == "non_patent" or not paragraphs:
        records, para_chunks, group_chunks = _build_non_patent_records_and_chunks(
            raw_text, pages, filename, doc_index, publication_no, title
        )
    else:
        records = _build_paragraph_records(
            paragraphs, pages, raw_text, filename, doc_index, publication_no, title
        )
        para_chunks = _paragraph_chunks(records)
        group_chunks = _group_chunks(records)

    return ExtractedDocument(
        document_type=doc_type,
        pdf_path=str(Path(pdf_path).resolve()),
        paragraphs=paragraphs,
        paragraph_records=records,
        paragraph_chunks=para_chunks,
        group_chunks=group_chunks,
        pages=pages,
        page_layouts=page_layouts or [],
        claims=claims,
        raw_text=raw_text,
        filename=filename,
        doc_index=doc_index,
        doc_id=f"D{doc_index + 1}",
        publication_no=publication_no,
        title=title,
        metadata={
            "extractor_schema_version": str(PDF_EXTRACTOR_SCHEMA_VERSION),
            "publication_no": publication_no,
            "title": title,
            "source_filename": filename,
        },
    )


def _parse_kr_claims(section: str, claims: Dict[str, str]) -> None:
    positions = [(m.start(), m.group(1)) for m in _KR_CLAIM_ITEM.finditer(section)]
    for i, (start, num) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(section)
        body = section[start:end].strip()
        # "청구항 N" 헤더 제거
        body = _KR_CLAIM_ITEM.sub("", body, count=1).strip()
        claims[num] = body


def _parse_us_claims(section: str, claims: Dict[str, str]) -> None:
    positions = [(m.start(), m.group(1)) for m in _US_CLAIM_ITEM.finditer(section)]
    for i, (start, num) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(section)
        body = section[start:end].strip()
        body = re.sub(r"^\d+\.\s*", "", body).strip()
        claims[num] = body
