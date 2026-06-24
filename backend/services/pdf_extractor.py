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

# 단락번호 패턴: [0001], 【0001】 등 0으로 시작하는 4자리만 인식.
# 0 시작 제한으로 논문의 인용 연도(2024)·도면부호(1706) 오인식을 막는다.
# (PDF 추출 시 (0146]처럼 괄호 짝이 깨지는 경우가 있어 여닫기 혼용은 허용)
_PARA_PATTERN = re.compile(r"[\[【\(]\s*(0\d{3})\s*[\]】\)]")
# OCR 텍스트 레이어에서 대괄호가 유실되는 경우([0006] → "0006 FIG. ...")가 있어,
# 괄호 매칭이 전무하면 줄 시작의 맨숫자 단락번호로 재시도한다.
_PARA_PATTERN_BARE = re.compile(r"^\s*(0\d{3})(?:\.|\s|$)")
# 청구항 섹션 시작 패턴 (한국 특허 다양한 포맷 모두 지원)
_KR_CLAIMS_START = re.compile(
    r"【\s*청구의\s*범위\s*】"          # 【청구의 범위】
    r"|【\s*특허청구의?\s*범위\s*】"     # 【특허청구의 범위】 / 【특허청구범위】
    r"|^\s*청구의\s*범위\s*$"            # 줄 단독: 청구의 범위
    r"|^\s*특허청구(의)?\s*범위\s*$"     # 줄 단독: 특허청구범위 / 특허청구의 범위
    r"|\[청구의\s*범위\]"                # [청구의 범위]
    r"|\[특허청구(의)?\s*범위\]",        # [특허청구범위]
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
_CLAIM_SECTION_RE = re.compile(r"청구의\s*범위|특허청구(?:의)?\s*범위|^CLAIMS$", re.IGNORECASE)
_PUBLICATION_RE = re.compile(
    r"(KR\s*\d{2}-?\d{4}-?\d{7}|KR\s*10-?\d{4}-?\d{7}|US\s*\d{4}/\d{7}|US\s*\d{7,})",
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
    doc_type = "patent" if paragraphs else "non_patent"
    claims = _extract_claims(raw_text, doc_type)

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
    doc_type = "patent" if paragraphs else "non_patent"
    claims = _extract_claims(raw_text, doc_type)

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
        bare = _split_paragraphs(text, _PARA_PATTERN_BARE)
        # 숫자 데이터 줄을 단락번호로 오인하지 않도록 충분히 많을 때만 채택
        if len(bare) >= 5:
            paragraphs = bare
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
    for source in (text[:5000], filename):
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


def _find_page_no(pages: Dict[str, str], para_no: str, para_text: str) -> Optional[int]:
    marker_variants = [f"[{para_no}]", f"【{para_no}】", f"({para_no})", para_no]
    needle = (para_text or "").strip()[:40]
    for page_key, page_text in pages.items():
        page = page_text or ""
        if any(marker in page for marker in marker_variants) or (needle and needle in page):
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
    records = _build_paragraph_records(
        paragraphs, pages, raw_text, filename, doc_index, publication_no, title
    )
    return ExtractedDocument(
        document_type=doc_type,
        pdf_path=str(Path(pdf_path).resolve()),
        paragraphs=paragraphs,
        paragraph_records=records,
        paragraph_chunks=_paragraph_chunks(records),
        group_chunks=_group_chunks(records),
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
