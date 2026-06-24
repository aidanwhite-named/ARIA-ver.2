from __future__ import annotations
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class PageTextSpan(BaseModel):
    text: str = ""
    bbox: List[float] = Field(default_factory=list)
    font: str = ""
    size: Optional[float] = None
    flags: Optional[int] = None
    color: Optional[int] = None


class PageTextLine(BaseModel):
    bbox: List[float] = Field(default_factory=list)
    spans: List[PageTextSpan] = Field(default_factory=list)


class PageTextBlock(BaseModel):
    block_no: int = 0
    block_type: int = 0
    bbox: List[float] = Field(default_factory=list)
    text: str = ""
    lines: List[PageTextLine] = Field(default_factory=list)


class PageLayout(BaseModel):
    page_no: int
    width: Optional[float] = None
    height: Optional[float] = None
    rotation: int = 0
    text: str = ""
    blocks: List[PageTextBlock] = Field(default_factory=list)


class ParagraphRecord(BaseModel):
    doc_id: str = ""
    publication_no: str = ""
    title: str = ""
    page_no: Optional[int] = None
    section: str = ""
    paragraph_no: str = ""
    claim_no: Optional[str] = None
    figure_no: Optional[str] = None
    reference_signs: List[str] = Field(default_factory=list)
    original_text: str = ""
    normalized_text: str = ""
    text_hash: str = ""
    chunk_excluded: bool = False
    exclusion_reason: str = ""


class PatentChunk(BaseModel):
    chunk_type: str = "paragraph"  # "paragraph" | "group"
    chunk_id: str = ""
    doc_id: str = ""
    publication_no: str = ""
    title: str = ""
    section: str = ""
    paragraph_no: Optional[str] = None
    paragraph_range: List[str] = Field(default_factory=list)
    page_no: Optional[int] = None
    page_range: List[int] = Field(default_factory=list)
    original_text: str = ""
    normalized_text: str = ""
    text_hash: str = ""
    source: str = "description"


class ExtractedDocument(BaseModel):
    document_type: str = "patent"  # "patent" | "non_patent"
    pdf_path: str = ""             # 원본 PDF 경로(인용문 검증 시 전체 본문 조회에 사용)
    paragraphs: Dict[str, str] = Field(default_factory=dict)
    paragraph_records: List[ParagraphRecord] = Field(default_factory=list)
    paragraph_chunks: List[PatentChunk] = Field(default_factory=list)
    group_chunks: List[PatentChunk] = Field(default_factory=list)
    pages: Dict[str, str] = Field(default_factory=dict)
    page_layouts: List[PageLayout] = Field(default_factory=list)
    claims: Dict[str, str] = Field(default_factory=dict)
    raw_text: str = ""
    filename: str = ""
    doc_index: int = 0
    doc_id: str = ""
    publication_no: str = ""
    title: str = ""
    metadata: Dict[str, str] = Field(default_factory=dict)


class ClaimElement(BaseModel):
    label: str
    text: str
    importance: str = "3"  # 1~5 중요도
    is_sub: bool = False   # True if sub-component (A-1, B-1, ...)
    parent_label: Optional[str] = None  # e.g. "A" for A-1


class ParsedClaim(BaseModel):
    claim_number: int
    claim_type: str = "independent"  # "independent" | "dependent"
    parent_claim: Optional[int] = None
    text: str
    elements: List[ClaimElement] = Field(default_factory=list)
    preamble: Optional[str] = None   # 청구항 전제부
    closing: Optional[str] = None    # 청구항 종결부
    split_method: str = "regex"      # "labeled" | "regex" | "fallback" | "llm"


class ElementMatch(BaseModel):
    label: str
    found: bool = False
    quote: str = ""
    chunk_id: str = ""
    judgment: str = "대응 없음"
    cited_invention_index: int = 0
    similarity_reason: str = ""


class ManualClaimRequest(BaseModel):
    claim_text: str
    claim_number: int = 1
    claim_type: str = "independent"
    parent_claim: Optional[int] = None


class BatchDependentRequest(BaseModel):
    claim_numbers: List[int]
    use_context: bool = True
    force: bool = False


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    report_md: str = ""
    web_search: bool = False


class Settings(BaseModel):
    model_config = {"extra": "ignore"}  # 이전 버전 설정의 추가 필드는 무시한다.

    engine: str = "claude"
    comparison_mode: Literal["per_doc", "hybrid"] = "per_doc"
    # 작업별 모델 선택(비어 있으면 엔진 기본 모델 사용)
    model_parser: str = ""    # 청구항 분석 및 보정
    model_compare: str = ""   # 구성요소 대비
    model_report: str = ""    # Phase 1 보고서 생성
    # RAG 검색 설정(BGE-M3 Hybrid Search)
    use_rag_retrieval: bool = True  # True: Dense+BM25 검색 / False: 전체 본문 사용
    rag_top_k: int = 20             # RAG 후보 문단 수
    use_reranker: bool = True
    reranker_top_k: int = 10
    dependent_candidate_doc_limit: int = 3  # Dependent claims: RAG-routed docs to compare before batch reporting.
    pdf_primary_parser: str = "pymupdf"
    pdf_fallback_parser: str = "opendataloader_pdf"
    vector_store: str = "qdrant_local"
    bm25_backend: str = "sqlite_fts5"
    metadata_store: str = "sqlite"
    # 내부 저장 경로
    rag_uploads_dir: str = "uploads"  # RAG 임베딩 캐시 기준 경로


class ModelListResponse(BaseModel):
    claude: List[str] = [
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ]
    gemini: List[str] = [
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
    ]
    agy: List[str] = [
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-3.1-pro-preview",
    ]
