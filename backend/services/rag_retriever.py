"""
Hybrid RAG Retriever — BGE-M3 Dense + SQLite FTS5 BM25 + RRF 결합

구성요소 텍스트를 쿼리로 삼아 인용발명 단락에서 관련 단락 top-K개를 선택한다.
LLM에게 인용발명 원문(全文) 대신 선택된 단락만 전달하여 입력 토큰을 줄인다.

캐시 의존:
- Dense 검색: cases/{job_id}/vector_db/qdrant 로컬 Qdrant 컬렉션
- 본문/metadata/reference DB 및 BM25: cases/{job_id}/reference.sqlite (SQLite FTS5)
- 단락 Dense 임베딩: npy 및 Qdrant 컬렉션 생성시용 빌드 캐시로 저장
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.models.schemas import ClaimElement, ExtractedDocument
from backend.services.reference_store import search_paragraphs_fts5

logger = logging.getLogger(__name__)

# BGE-M3 모델 ID (HuggingFace)
_BGE_MODEL_ID = "BAAI/bge-m3"
_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"

# 전역 모델 인스턴스(최초 1번만 로드)
_bge_model = None
_model_load_failed = False  # 모델 로드 실패 시 폴백 방지를 위한 플래그
_reranker_model = None
_reranker_load_failed = False
_reranker_inference_failed = False
_runtime_status = {
    "dense": "not_attempted",
    "qdrant": "not_attempted",
    "bm25": "not_attempted",
    "reranker": "not_attempted",
    "fallback_reason": "",
}

def _get_bge_model():
    """BGE-M3 모델을 싱글톤으로 로드한다. 실패 시 None 반환."""
    global _bge_model, _model_load_failed
    if _model_load_failed:
        return None
    if _bge_model is not None:
        return _bge_model
    try:
        from FlagEmbedding import BGEM3FlagModel  # type: ignore
        logger.info("BGE-M3 모델 로드 중(최초 실행 시 다운로드가 발생할 수 있음)...")
        _bge_model = BGEM3FlagModel(_BGE_MODEL_ID, use_fp16=False)
        _runtime_status["dense"] = "ready"
        logger.info("BGE-M3 모델 로드 완료")
        return _bge_model
    except Exception as e:
        logger.warning(f"BGE-M3 모델 로드 실패, dense RAG 비활성화: {e}")
        _model_load_failed = True
        _runtime_status["dense"] = "failed"
        _runtime_status["fallback_reason"] = f"BGE-M3 로드 실패: {e}"
        return None


def _get_reranker_model():
    """Load the local reranker once; failure keeps the existing RRF ordering."""
    global _reranker_model, _reranker_load_failed
    if _reranker_load_failed:
        return None
    if _reranker_model is not None:
        return _reranker_model
    try:
        from FlagEmbedding import FlagReranker  # type: ignore
        logger.info("BGE reranker 모델 로드 중(최초 실행 시 다운로드가 발생할 수 있음)...")
        _reranker_model = FlagReranker(_RERANKER_MODEL_ID, use_fp16=False)
        tokenizer = getattr(_reranker_model, "tokenizer", None)
        if tokenizer is not None and not hasattr(tokenizer, "prepare_for_model"):
            raise RuntimeError(
                "?? ??? transformers/FlagEmbedding ????? reranker tokenizer.prepare_for_model? ???? ????"
            )
        _runtime_status["reranker"] = "ready"
        logger.info("BGE reranker 모델 로드 완료")
        return _reranker_model
    except Exception as exc:
        logger.warning("Reranker 모델 로드 실패, RRF 순서를 사용합니다: %s", exc)
        _reranker_load_failed = True
        _runtime_status["reranker"] = "failed"
        _runtime_status["fallback_reason"] = f"Reranker 로드 실패: {exc}"
        return None


def get_rag_runtime_status() -> Dict[str, object]:
    """Return dependency availability and the latest observed RAG runtime state."""
    return {
        **_runtime_status,
        "flag_embedding_installed": importlib.util.find_spec("FlagEmbedding") is not None,
        "qdrant_client_installed": importlib.util.find_spec("qdrant_client") is not None,
        "dense_model_id": _BGE_MODEL_ID,
        "reranker_model_id": _RERANKER_MODEL_ID,
    }


def _rerank_candidates(
    elements: List[ClaimElement],
    chunks: List[_SearchChunk],
    candidate_idxs: List[int],
) -> Optional[List[Tuple[float, int]]]:
    global _reranker_inference_failed, _reranker_model, _reranker_load_failed
    if _reranker_inference_failed:
        return None
    model = _get_reranker_model()
    queries = [(elem.text or "").strip() for elem in elements if (elem.text or "").strip()]
    if model is None or not queries or not candidate_idxs:
        return None

    pairs = [[query, chunks[idx].search_text] for idx in candidate_idxs for query in queries]
    try:
        raw_scores = model.compute_score(pairs, normalize=True)
        if isinstance(raw_scores, (int, float)):
            raw_scores = [float(raw_scores)]
        scores = [float(score) for score in raw_scores]
        width = len(queries)
        ranked = []
        for pos, idx in enumerate(candidate_idxs):
            candidate_scores = scores[pos * width:(pos + 1) * width]
            if candidate_scores:
                ranked.append((max(candidate_scores), idx))
        _runtime_status["reranker"] = "active"
        return sorted(ranked, key=lambda item: (-item[0], item[1]))
    except Exception as exc:
        logger.warning("Reranker 추론 실패, RRF 순서를 사용합니다: %s", exc)
        _reranker_inference_failed = True
        _reranker_load_failed = True
        _reranker_model = None
        _runtime_status["reranker"] = "failed"
        _runtime_status["fallback_reason"] = f"Reranker 추론 실패: {exc}"
        return None


# ---------------------------------------------------------------------------
# 단락/그룹 검색 청크 데이터클래스
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 1_200


@dataclass
class _SearchChunk:
    chunk_id: str
    search_text: str
    original_text: str
    chunk_type: str = "paragraph"
    paragraph_no: str = ""
    paragraph_range: Tuple[str, ...] = ()
    page_no: Optional[int] = None
    section: str = ""
    publication_no: str = ""
    doc_id: str = ""
    score: float = 0.0


def _doc_chunks(doc: ExtractedDocument) -> List[_SearchChunk]:
    """문서에서 검색용 청크를 추출한다.

    사전 metadata가 있는 paragraph/group chunk를 사용한다. 인용발명 PDF 대량
    처리시 paragraph_chunks/group_chunks 생성 단계에서 제외되어 검색 index에
    들어가지 않는 경우, 구버전 캐시를 기존 paragraphs/pages로 대체한다.
    """
    chunks: List[_SearchChunk] = []

    for chunk in getattr(doc, "paragraph_chunks", []) or []:
        text = (chunk.normalized_text or chunk.original_text or "").strip()
        original = (chunk.original_text or "").strip()
        if text and original:
            chunks.append(_SearchChunk(
                chunk_id=chunk.chunk_id or f"[{chunk.paragraph_no}]",
                search_text=text,
                original_text=original,
                chunk_type="paragraph",
                paragraph_no=chunk.paragraph_no or "",
                paragraph_range=tuple(chunk.paragraph_range or ([chunk.paragraph_no] if chunk.paragraph_no else [])),
                page_no=chunk.page_no,
                section=chunk.section or "",
                publication_no=chunk.publication_no or getattr(doc, "publication_no", ""),
                doc_id=chunk.doc_id or getattr(doc, "doc_id", ""),
            ))

    for chunk in getattr(doc, "group_chunks", []) or []:
        text = (chunk.normalized_text or chunk.original_text or "").strip()
        original = (chunk.original_text or "").strip()
        if text and original:
            chunks.append(_SearchChunk(
                chunk_id=chunk.chunk_id,
                search_text=text,
                original_text=original,
                chunk_type="group",
                paragraph_range=tuple(chunk.paragraph_range or []),
                section=chunk.section or "",
                publication_no=chunk.publication_no or getattr(doc, "publication_no", ""),
                doc_id=chunk.doc_id or getattr(doc, "doc_id", ""),
            ))

    if chunks:
        return chunks

    if doc.paragraphs:
        return [
            _SearchChunk(
                chunk_id=para_id,
                search_text=text.strip(),
                original_text=f"{para_id} {text.strip()}".strip(),
                chunk_type="paragraph",
                paragraph_no=para_id.strip("[]"),
                paragraph_range=(para_id.strip("[]"),),
                publication_no=getattr(doc, "publication_no", ""),
                doc_id=getattr(doc, "doc_id", ""),
            )
            for para_id, text in doc.paragraphs.items()
            if text and text.strip()
        ]
    if doc.pages:
        chunks = []
        for page_num, page_text in doc.pages.items():
            text = (page_text or "").strip()
            if not text:
                continue
            for idx in range(0, len(text), _CHUNK_SIZE):
                chunk = text[idx:idx + _CHUNK_SIZE].strip()
                if chunk:
                    chunks.append(_SearchChunk(
                        chunk_id=f"[P{page_num}-{idx // _CHUNK_SIZE + 1}]",
                        search_text=chunk,
                        original_text=chunk,
                        page_no=int(page_num) if str(page_num).isdigit() else None,
                        publication_no=getattr(doc, "publication_no", ""),
                        doc_id=getattr(doc, "doc_id", ""),
                    ))
        return chunks
    raw = doc.raw_text or ""
    return [
        _SearchChunk(
            chunk_id=f"[T{idx // _CHUNK_SIZE + 1}]",
            search_text=raw[idx:idx + _CHUNK_SIZE].strip(),
            original_text=raw[idx:idx + _CHUNK_SIZE].strip(),
            publication_no=getattr(doc, "publication_no", ""),
            doc_id=getattr(doc, "doc_id", ""),
        )
        for idx in range(0, len(raw), _CHUNK_SIZE)
        if raw[idx:idx + _CHUNK_SIZE].strip()
    ]


# ---------------------------------------------------------------------------
# 임베딩 캐시 경로 계산
# ---------------------------------------------------------------------------

def _embedding_cache_path(sha256: str, uploads_dir: Path) -> Path:
    cache_dir = uploads_dir / "_doc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{sha256}_embeddings.npy"


def _sha256_of_doc(doc: ExtractedDocument) -> str:
    """문서 raw_text의 SHA-256. pdf_path가 있으면 파일 기반, 없으면 텍스트 기반."""
    if doc.pdf_path and Path(doc.pdf_path).exists():
        try:
            h = hashlib.sha256()
            with open(doc.pdf_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            pass
    return hashlib.sha256((doc.raw_text or "").encode()).hexdigest()


# ---------------------------------------------------------------------------
# Dense 임베딩 생성 및 캐시
# ---------------------------------------------------------------------------

def _build_dense_embeddings(
    texts: List[str],
    model,
) -> np.ndarray:
    """BGE-M3로 dense 임베딩 배열을 반환한다. shape: (N, dim)"""
    if not texts:
        return np.zeros((0, 1024), dtype=np.float32)
    result = model.encode(
        texts,
        batch_size=16,
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    vecs = result["dense_vecs"]
    # L2 정규화(코사인 유사도와 동일하게 계산)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    return (vecs / norms).astype(np.float32)


def get_or_build_dense_index(
    doc: ExtractedDocument,
    chunks: List[_SearchChunk],
    uploads_dir: Path,
) -> Optional[np.ndarray]:
    """
    Dense 임베딩을 캐시에서 로드하거나 새로 생성한다.

    반환: shape (N, dim) float32 배열. 모델 없으면 None.
    """
    model = _get_bge_model()
    if model is None:
        return None

    sha256 = _sha256_of_doc(doc)
    cache_path = _embedding_cache_path(sha256, uploads_dir)

    if cache_path.exists():
        try:
            arr = np.load(str(cache_path))
            if arr.shape[0] == len(chunks):
                logger.info(f"RAG: 임베딩 캐시 로드 {doc.filename} ({arr.shape[0]}개 청크)")
                return arr
        except Exception as e:
            logger.warning(f"임베딩 캐시 손상, 재생성: {e}")

    texts = [chunk.search_text for chunk in chunks]
    logger.info(f"RAG: BGE-M3 임베딩 생성 {doc.filename} ({len(texts)}개 청크)...")
    embeddings = _build_dense_embeddings(texts, model)
    np.save(str(cache_path), embeddings)
    logger.info(f"RAG: 임베딩 저장 완료: {cache_path.name}")
    return embeddings


# ---------------------------------------------------------------------------
# SQLite FTS5 BM25 / Qdrant 로컬 dense 검색
# ---------------------------------------------------------------------------

def _case_dir_for_doc(doc: ExtractedDocument, uploads_dir: Path) -> Optional[Path]:
    """uploads/{job_id}/pdfs/foo.pdf -> cases/{job_id}."""
    try:
        pdf_path = Path(doc.pdf_path)
        if pdf_path.parent.name.lower() == "pdfs":
            job_id = pdf_path.parent.parent.name
            return uploads_dir.parent / "cases" / job_id
    except Exception:
        return None
    return None


def _qdrant_collection_name(doc: ExtractedDocument) -> str:
    sha = _sha256_of_doc(doc)[:12]
    base = re.sub(r"[^A-Za-z0-9_]+", "_", doc.doc_id or f"D{doc.doc_index + 1}")
    return f"aria_{base}_{sha}"


def _qdrant_collection_exists(client, collection_name: str) -> bool:
    try:
        return bool(client.collection_exists(collection_name))
    except Exception:
        try:
            client.get_collection(collection_name)
            return True
        except Exception:
            return False


def build_qdrant_index(
    doc: ExtractedDocument,
    chunks: List[_SearchChunk],
    uploads_dir: Path,
):
    """로컬 Qdrant 컬렉션을 준비하고 (client, collection, model)을 반환한다."""
    if not chunks:
        return None
    model = _get_bge_model()
    if model is None:
        return None
    case_dir = _case_dir_for_doc(doc, uploads_dir)
    if case_dir is None:
        return None

    try:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.models import Distance, PointStruct, VectorParams  # type: ignore
    except ImportError:
        logger.warning("qdrant-client 미설치: dense 검색 비활성화")
        _runtime_status["qdrant"] = "missing"
        return None

    embeddings = get_or_build_dense_index(doc, chunks, uploads_dir)
    if embeddings is None or embeddings.shape[0] == 0:
        return None

    qdrant_dir = case_dir / "vector_db" / "qdrant"
    case_id = case_dir.name
    qdrant_dir.mkdir(parents=True, exist_ok=True)
    collection = _qdrant_collection_name(doc)
    try:
        client = QdrantClient(path=str(qdrant_dir))
        expected_count = len(chunks)
        if _qdrant_collection_exists(client, collection):
            try:
                count = client.count(collection_name=collection, exact=True).count
                if count == expected_count:
                    _runtime_status["qdrant"] = "active"
                    return client, collection, model
            except Exception:
                pass
            client.delete_collection(collection_name=collection)

        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=int(embeddings.shape[1]), distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=idx,
                vector=embeddings[idx].tolist(),
                payload={
                    "case_id": case_id,
                    "doc_id": chunks[idx].doc_id,
                    "publication_no": chunks[idx].publication_no,
                    "chunk_id": chunks[idx].chunk_id,
                    "chunk_type": chunks[idx].chunk_type,
                    "section": chunks[idx].section,
                    "paragraph_no": chunks[idx].paragraph_no,
                    "paragraph_range": list(chunks[idx].paragraph_range or []),
                    "page_no": chunks[idx].page_no,
                    "group_id": chunks[idx].chunk_id if chunks[idx].chunk_type == "group" else "",
                },
            )
            for idx in range(len(chunks))
        ]
        client.upsert(collection_name=collection, points=points)
        logger.info(f"Qdrant: {doc.filename} 컬렉션 생성 완료 ({len(points)} chunks)")
        _runtime_status["qdrant"] = "active"
        return client, collection, model
    except Exception as e:
        logger.warning(f"Qdrant 로컬 인덱스 준비 실패, dense 검색 비활성화: {e}")
        _runtime_status["qdrant"] = "failed"
        _runtime_status["fallback_reason"] = f"Qdrant 준비 실패: {e}"
        return None


def _case_filter(case_id: str):
    from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

    return Filter(
        must=[
            FieldCondition(
                key="case_id",
                match=MatchValue(value=case_id),
            )
        ]
    )


def _qdrant_search(query: str, qdrant_handle, case_id: str, limit: int) -> List[int]:
    if qdrant_handle is None:
        return []
    client, collection, model = qdrant_handle
    try:
        q_emb = _build_dense_embeddings([query], model)[0].tolist()
        q_filter = _case_filter(case_id)
        try:
            result = client.query_points(
                collection_name=collection,
                query=q_emb,
                query_filter=q_filter,
                limit=limit,
                with_payload=True,
            )
            points = result.points
        except Exception:
            points = client.search(
                collection_name=collection,
                query_vector=q_emb,
                query_filter=q_filter,
                limit=limit,
            )
        return [int(point.id) for point in points]
    except Exception as e:
        logger.debug(f"Qdrant 검색 오류: {e}")
        return []


def _close_qdrant(qdrant_handle) -> None:
    if qdrant_handle is None:
        return
    try:
        qdrant_handle[0].close()
    except Exception:
        pass


def _paragraph_index(chunks: List[_SearchChunk]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for idx, chunk in enumerate(chunks):
        paras = chunk.paragraph_range or ((chunk.paragraph_no,) if chunk.paragraph_no else ())
        if chunk.chunk_type == "paragraph" and not paras:
            paras = (chunk.chunk_id.strip("[]"),)
        for para in paras:
            key = str(para).strip("[]")
            if key:
                mapping.setdefault(key, []).append(idx)
    return mapping


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense_ranks: List[int],       # 단락 인덱스 순서 (dense top-K)
    bm25_ranks: List[int],        # 단락 인덱스 순서 (bm25 top-K)
    k: int = 60,
) -> List[Tuple[float, int]]:
    """RRF 점수를 계산하여 (score, para_idx) 목록을 내림차순으로 반환한다."""
    scores: Dict[int, float] = {}
    for rank, idx in enumerate(dense_ranks):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_ranks):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(((score, idx) for idx, score in scores.items()), key=lambda x: -x[0])


# ---------------------------------------------------------------------------
# 단일 쿼리 파트 검색
# ---------------------------------------------------------------------------

def _search_single_query(
    query: str,
    qdrant_handle,
    case_dir: Optional[Path],
    doc_id: str,
    chunks: List[_SearchChunk],
    top_k_per_source: int = 10,
) -> List[int]:
    """
    하나의 쿼리 문자열에 대해 Qdrant Dense + SQLite FTS5 BM25 검색을 수행하고,
    RRF로 결합된 단락 인덱스 목록(중요도 내림차순)을 반환한다.
    """
    n = len(chunks)
    if n == 0:
        return []

    dense_ranks: List[int] = []
    bm25_ranks: List[int] = []

    case_id = case_dir.name if case_dir is not None else ""
    dense_ranks = _qdrant_search(query, qdrant_handle, case_id, min(top_k_per_source, n))

    if case_dir is not None:
        para_to_indices = _paragraph_index(chunks)
        rows = search_paragraphs_fts5(case_dir, doc_id, query, limit=top_k_per_source) or []
        _runtime_status["bm25"] = "active" if rows else "ready"
        seen: set[int] = set()
        for row in rows:
            para = str(row.get("paragraph_no", "")).strip("[]")
            for idx in para_to_indices.get(para, []):
                if idx not in seen:
                    bm25_ranks.append(idx)
                    seen.add(idx)

    # 둘 다 없으면 빈 결과
    if not dense_ranks and not bm25_ranks:
        return []

    # RRF 결합
    fused = _rrf_fuse(dense_ranks, bm25_ranks)
    return [idx for _, idx in fused]


# ---------------------------------------------------------------------------
# 구성요소별 다중 쿼리 통합 단락 선택
# ---------------------------------------------------------------------------

def retrieve_for_elements(
    elements: List[ClaimElement],
    doc: ExtractedDocument,
    uploads_dir: Path,
    top_k: int = 20,
    top_k_per_element: int = 5,
    use_reranker: bool = False,
    reranker_top_k: Optional[int] = None,
) -> Optional[Dict[str, str]]:
    """
    구성요소 리스트를 쿼리로 삼아 인용발명 단락 중 관련성 높은 단락을 선택한다.

    동작:
    1. 문서 단락 당 Dense 임베딩 캐시) + BM25 인덱스 빌드
    2. 각 구성요소 텍스트를 쿼리로 Dense+BM25 검색. 구성요소별 top_k_per_element가 선택
    3. 전체 구성요소의 결과를 합산 후 RRF 스코어링하여 최종 top_k개 단락 선택
    4. 선택된 단락을 {chunk_id: text} dict로 반환 (순서: 원문 순)

    실패(모델 없음, 인덱스 없음) 시 None 반환 후 호출부에서 기존 방식 대체.
    """
    hits = retrieve_with_metadata(
        elements=elements,
        doc=doc,
        uploads_dir=uploads_dir,
        top_k=top_k,
        top_k_per_element=top_k_per_element,
        use_reranker=use_reranker,
        reranker_top_k=reranker_top_k,
    )
    if hits is None:
        return None
    return {hit["paragraph_id"]: hit["original_text"] for hit in hits}


def retrieve_with_metadata(
    elements: List[ClaimElement],
    doc: ExtractedDocument,
    uploads_dir: Path,
    top_k: int = 20,
    top_k_per_element: int = 5,
    use_reranker: bool = False,
    reranker_top_k: Optional[int] = None,
) -> Optional[List[Dict]]:
    """
    구성요소별 RAG 검색 결과를 metadata와 함께 반환한다.

    검색은 paragraph chunk와 group chunk를 함께 대상으로 되며, 최종 반환은
    단락 원문 DB(paragraph_records/paragraph_chunks)에서 조회한 paragraph
    단위 hit만 제공한다. group chunk는 문맥 검색과 문구 정정 인자에만 관여한다.
    """
    chunks = _doc_chunks(doc)
    if not chunks:
        return None

    model = _get_bge_model()
    case_dir = _case_dir_for_doc(doc, uploads_dir)

    qdrant_handle = build_qdrant_index(doc, chunks, uploads_dir) if model is not None else None

    if qdrant_handle is None and case_dir is None:
        logger.warning("RAG: Qdrant와 SQLite FTS5를 모두 사용할 수 없어 전체 본문 방식으로 대체")
        _runtime_status["fallback_reason"] = "Qdrant와 SQLite FTS5를 모두 사용할 수 없음"
        return None

    # 구성요소별 검색 결과 글로벌 점수 적산
    global_scores: Dict[int, float] = {}
    k_rrf = 60

    for elem_idx, elem in enumerate(elements or []):
        query_text = elem.text.strip()
        if not query_text:
            continue

        ranked = _search_single_query(
            query_text,
            qdrant_handle,
            case_dir,
            doc.doc_id or f"D{doc.doc_index + 1}",
            chunks,
            top_k_per_source=top_k_per_element * 2,
        )
        # 중요도 높은 구성요소(importance 점수로 계산)에 한한 boost
        try:
            imp_weight = float(elem.importance) / 3.0
        except (ValueError, TypeError):
            imp_weight = 1.0

        for rank, para_idx in enumerate(ranked[:top_k_per_element * 2]):
            rrf_score = imp_weight / (k_rrf + rank + 1)
            global_scores[para_idx] = global_scores.get(para_idx, 0.0) + rrf_score

    if not global_scores:
        _close_qdrant(qdrant_handle)
        return None

    # 상위 top_k개 선택 후 본문 순서 복원
    sorted_idxs = sorted(global_scores.keys(), key=lambda i: -global_scores[i])
    output_k = max(1, min(int(reranker_top_k or top_k), top_k)) if use_reranker else top_k
    candidate_idxs = sorted_idxs[:top_k]
    reranked = _rerank_candidates(elements, chunks, candidate_idxs) if use_reranker else None
    if reranked:
        selected_idxs = [idx for _score, idx in reranked[:output_k]]
    else:
        selected_idxs = candidate_idxs[:output_k]

    selected_para_nos: Dict[str, float] = {}
    for idx in selected_idxs:
        if not (0 <= idx < len(chunks)):
            continue
        chunk = chunks[idx]
        score = global_scores.get(idx, 0.0)
        paras = chunk.paragraph_range or ((chunk.paragraph_no,) if chunk.paragraph_no else ())
        if chunk.chunk_type == "paragraph" and not paras:
            paras = (chunk.chunk_id.strip("[]"),)
        for para in paras:
            key = str(para).strip("[]")
            if key:
                selected_para_nos[key] = max(selected_para_nos.get(key, 0.0), score)

    paragraph_lookup: Dict[str, _SearchChunk] = {}
    for chunk in chunks:
        if chunk.chunk_type != "paragraph":
            continue
        key = (chunk.paragraph_no or chunk.chunk_id).strip("[]")
        paragraph_lookup[key] = chunk

    ordered_para_nos = sorted(
        selected_para_nos.keys(),
        key=lambda p: (
            -selected_para_nos[p],
            int(p) if p.isdigit() else 999999,
            p,
        ),
    )[:output_k]

    hits: List[Dict] = []
    ordered_records = [paragraph_lookup[p] for p in ordered_para_nos if p in paragraph_lookup]
    for pos, chunk in enumerate(ordered_records):
        para = (chunk.paragraph_no or chunk.chunk_id).strip("[]")
        before = ordered_records[pos - 1].original_text if pos > 0 else ""
        after = ordered_records[pos + 1].original_text if pos + 1 < len(ordered_records) else ""
        hits.append({
            "doc_id": chunk.doc_id or getattr(doc, "doc_id", f"D{doc.doc_index + 1}"),
            "publication_no": chunk.publication_no or getattr(doc, "publication_no", ""),
            "group_id": "",
            "paragraph_id": f"[{para}]" if para and not para.startswith("[") else para,
            "paragraph_no": para,
            "page_no": chunk.page_no,
            "section": chunk.section,
            "score": selected_para_nos.get(para, 0.0),
            "original_text": chunk.original_text,
            "before": before,
            "after": after,
        })

    logger.info(
        f"RAG: {doc.filename} 전체 {len(chunks)}개 청크 중 {len(hits)}개 문단 선택 "
        f"(구성요소 {len(elements)}개 쿼리, reranker={'사용' if reranked else '미사용'})"
    )
    _close_qdrant(qdrant_handle)
    return hits


# ---------------------------------------------------------------------------
# 선택 단락 → LLM 입력 텍스트 포맷
# ---------------------------------------------------------------------------

def format_rag_doc_text(selected: Dict[str, str]) -> str:
    """선택된 단락 dict를 `[XXXX] 텍스트` 형식의 문자열로 직렬화한다."""
    lines = []
    for chunk_id, text in selected.items():
        lines.append(f"{chunk_id} {text}")
    return "\n".join(lines)
