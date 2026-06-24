"""
?몄슜 異붿텧 ?먯씠?꾪듃 ???몄슜諛쒕챸 ?꾨Ц(?ⓩ뻼)??Claude??吏곸젒 ?꾨떖?섏뿬 援ъ꽦?붿냼 ?鍮?

[理쒖쟻??援ъ“]
- 以鍮??④퀎?먯꽌 紐⑤뱺 泥?뎄??쓣 ??踰덉뿉 ?鍮???comparisons_{doc_idx}.json 罹먯떆
- 蹂닿퀬???앹꽦 ?쒖뿉??罹먯떆?먯꽌 濡쒕뱶留???(?몄슜諛쒕챸 ?꾨Ц ?ъ쟾???놁쓬)
- ?몄슜諛쒕챸 1媛쒕떦 LLM 1???몄텧 (泥?뎄??N媛?紐⑤몢 泥섎━)
"""
from __future__ import annotations
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from backend.models.schemas import ClaimElement, ElementMatch, ExtractedDocument, ParsedClaim, Settings
from backend.services.ai_engine import call_ai
from backend.services.prompt_loader import load_prompt, render_prompt
from backend.services.rag_retriever import format_rag_doc_text, retrieve_with_metadata

logger = logging.getLogger(__name__)


class CompareFailed(Exception):
    """구성대비 LLM 호출 또는 응답 파싱 실패를 나타낸다.

    실제로 인용발명에 대응 내용이 없어 나온 정상 JSON 결과와
    CLI 호출/파싱 실패로 인한 빈 결과를 구분하기 위한 예외다.
    이 예외가 발생하면 빈 비교 결과를 캐시하지 않고 호출부에 오류를 전달한다.
    """


# ?붿쭊蹂??鍮??낅젰 ?덉궛 (relevant, hard, hybrid_total, hybrid_min).
# Claude CLI??湲?stdin?먯꽌 留ㅼ슦 ?먮젮??蹂댁닔?곸쑝濡??덈떒?쒕떎.
# Gemini??100留??좏겙 而⑦뀓?ㅽ듃???몄슜諛쒕챸 ?꾨Ц??洹몃?濡??ｌ뼱 ?덈떒???뚰뵾?쒕떎.
# (?쒓뎅??泥?뎄???ㅼ썙?쒕뒗 ?곸뼱 ?몄슜諛쒕챸 蹂몃Ц怨?留ㅼ묶?섏? ?딆븘 ?ㅼ썙???좊퀎??
#  ?ъ떎??臾대젰?붾릺怨??욌?遺??덈떒?쇰줈 ?대갚?쒕떎. ?덉궛???ㅼ썙 ?덈떒 ?먯껜瑜??쇳븯??
#  履쎌씠 援ъ꽦?鍮??뺥빀?깆뿉 ?좊━?섎떎.)
_ENGINE_BUDGETS = {
    "gemini": (300_000, 400_000, 300_000, 30_000),
    "agy": (300_000, 400_000, 300_000, 30_000),
    "claude": (45_000, 60_000, 55_000, 5_000),
}
_DEFAULT_BUDGET = (45_000, 60_000, 55_000, 5_000)
_CHUNK_SIZE = 1_200
_CACHE_META_KEY = "_meta"
_CACHE_SCHEMA_VERSION = 3
_DEFAULT_DEPENDENT_CANDIDATE_DOC_LIMIT = 3
_JUDGMENT_RANK = {
    "동일": 5,
    "실질적 동일": 4,
    "일부 차이": 3,
    "일부 유사": 2,
    "차이": 1,
    "대응 없음": 0,
}


def _budgets(engine: str) -> tuple[int, int, int, int]:
    return _ENGINE_BUDGETS.get((engine or "").lower(), _DEFAULT_BUDGET)


def _full_doc_text(doc: ExtractedDocument) -> str:
    chunks = _doc_chunks(doc)
    return "\n".join(f"{cid} {text}" for cid, text in chunks)


def _rag_enabled(settings: Optional[Settings]) -> bool:
    return bool(settings is not None and getattr(settings, "use_rag_retrieval", False))


def _comparison_mode(value: object) -> str:
    return "hybrid" if str(value or "").strip().lower() == "hybrid" else "per_doc"


def _rag_hits_for_doc(
    doc: ExtractedDocument,
    elements: List[ClaimElement],
    settings: Settings,
    top_k: Optional[int] = None,
) -> Optional[List[Dict]]:
    try:
        resolved_top_k = int(top_k or getattr(settings, "rag_top_k", 20) or 20)
    except (TypeError, ValueError):
        resolved_top_k = 20
    resolved_top_k = max(1, min(resolved_top_k, 50))
    top_k_per_element = max(3, min(8, resolved_top_k // max(1, len(elements)) + 2))
    return retrieve_with_metadata(
        elements,
        doc,
        Path(getattr(settings, "rag_uploads_dir", "uploads") or "uploads"),
        top_k=resolved_top_k,
        top_k_per_element=top_k_per_element,
        use_reranker=bool(getattr(settings, "use_reranker", False)),
        reranker_top_k=int(getattr(settings, "reranker_top_k", 10) or 10),
    )


def select_candidate_doc_indices_for_elements(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
    max_docs: Optional[int] = None,
) -> List[int]:
    """Pick a small set of prior documents for dependent-claim comparison.

    The expensive LLM comparison should not run as claim_count * doc_count when
    dependent claims only add a few limitations. This function uses the same
    local RAG signal as the comparison prompt builder to route each dependent
    claim to the most relevant documents first. If RAG is disabled or cannot
    produce any candidates, it falls back to all documents so accuracy wins over
    speed in uncertain cases.
    """
    if not prior_docs:
        return []
    if not _rag_enabled(settings) or len(prior_docs) <= 3:
        return list(range(len(prior_docs)))

    try:
        configured_limit = int(
            max_docs
            or getattr(settings, "dependent_candidate_doc_limit", 0)
            or _DEFAULT_DEPENDENT_CANDIDATE_DOC_LIMIT
        )
    except (TypeError, ValueError):
        configured_limit = _DEFAULT_DEPENDENT_CANDIDATE_DOC_LIMIT
    limit = max(1, min(configured_limit, len(prior_docs)))

    scored_docs: list[tuple[float, int, int]] = []
    for doc_idx, doc in enumerate(prior_docs):
        hits = _rag_hits_for_doc(doc, elements, settings, top_k=getattr(settings, "rag_top_k", 20))
        if not hits:
            continue
        score = sum(float(hit.get("score", 0.0) or 0.0) for hit in hits)
        scored_docs.append((score, len(hits), doc_idx))

    if not scored_docs:
        logger.info("Dependent candidate routing: no RAG hits; comparing all documents")
        return list(range(len(prior_docs)))

    scored_docs.sort(key=lambda item: (-item[0], -item[1], item[2]))
    selected = [doc_idx for _score, _count, doc_idx in scored_docs[:limit]]
    logger.info(
        "Dependent candidate routing: selected docs %s from scores %s",
        selected,
        [(doc_idx, round(score, 4), count) for score, count, doc_idx in scored_docs[:limit]],
    )
    return selected

# 諛쒖톸(quote) 湲몄씠 ?덉쟾?μ튂 ??LLM???듭떖 ?닿뎄 諛쒖톸 吏?쒕? 吏?ㅼ? 紐삵븯怨??⑤씫??
# ?듭㎏濡?蹂듭궗??寃쎌슦?먮쭔 ?묐룞?쒕떎. ??湲몄씠瑜??섍퀬 ?앸왂遺(...)媛 ?놁쑝硫?癒몃━/瑗щ━
# ?닿뎄留??먮Ц 洹몃?濡??④린怨?媛?대뜲瑜?' ... '濡?異뺤빟?쒕떎.
# LLM??吏곸젒 ?좊땲肄붾뱶 留먯쨪?꾪몴濡?異뺤빟??諛쒖톸??ASCII '...'濡??뺢퇋?뷀븳??
_QUOTE_MAX_CHARS = 350
_QUOTE_HEAD_CHARS = 190
_QUOTE_TAIL_CHARS = 140
_ELLIPSIS = " ... "


def _shorten_quote(quote: str) -> str:
    """怨쇰룄?섍쾶 湲?諛쒖톸瑜?癒몃━/瑗щ━留??④린怨?媛?대뜲瑜?' ... '濡?異뺤빟?쒕떎."""
    q = (quote or "").strip().replace("…", "...")
    if len(q) <= _QUOTE_MAX_CHARS:
        return q
    head = q[:_QUOTE_HEAD_CHARS].rsplit(" ", 1)[0].rstrip() or q[:_QUOTE_HEAD_CHARS]
    tail = q[-_QUOTE_TAIL_CHARS:].split(" ", 1)[-1].lstrip() or q[-_QUOTE_TAIL_CHARS:]
    return f"{head}{_ELLIPSIS}{tail}"


def normalize_label(label: str) -> str:
    """援ъ꽦?붿냼 ?쇰꺼???쒖????臾몄옄 + ?좏깮??-?レ옄 ?쒕툕?쇰꺼)?쇰줈 ?뺢퇋?뷀븳??

    泥?뎄???뚯떛쨌LLM ?鍮꽷룹틦?쒓? ?쇰꺼??'A', '(A) ?④퀎', '(a)', 'A-1' ???쒓컖媛?
    ?뺤떇?쇰줈 ?대낫?대룄 ?숈씪 援ъ꽦?붿냼濡?留ㅼ묶?섍쾶 ?쒕떎. ?뺤떇 遺덉씪移???罹먯떆 議고쉶媛
    鍮?寃곌낵('????놁쓬')濡??⑥뼱吏??臾몄젣瑜?留됰뒗 ?듭떖 諛⑹뼱?좎씠??"""
    m = re.search(r'([A-Ja-j])\s*(?:-\s*(\d+))?', label or "")
    if not m:
        return (label or "").strip().upper()
    base = m.group(1).upper()
    return f"{base}-{m.group(2)}" if m.group(2) else base


def _build_doc_text(
    doc: ExtractedDocument,
    elements: Optional[List[ClaimElement]] = None,
    max_chars: Optional[int] = None,
    engine: str = "",
    settings: Optional[Settings] = None,
) -> str:
    """
    ?몄슜諛쒕챸 ?띿뒪?몃? LLM ?낅젰?⑹쑝濡?理쒖쟻?뷀븯??諛섑솚.

    ?곗꽑?쒖쐞:
    1. doc.paragraphs (?⑤씫踰덊샇[XXXX] 援ъ“泥? ???ㅻ뜑/?꾨㈃?ㅻ챸 ?쒖쇅, chunk_id 李몄“ 媛??
    2. doc.raw_text ?덉궘 ???⑤씫 ?뺣낫 ?놁쓣 ???대갚

    max_chars: ?몃??먯꽌 ?덉궛??吏?뺥븷 ???ъ슜. 誘몄??????대? ?곸닔 ?ъ슜.
    """
    chunks = _doc_chunks(doc)
    if not chunks:
        return ""

    if settings is not None:
        engine = settings.engine
    relevant_default, hard_default, _, _ = _budgets(engine)
    hard_limit = min(max_chars, hard_default) if max_chars else hard_default
    relevant_limit = min(max_chars, relevant_default) if max_chars else relevant_default

    full_text = "\n".join(f"{cid} {text}" for cid, text in chunks)
    if not elements or len(full_text) <= hard_limit:
        return full_text[:hard_limit]

    if _rag_enabled(settings) and elements:
        hits = _rag_hits_for_doc(doc, elements, settings)
        if hits:
            selected = {hit["paragraph_id"]: hit["original_text"] for hit in hits}
            rag_text = format_rag_doc_text(selected)
            logger.info(
                f"{doc.filename}: full text too long ({len(full_text)} chars); "
                f"RAG selected {len(selected)} paragraphs ({len(rag_text)} chars)"
            )
            return rag_text[:hard_limit]
        logger.warning(f"{doc.filename}: RAG returned no hits; using keyword context fallback")

    keywords = _claim_keywords(elements)
    if not keywords:
        return full_text[:hard_limit]

    scored = []
    for order, (chunk_id, text) in enumerate(chunks):
        lowered = text.lower()
        score = sum(1 for kw in keywords if kw in lowered)
        if score:
            scored.append((score, order, chunk_id, text))

    if not scored:
        logger.info(f"{doc.filename}: no keyword hits, using first {relevant_limit} chars")
        return full_text[:relevant_limit]

    selected_orders = {0}
    total = len(chunks[0][0]) + len(chunks[0][1]) + 2
    # break ???continue: ?덉궛 珥덇낵 ?⑤씫??嫄대꼫?곕릺 ?댄썑 ?⑤씫??怨꾩냽 寃?ы븳??
    # ?먯닔 ?숇쪧 ?????⑤씫??癒쇱? ?ㅻ뒗 ?뺣젹 ?뚮Ц?? break瑜??곕㈃ ?덉궛??珥덇린 ?⑤씫?쇰줈
    # ?뚯쭊???룸?遺꾩쓽 ?듭떖 ?ㅼ떆???⑤씫???듭㎏濡??꾨씫?섎뒗 臾몄젣媛 ?덉뿀??
    for score, order, _chunk_id, text in sorted(scored, key=lambda x: (-x[0], x[1])):
        item_len = len(text) + 20
        if total + item_len > relevant_limit:
            continue
        selected_orders.add(order)
        total += item_len

    selected = [
        f"{chunk_id} {text}"
        for order, (chunk_id, text) in enumerate(chunks)
        if order in selected_orders
    ]
    result = "\n".join(selected)
    logger.info(
        f"{doc.filename}: reduced LLM context {len(full_text)} -> {len(result)} chars "
        f"({len(selected)}/{len(chunks)} chunks)"
    )
    return result


def _doc_chunks(doc: ExtractedDocument) -> List[tuple[str, str]]:
    if doc.paragraphs:
        return [
            (para_id, text.strip())
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
                    chunks.append((f"[P{page_num}-{idx // _CHUNK_SIZE + 1}]", chunk))
        return chunks

    raw = doc.raw_text or ""
    return [
        (f"[T{idx // _CHUNK_SIZE + 1}]", raw[idx:idx + _CHUNK_SIZE].strip())
        for idx in range(0, len(raw), _CHUNK_SIZE)
        if raw[idx:idx + _CHUNK_SIZE].strip()
    ]


def _build_hybrid_docs_block(
    prior_docs: List[ExtractedDocument],
    elements: List[ClaimElement],
    engine: str = "",
    settings: Optional[Settings] = None,
) -> str:
    """Build one compact, chat-like comparison context from all prior documents."""
    if not prior_docs:
        return ""

    if settings is not None:
        engine = settings.engine
    _, _, hybrid_total, hybrid_min = _budgets(engine)

    full_blocks = [
        f"[doc_index={doc_idx}] {doc.filename}\n{_full_doc_text(doc)}"
        for doc_idx, doc in enumerate(prior_docs)
    ]
    full_docs_block = "\n\n---\n\n".join(full_blocks)
    if len(full_docs_block) <= hybrid_total:
        logger.info(
            f"Hybrid comparison: using full text for all {len(prior_docs)} docs "
            f"({len(full_docs_block)} chars)"
        )
        return full_docs_block

    # The integrated mode must keep every cited document in the one prompt. If
    # the combined full text is too large, divide the input budget across all
    # documents and compact each one independently (RAG first, keyword fallback).
    separator_chars = len("\n\n---\n\n") * max(0, len(prior_docs) - 1)
    header_chars = sum(
        len(f"[doc_index={doc_idx}] {doc.filename}\n")
        for doc_idx, doc in enumerate(prior_docs)
    )
    available_text_chars = max(0, hybrid_total - separator_chars - header_chars)
    per_doc_budget = max(hybrid_min, available_text_chars // len(prior_docs))
    blocks = []
    for doc_idx, doc in enumerate(prior_docs):
        # per_doc_budget瑜?_build_doc_text ?대? ?좊퀎 ?④퀎??吏곸젒 ?꾨떖?쒕떎.
        # ?댁쟾?먮뒗 ?좊퀎 ??[:per_doc_budget] 濡?臾몄옄?댁쓣 ?욎뿉???섎옄?붾뜲,
        # 臾몄꽌 ?쒖꽌?濡?異쒕젰???띿뒪?몃? ?욎뿉???먮Ⅴ硫??ㅼそ???듭떖 ?⑤씫???섎젮?섍컮??
        doc_text = _build_doc_text(
            doc,
            elements,
            max_chars=per_doc_budget,
            engine=engine,
            settings=settings,
        )
        blocks.append(
            f"[doc_index={doc_idx}] {doc.filename}\n"
            f"{doc_text}"
        )
    return "\n\n---\n\n".join(blocks)


def _claim_keywords(elements: List[ClaimElement]) -> List[str]:
    text = " ".join(e.text for e in elements)
    tokens = re.findall(r"[A-Za-z0-9가-힣]{2,}", text.lower())
    stopwords = {
        "하는", "하고", "하며", "포함", "포함하는", "구비", "구비하는", "상기",
        "및", "또는", "위해", "위한", "방법", "장치", "시스템", "단계",
        "the", "and", "for", "with", "that", "this", "from", "into", "wherein",
    }
    seen = set()
    keywords = []
    for token in tokens:
        if token in stopwords or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
    return keywords[:80]


_SYSTEM_BATCH = """당신은 특허 구성대비 전문가입니다.
청구항 구성요소와 인용발명의 원문을 중립적으로 비교하십시오.
판정은 동일, 실질적 동일, 일부 차이, 일부 유사, 차이, 대응 없음 중 하나만 사용합니다.
quote는 인용발명의 원문을 그대로 인용하고, 판단_이유에는 대응점과 차이만 간결하게 적습니다.
분석 과정이나 설명문 없이 요청된 JSON 배열만 출력하십시오."""


# ---------------------------------------------------------------------------
# ?몄슜 寃利???臾대즺 臾몄옄??寃??(LLM ?몄텧 ?놁쓬)
# ---------------------------------------------------------------------------

# 寃利??먯젙
_VERIFIED = "원문 확인"
_PARTIAL = "일부 일치(요약 또는 생략 가능성)"
_NOT_FOUND = "원문 미확인 — 인용문 검토 필요"
_EMPTY = "인용 없음"
_SHORT = "인용문이 너무 짧아 검증 불가"


def _probe_status(probe_text: str, corpus: str) -> Optional[str]:
    """??諛쒖톸 ?좊쭑??corpus???덈뒗吏 ?먯젙: 'verified' | 'partial' | None.
    ??70???꾩쟾?쇱튂 ??verified, ??30???쇱튂 ?먮뒗 ?⑥뼱 60%+ ?쇱튂 ??partial."""
    probe_full = probe_text[:70].lower()
    if probe_full and probe_full in corpus:
        return "verified"
    probe_short = probe_text[:30].lower()
    if probe_short and probe_short in corpus:
        return "partial"
    words = [w for w in probe_full.split() if len(w) >= 3]
    if words:
        ratio = sum(1 for w in words if w in corpus) / len(words)
        if ratio >= 0.60:
            return "partial"
    return None


def verify_quotes(
    matches: List[ElementMatch],
    prior_docs: List[ExtractedDocument],
    min_quote_len: int = 15,
) -> List[Dict]:
    """
    媛?ElementMatch??quote媛 ?ㅼ젣 臾몄꽌??議댁옱?섎뒗吏 臾몄옄??寃?됱쑝濡?寃利앺븳??
    LLM ?몄텧 ?놁씠 利됱떆 ?ㅽ뻾?쒕떎.

    諛섑솚媛? [{"label": "A", "status": "verified"|"partial"|"not_found"|"empty"|"short",
               "icon": "info", "message": "..."}]
    """
    results = []
    corpus_cache: Dict[int, str] = {}  # 臾몄꽌 ?꾩껜 ?띿뒪??寃고빀쨌?뚮Ц??蹂?섏? 臾몄꽌??1?뚮쭔

    for m in matches:
        label = m.label
        quote = (m.quote or "").strip()

        if not quote:
            results.append({"label": label, "status": "empty",
                             "icon": "info", "message": f"({label}) {_EMPTY}"})
            continue

        if len(quote) < min_quote_len:
            results.append({"label": label, "status": "short",
                             "icon": "warning", "message": f"({label}) {_SHORT}: \"{quote}\""})
            continue

        # ?대떦 ?몄슜諛쒕챸 臾몄꽌 ?띿뒪???섏쭛
        doc_idx = m.cited_invention_index
        doc = prior_docs[doc_idx] if 0 <= doc_idx < len(prior_docs) else None
        if doc is None:
            results.append({"label": label, "status": "no_doc",
                             "icon": "warning", "message": f"({label}) 인용발명 문서를 찾을 수 없음"})
            continue

        # 寃??????띿뒪?? paragraphs + pages + raw_text ?꾨? ?⑹묠
        if doc_idx not in corpus_cache:
            corpus_cache[doc_idx] = (
                " ".join(doc.paragraphs.values()) + " "
                + " ".join(doc.pages.values()) + " "
                + doc.raw_text
            ).lower()
        search_corpus = corpus_cache[doc_idx]

        # '...'濡?異뺤빟??諛쒖톸??媛??좊쭑???곕줈 寃利앺븳?? 異뺤빟???놁쑝硫?諛쒖톸 ?꾩껜媛
        # ?⑥씪 ?좊쭑???섏뼱 湲곗〈 ?숈옉(??70/30??+ ?⑥뼱 ?쇱튂)怨??숈씪?섍쾶 泥섎━?쒕떎.
        segments = [
            s.strip() for s in re.split(r"\s*(?:…|\.{3,})\s*", quote)
            if len(s.strip()) >= min_quote_len
        ]
        if not segments:
            segments = [quote]

        seg_statuses = [_probe_status(seg, search_corpus) for seg in segments]
        if all(s == "verified" for s in seg_statuses):
            results.append({"label": label, "status": "verified",
                             "icon": "info", "message": f"({label}) {_VERIFIED}"})
        elif any(s in ("verified", "partial") for s in seg_statuses):
            found = sum(1 for s in seg_statuses if s in ("verified", "partial"))
            results.append({"label": label, "status": "partial",
                             "icon": "warning",
                             "message": f"({label}) {_PARTIAL} (인용 구간 {found}/{len(seg_statuses)} 확인)"})
        else:
            results.append({"label": label, "status": "not_found",
                             "icon": "info", "message": f"({label}) {_NOT_FOUND}"})

    return results


# ---------------------------------------------------------------------------
# 蹂닿퀬???앹꽦 ?④퀎: 罹먯떆?먯꽌 濡쒕뱶
# ---------------------------------------------------------------------------

def load_comparisons(job_dir: str, doc_idx: int) -> Optional[Dict]:
    """??λ맂 ?鍮?寃곌낵 濡쒕뱶"""
    path = Path(job_dir) / f"comparisons_{doc_idx}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid comparison cache %s: %s", path, exc)
        return None


def _cache_is_compatible(
    cache: Optional[Dict],
    require_rag: bool,
    comparison_mode: Optional[str] = None,
) -> bool:
    if not cache:
        return False
    meta = cache.get(_CACHE_META_KEY, {})
    if meta.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return False
    if require_rag and not bool(meta.get("use_rag_retrieval")):
        return False
    if comparison_mode is not None:
        cached_mode = _comparison_mode(meta.get("comparison_mode", "per_doc"))
        if cached_mode != _comparison_mode(comparison_mode):
            return False
    return True


def reset_incompatible_comparison_caches(
    job_dir: str,
    num_docs: int,
    settings: Settings,
) -> bool:
    """Clear derived comparison caches when their input strategy changed.

    Cache metadata is stored per document, while a file can contain several
    claims. Clearing the whole derived cache prevents old per-document results
    from being mixed with new integrated-mode results in citation-chain scoring.
    """
    expected_mode = _comparison_mode(getattr(settings, "comparison_mode", "per_doc"))
    expected_rag = bool(getattr(settings, "use_rag_retrieval", False))
    expected_top_k = int(getattr(settings, "rag_top_k", 20) or 20)
    expected_reranker = bool(getattr(settings, "use_reranker", False))
    expected_reranker_top_k = int(getattr(settings, "reranker_top_k", 10) or 10)
    reset_any = False

    for doc_idx in range(num_docs):
        path = Path(job_dir) / f"comparisons_{doc_idx}.json"
        cache = load_comparisons(job_dir, doc_idx)
        if not cache:
            continue
        meta = cache.get(_CACHE_META_KEY, {})
        cached_mode = _comparison_mode(meta.get("comparison_mode", "per_doc"))
        try:
            cached_top_k = int(meta.get("rag_top_k", 20) or 20)
        except (TypeError, ValueError):
            cached_top_k = 20
        incompatible = (
            meta.get("schema_version") != _CACHE_SCHEMA_VERSION
            or cached_mode != expected_mode
            or bool(meta.get("use_rag_retrieval", False)) != expected_rag
            or (expected_rag and cached_top_k != expected_top_k)
            or (expected_rag and bool(meta.get("use_reranker", False)) != expected_reranker)
            or (expected_rag and expected_reranker and int(meta.get("reranker_top_k", 10) or 10) != expected_reranker_top_k)
        )
        if not incompatible:
            continue

        fresh_cache = {
            _CACHE_META_KEY: {
                "schema_version": _CACHE_SCHEMA_VERSION,
                "comparison_mode": expected_mode,
                "use_rag_retrieval": expected_rag,
                "rag_top_k": expected_top_k,
                "use_reranker": expected_reranker,
                "reranker_top_k": expected_reranker_top_k,
            }
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(fresh_cache, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp_path.replace(path)
        reset_any = True

    return reset_any


def get_matches_from_cache(
    claim: ParsedClaim,
    prior_docs: List[ExtractedDocument],
    job_dir: str,
    allowed_docs: Optional[List[int]] = None,
    require_rag: bool = False,
    comparison_mode: Optional[str] = None,
) -> tuple[List[ElementMatch], bool]:
    """罹먯떆?먯꽌 ?대떦 泥?뎄??쓽 ElementMatch 紐⑸줉??諛섑솚?쒕떎.

    醫낆냽??룄 ?먭린 泥?뎄??踰덊샇瑜?罹먯떆 ?ㅻ줈 ?ъ슜?쒕떎(異붽? 援ъ꽦 ?꾩슜 ?먯젙).
    怨쇨굅?먮뒗 遺紐??낅┰??罹먯떆瑜?怨듭쑀?덉쑝?? 洹몃윭硫?醫낆냽??쓽 異붽? 援ъ꽦??
    ?대뼡 臾명뿄怨쇰룄 ?鍮꾨릺吏 ?딆븘 遺紐⑦빆?????곗씤 ?몄슜諛쒕챸(?? ?몄슜諛쒕챸 3)??
    異붽? 援ъ꽦??媛쒖떆?대룄 諛섏쁺?섏? 紐삵뻽??

    ??踰덉㎏ 諛섑솚媛?cached)? ??泥?뎄???먯떊???ㅺ? 罹먯떆??議댁옱?덈뒗吏 ?щ???
    matches媛 ?꾨? '????놁쓬'?댁뼱??罹먯떆???덉뿀?ㅻ㈃ ?щ?鍮꾨? 嫄대꼫?곌린 ?꾪빐
    ?몄텧遺媛 ?ъ슜?쒕떎. ?ㅺ? ?놁쑝硫??몄텧遺媛 ??泥?뎄??쓽 ?鍮꾨? ?덈줈 ?ㅽ뻾?쒕떎.
    """
    num_docs = len(prior_docs)
    claim_key = str(claim.claim_number)

    cached_doc_count = 0
    doc_results = []
    for doc_idx in range(num_docs):
        cache = load_comparisons(job_dir, doc_idx)
        if _cache_is_compatible(cache, require_rag, comparison_mode) and claim_key in cache:
            doc_results.append(cache[claim_key])
            cached_doc_count += 1
        else:
            doc_results.append([])

    return _select_best_matches(claim.elements, doc_results, num_docs, allowed_docs), cached_doc_count == num_docs


def get_cached_doc_indices(
    job_dir: str,
    claim_number: int,
    num_docs: int,
    require_rag: bool = False,
    comparison_mode: Optional[str] = None,
) -> set[int]:
    """Return active document indices that already have comparison cache for a claim."""
    claim_key = str(claim_number)
    cached: set[int] = set()
    for doc_idx in range(num_docs):
        cache = load_comparisons(job_dir, doc_idx)
        if _cache_is_compatible(cache, require_rag, comparison_mode) and claim_key in cache:
            cached.add(doc_idx)
    return cached


async def analyze_claim_elements_for_docs(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    doc_indices: List[int],
    settings: Settings,
    job_dir: Optional[str] = None,
    claim_number: Optional[int] = None,
) -> None:
    """Compare one claim only against selected prior documents and cache the results.

    This is used when a refreshed job reuses comparison cache for unchanged PDFs and
    only newly added PDFs need an extra LLM comparison.
    """
    for doc_idx in doc_indices:
        if doc_idx < 0 or doc_idx >= len(prior_docs):
            continue
        result = await _batch_judge_for_doc(elements, prior_docs[doc_idx], doc_idx, settings)
        if job_dir is not None and claim_number is not None:
            _merge_into_cache(job_dir, doc_idx, str(claim_number), result, settings)
            logger.info(
                f"[partial cache saved] comparisons_{doc_idx}.json claim {claim_number} "
                f"({len(result)} elements)"
            )


# ---------------------------------------------------------------------------
# ?대갚: 罹먯떆 ?놁쓣 ???⑥씪 泥?뎄??利됱꽍 ?鍮?+ 寃곌낵瑜?罹먯떆?????
# ---------------------------------------------------------------------------

async def analyze_claim_elements(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
    job_dir: Optional[str] = None,
    claim_number: Optional[int] = None,
) -> List[ElementMatch]:
    """?⑥씪 泥?뎄??援ъ꽦?붿냼瑜??몄슜諛쒕챸 ?꾨Ц怨??鍮?
    job_dir + claim_number媛 二쇱뼱吏硫?寃곌낵瑜?comparisons_{doc_idx}.json??罹먯떆 ???
    ?ㅼ쓬 踰?媛숈? 泥?뎄???ъ깮????LLM ?鍮꾨? ?ㅽ궢?????덈떎.
    """
    num_docs = len(prior_docs)
    doc_results = []
    for doc_idx in range(num_docs):
        result = await _batch_judge_for_doc(elements, prior_docs[doc_idx], doc_idx, settings)
        doc_results.append(result)

        # ?? 罹먯떆 ????????????????????????????????????????????????????????
        if job_dir is not None and claim_number is not None:
            _merge_into_cache(job_dir, doc_idx, str(claim_number), result, settings)
            logger.info(
                f"[cache saved] comparisons_{doc_idx}.json claim {claim_number} "
                f"({len(result)} elements)"
            )
        # ??????????????????????????????????????????????????????????????????

    return _select_best_matches(elements, doc_results, num_docs)


async def analyze_claim_elements_hybrid(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
    job_dir: Optional[str] = None,
    claim_number: Optional[int] = None,
    doc_index_map: Optional[List[int]] = None,
) -> List[ElementMatch]:
    """
    Compare one claim against all prior documents in a single LLM call.

    Hybrid mode still stores a per-document, per-element judgment matrix.
    Citation-chain scoring depends on comparisons_{doc_idx}.json representing
    each document's own coverage, not only the globally best document per element.
    """
    num_docs = len(prior_docs)
    original_doc_indices = doc_index_map or list(range(num_docs))
    if num_docs <= 1:
        if not prior_docs:
            return _select_best_matches(elements, [], 0)
        original_idx = original_doc_indices[0] if original_doc_indices else 0
        result = await _batch_judge_for_doc(elements, prior_docs[0], original_idx, settings)
        if job_dir is not None and claim_number is not None:
            _merge_into_cache(job_dir, original_idx, str(claim_number), result, settings)
        return [
            match.model_copy(update={"cited_invention_index": original_idx})
            for match in _select_best_matches(elements, [result], 1)
        ]

    doc_results = [
        [
            {
                "label": elem.label,
                "found": False,
                "judgment": "대응 없음",
                "quote": "",
                "chunk_id": "",
                "판단_이유": "",
            }
            for elem in elements
        ]
        for _ in range(num_docs)
    ]

    try:
        hybrid_results = await _batch_judge_hybrid(elements, prior_docs, settings)
    except CompareFailed:
        # 응답 형식 오류로 LLM을 자동 재호출하지 않고 사용자가 재시도할 수 있게 전달한다.
        raise
    except Exception as e:
        logger.error(f"Hybrid batch judge error: {e}")
        raise CompareFailed(f"하이브리드 구성대비 LLM 호출 실패: {e}") from e
    else:
        for item in hybrid_results:
            label = item.get("label", "")
            try:
                doc_idx = int(item.get("doc_index", item.get("cited_invention_index", 0)))
            except (TypeError, ValueError):
                doc_idx = 0
            if doc_idx < 0 or doc_idx >= num_docs:
                doc_idx = 0

            target = next(
                (m for m in doc_results[doc_idx]
                 if normalize_label(m.get("label")) == normalize_label(label)),
                None,
            )
            if target is None:
                continue
            target.update({
                "label": label,
                "found": bool(item.get("found", False)),
                "judgment": item.get("judgment", "대응 없음"),
                "quote": item.get("quote", ""),
                "chunk_id": item.get("chunk_id", ""),
                "판단_이유": item.get("판단_이유", item.get("similarity_reason", "")),
            })

    if job_dir is not None and claim_number is not None:
        for doc_idx, results in enumerate(doc_results):
            original_idx = original_doc_indices[doc_idx] if doc_idx < len(original_doc_indices) else doc_idx
            _merge_into_cache(job_dir, original_idx, str(claim_number), results, settings)
            logger.info(
                f"[hybrid cache saved] comparisons_{original_idx}.json claim {claim_number} "
                f"({len(results)} elements)"
            )

    matches = _select_best_matches(elements, doc_results, num_docs)
    return [
        match.model_copy(update={
            "cited_invention_index": original_doc_indices[match.cited_invention_index]
            if match.cited_invention_index < len(original_doc_indices)
            else match.cited_invention_index
        })
        for match in matches
    ]


def _merge_into_cache(
    job_dir: str,
    doc_idx: int,
    claim_key: str,
    results: List[Dict],
    settings: Optional[Settings] = None,
) -> None:
    """comparisons_{doc_idx}.json ??claim_key 寃곌낵瑜?蹂묓빀 ???
    湲곗〈 ?ㅻⅨ 泥?뎄??罹먯떆??蹂댁〈?섍퀬 ?대떦 ?ㅻ쭔 ??뼱?대떎."""
    path = Path(job_dir) / f"comparisons_{doc_idx}.json"
    cache: Dict = {}
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    cache[claim_key] = results
    if settings is not None:
        cache[_CACHE_META_KEY] = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "comparison_mode": _comparison_mode(getattr(settings, "comparison_mode", "per_doc")),
            "use_rag_retrieval": bool(getattr(settings, "use_rag_retrieval", False)),
            "rag_top_k": int(getattr(settings, "rag_top_k", 20) or 20),
            "use_reranker": bool(getattr(settings, "use_reranker", False)),
            "reranker_top_k": int(getattr(settings, "reranker_top_k", 10) or 10),
        }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


async def _batch_judge_for_doc(
    elements: List[ClaimElement],
    doc: ExtractedDocument,
    doc_idx: int,
    settings: Settings,
) -> List[Dict]:
    full_text = _build_doc_text(doc, elements, engine=settings.engine, settings=settings)

    elements_text = "\n".join(f"({e.label}) {e.text}" for e in elements)

    prompt = render_prompt(
        "prompt_compare_single.txt",
        doc_filename=doc.filename,
        elements_text=elements_text,
        full_text=full_text,
    )

    return await _call_and_parse_comparison(
        prompt,
        elements,
        settings,
        context=f"인용발명 {doc_idx + 1} 구성대비",
    )


# ---------------------------------------------------------------------------
# ?대? ?ы띁
# ---------------------------------------------------------------------------

async def _batch_judge_hybrid(
    elements: List[ClaimElement],
    prior_docs: List[ExtractedDocument],
    settings: Settings,
) -> List[Dict]:
    docs_block = _build_hybrid_docs_block(prior_docs, elements, engine=settings.engine, settings=settings)
    elements_text = "\n".join(f"({e.label}) {e.text}" for e in elements)
    doc_list = "\n".join(
        f"- doc_index={idx}: {doc.filename}"
        for idx, doc in enumerate(prior_docs)
    )

    prompt = render_prompt(
        "prompt_compare_hybrid.txt",
        doc_list=doc_list,
        elements_text=elements_text,
        docs_block=docs_block,
    )

    return await _call_and_parse_comparison(
        prompt,
        elements,
        settings,
        expected_doc_indices=list(range(len(prior_docs))),
        context="하이브리드 구성대비",
    )


async def _call_and_parse_comparison(
    prompt: str,
    elements: List[ClaimElement],
    settings: Settings,
    *,
    expected_doc_indices: Optional[List[int]] = None,
    context: str,
) -> List[Dict]:
    """Call the comparison model once and validate its recovered response locally."""
    system = load_prompt("system_compare.txt", _SYSTEM_BATCH)
    try:
        response = await call_ai(prompt, system, settings, agent="compare")
    except Exception as exc:
        raise CompareFailed(f"{context} LLM 호출 실패: {exc}") from exc

    try:
        return _parse_json_array(response, elements, expected_doc_indices)
    except CompareFailed as exc:
        raise CompareFailed(f"{context} 응답 형식 검증 실패: {exc}") from exc


def _select_best_matches(
    elements: List[ClaimElement],
    doc_results: List[List[Dict]],
    num_docs: int,
    allowed_docs: Optional[List[int]] = None,
) -> List[ElementMatch]:
    # allowed_docs: 蹂닿퀬?쒓? ?몄슜 泥댁씤??梨꾪깮???몄슜諛쒕챸留?洹쇨굅濡??쇰룄濡??쒗븳?쒕떎.
    # 誘몄???None)?대㈃ ?꾩껜 臾몄꽌?먯꽌 ?좏깮(以鍮??鍮??④퀎 湲곕낯 ?숈옉).
    fallback_idx = allowed_docs[0] if allowed_docs else 0
    # Primary document first so it wins ties ??doc[0] is not always the primary.
    if allowed_docs:
        priority_order = [d for d in allowed_docs if d < num_docs]
    else:
        priority_order = list(range(num_docs))
    matches = []
    for elem in elements:
        best_match, best_rank, best_doc_idx = None, -1, fallback_idx
        for doc_idx in priority_order:
            if doc_idx >= len(doc_results):
                continue
            item = next(
                (m for m in doc_results[doc_idx]
                 if normalize_label(m.get("label")) == normalize_label(elem.label)),
                None,
            )
            if item is None:
                continue
            rank = _JUDGMENT_RANK.get(item.get("judgment", "대응 없음"), 0)
            if rank > best_rank:
                best_rank, best_match, best_doc_idx = rank, item, doc_idx

        if best_match and best_rank > 0:
            matches.append(ElementMatch(
                label=elem.label,
                found=bool(best_match.get("found", False)),
                quote=_shorten_quote(best_match.get("quote", "")),
                chunk_id=best_match.get("chunk_id", ""),
                judgment=best_match.get("judgment", "대응 없음"),
                cited_invention_index=best_doc_idx,
                similarity_reason=best_match.get("판단_이유", best_match.get("similarity_reason", "")),
            ))
        else:
            matches.append(ElementMatch(
                label=elem.label, found=False, quote="", chunk_id="",
                judgment="대응 없음", cited_invention_index=fallback_idx, similarity_reason="",
            ))
    return matches


def _extract_json_arrays(text: str) -> List[Dict]:
    """蹂몃Ц???깆옣?섎뒗 理쒖긽??JSON 諛곗뿴?ㅼ쓣 ?쒖꽌?濡??뚯떛????ぉ???⑹튇??

    Gemini ?깆씠 諛곗뿴 ?ㅼ뿉 ?ㅻ챸/以묐났 諛곗뿴???㏓텤??'Extra data' ?뚯떛 ?ㅻ쪟媛
    ?섎뒗 寃쎌슦瑜?寃щ뵖?? greedy ?뺢퇋??\[.*\])怨??щ━ 泥?諛곗뿴 ??援곕뜑?붽린瑜?
    臾댁떆?섍퀬, 諛곗뿴???섎줈 履쇨컻???덉뼱????ぉ??紐⑥븘 ?곗씠???먯떎??留됰뒗??
    """
    decoder = json.JSONDecoder()
    items: List[Dict] = []
    idx = 0
    while True:
        start = text.find("[", idx)
        if start == -1:
            break
        try:
            arr, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(arr, list):
            items.extend(item for item in arr if isinstance(item, dict))
            idx = end
        else:
            idx = start + 1
    return items


def _parse_json_array(
    response: str,
    elements: List[ClaimElement],
    expected_doc_indices: Optional[List[int]] = None,
) -> List[Dict]:
    text = re.sub(r"```(?:json)?", "", response.strip()).replace("```", "").strip()
    parsed = _extract_json_arrays(text)
    if not parsed:
        raise CompareFailed(
            f"구성대비 응답에서 JSON 배열을 찾지 못했습니다. 응답 길이: {len(response)}자"
        )

    expected_labels = {normalize_label(element.label) for element in elements}
    expected_docs = set(expected_doc_indices or [])
    normalized: List[Dict] = []
    invalid_reasons: List[str] = []
    required_fields = {"label", "found", "quote", "chunk_id", "judgment"}
    judgment_aliases = {
        "부분 차이": "일부 차이",
        "부분 유사": "일부 유사",
        "유사": "일부 유사",
        "없음": "대응 없음",
    }

    for item in parsed:
        schema_markers = {
            "label", "claim_element", "found", "quote", "judgment", "doc_index"
        }
        if not schema_markers.intersection(item):
            continue

        missing_fields = required_fields.difference(item)
        if "판단_이유" not in item and "similarity_reason" not in item:
            missing_fields.add("판단_이유")
        if expected_doc_indices is not None and "doc_index" not in item:
            missing_fields.add("doc_index")
        if missing_fields:
            invalid_reasons.append(
                "필수 필드 누락: " + ", ".join(sorted(missing_fields))
            )
            continue

        label = normalize_label(str(item.get("label", "")))
        if not label or (expected_labels and label not in expected_labels):
            if "claim_element" in item and "label" not in item:
                invalid_reasons.append("claim_element 대신 label 필드를 사용해야 함")
            else:
                invalid_reasons.append(f"알 수 없는 label: {item.get('label', '')!r}")
            continue

        doc_idx: Optional[int] = None
        if expected_doc_indices is not None:
            try:
                doc_idx = int(item.get("doc_index"))
            except (TypeError, ValueError):
                invalid_reasons.append(f"{label}의 doc_index가 정수가 아님")
                continue
            if doc_idx not in expected_docs:
                invalid_reasons.append(f"{label}의 doc_index가 범위를 벗어남: {doc_idx}")
                continue

        judgment = str(item.get("judgment", "대응 없음")).strip()
        judgment = judgment_aliases.get(judgment, judgment)
        if judgment not in _JUDGMENT_RANK:
            invalid_reasons.append(f"{label}의 허용되지 않은 judgment: {judgment!r}")
            continue

        quote = _shorten_quote(str(item.get("quote", "") or ""))
        found_value = item.get("found", False)
        if isinstance(found_value, str):
            found = found_value.strip().lower() in {"true", "1", "yes"}
        else:
            found = bool(found_value)
        if found and not quote:
            invalid_reasons.append(f"{label}의 found=true 항목에 quote가 없음")
            continue
        if not found and (quote or judgment != "대응 없음"):
            invalid_reasons.append(
                f"{label}의 found=false 항목은 빈 quote와 대응 없음 판정이어야 함"
            )
            continue

        normalized_item = dict(item)
        normalized_item.update({
            "label": label,
            "found": found,
            "quote": quote,
            "chunk_id": str(item.get("chunk_id", "") or ""),
            "judgment": judgment,
            "판단_이유": str(item.get("판단_이유", item.get("similarity_reason", "")) or ""),
        })
        if doc_idx is not None:
            normalized_item["doc_index"] = doc_idx
        normalized.append(normalized_item)

    if not normalized:
        detail = f" ({'; '.join(invalid_reasons[:3])})" if invalid_reasons else ""
        raise CompareFailed(f"구성대비 응답에 유효한 구성요소 판정이 없습니다.{detail}")

    if expected_doc_indices is None:
        expected_keys = expected_labels
        actual_keys = {item["label"] for item in normalized}
        missing_keys = sorted(expected_keys - actual_keys)
        missing_text = ", ".join(missing_keys)
    else:
        expected_keys = {
            (doc_idx, label)
            for doc_idx in expected_doc_indices
            for label in expected_labels
        }
        actual_keys = {(item["doc_index"], item["label"]) for item in normalized}
        missing_pairs = sorted(expected_keys - actual_keys)
        missing_text = ", ".join(
            f"doc_index={doc}/label={label}" for doc, label in missing_pairs
        )
    duplicate_count = len(normalized) - len(actual_keys)

    if missing_text or duplicate_count or invalid_reasons:
        details = []
        if missing_text:
            details.append(f"누락: {missing_text}")
        if duplicate_count:
            details.append(f"중복 항목: {duplicate_count}개")
        if invalid_reasons:
            details.append("형식 오류: " + "; ".join(invalid_reasons[:3]))
        raise CompareFailed("구성대비 응답이 완전하지 않습니다. " + " / ".join(details))
    return normalized
