"""
SQLite reference store.

Stores original paragraphs, metadata, chunks, FTS5 lexical index, and report
reference entries per case. JSON cache files remain for compatibility, while
SQLite becomes the retrieval/verification database.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import List, Optional, Protocol

from backend.models.schemas import ExtractedDocument

logger = logging.getLogger(__name__)

DB_NAME = "reference.sqlite"


class ReferenceRepository(Protocol):
    """Repository boundary for original text/metadata/reference storage.

    SQLite is the MVP implementation. A PostgreSQL implementation can keep the
    same method contract and map JSON text columns to JSONB later.
    """

    def save_case_artifacts(self, docs: List[ExtractedDocument], manifest: list[dict]) -> None:
        ...

    def save_reference_entries(self, entries: list[dict]) -> None:
        ...

    def search_paragraphs(self, doc_id: str, query: str, limit: int = 20) -> Optional[list[dict]]:
        ...


class SQLiteReferenceRepository:
    def __init__(self, case_dir: Path):
        self.case_dir = case_dir

    def save_case_artifacts(self, docs: List[ExtractedDocument], manifest: list[dict]) -> None:
        save_case_artifacts_sqlite(self.case_dir, docs, manifest)

    def save_reference_entries(self, entries: list[dict]) -> None:
        save_reference_entries_sqlite(self.case_dir, entries)

    def search_paragraphs(self, doc_id: str, query: str, limit: int = 20) -> Optional[list[dict]]:
        return search_paragraphs_fts5(self.case_dir, doc_id, query, limit)


def db_path_for_case(case_dir: Path) -> Path:
    return case_dir / DB_NAME


def case_dir_for_job(cases_root: Path, job_id: str) -> Path:
    return cases_root / job_id


@contextmanager
def _connect(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_index INTEGER NOT NULL,
                filename TEXT NOT NULL,
                publication_no TEXT,
                title TEXT,
                document_type TEXT,
                pdf_path TEXT,
                raw_text_hash TEXT,
                page_layout_json TEXT,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS paragraphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                publication_no TEXT,
                title TEXT,
                page_no INTEGER,
                section TEXT,
                paragraph_no TEXT NOT NULL,
                claim_no TEXT,
                figure_no TEXT,
                reference_signs_json TEXT,
                original_text TEXT NOT NULL,
                normalized_text TEXT,
                text_hash TEXT,
                chunk_excluded INTEGER DEFAULT 0,
                exclusion_reason TEXT,
                UNIQUE(doc_id, paragraph_no),
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_type TEXT,
                publication_no TEXT,
                title TEXT,
                section TEXT,
                paragraph_no TEXT,
                paragraph_range_json TEXT,
                page_no INTEGER,
                page_range_json TEXT,
                original_text TEXT NOT NULL,
                normalized_text TEXT,
                text_hash TEXT,
                source TEXT,
                FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS reference_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                publication_no TEXT,
                title TEXT,
                used_in_case TEXT NOT NULL,
                claim_number INTEGER NOT NULL,
                role TEXT,
                rejection_type TEXT,
                key_paragraphs_json TEXT,
                matched_features_json TEXT,
                report_excerpt_json TEXT,
                UNIQUE(used_in_case, claim_number, publication_no, role)
            );
            """
        )
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS paragraph_fts USING fts5(
                    doc_id UNINDEXED,
                    paragraph_no UNINDEXED,
                    original_text,
                    normalized_text,
                    tokenize='unicode61'
                )
                """
            )
        except sqlite3.Error as exc:
            logger.warning(f"SQLite FTS5 unavailable; lexical search disabled: {exc}")


def save_case_artifacts_sqlite(
    case_dir: Path,
    docs: List[ExtractedDocument],
    manifest: list[dict],
) -> None:
    db_path = db_path_for_case(case_dir)
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM chunks")
        conn.execute("DELETE FROM paragraphs")
        conn.execute("DELETE FROM documents")
        try:
            conn.execute("DELETE FROM paragraph_fts")
        except sqlite3.Error:
            pass

        for doc in docs:
            resolved_doc_id = doc.doc_id or f"D{doc.doc_index + 1}"
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    doc_id, doc_index, filename, publication_no, title,
                    document_type, pdf_path, raw_text_hash, page_layout_json,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_doc_id,
                    doc.doc_index,
                    doc.filename,
                    doc.publication_no,
                    doc.title,
                    doc.document_type,
                    doc.pdf_path,
                    doc.metadata.get("raw_text_hash", ""),
                    json.dumps(
                        [layout.model_dump() for layout in (doc.page_layouts or [])],
                        ensure_ascii=False,
                    ),
                    json.dumps(doc.metadata or {}, ensure_ascii=False),
                ),
            )

            for rec in doc.paragraph_records or []:
                cur = conn.execute(
                    """
                    INSERT OR REPLACE INTO paragraphs (
                        doc_id, publication_no, title, page_no, section,
                        paragraph_no, claim_no, figure_no, reference_signs_json,
                        original_text, normalized_text, text_hash,
                        chunk_excluded, exclusion_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        resolved_doc_id,
                        rec.publication_no,
                        rec.title,
                        rec.page_no,
                        rec.section,
                        rec.paragraph_no,
                        rec.claim_no,
                        rec.figure_no,
                        json.dumps(rec.reference_signs or [], ensure_ascii=False),
                        rec.original_text,
                        rec.normalized_text,
                        rec.text_hash,
                        1 if rec.chunk_excluded else 0,
                        rec.exclusion_reason,
                    ),
                )
                row_id = cur.lastrowid
                if not rec.chunk_excluded:
                    try:
                        conn.execute(
                            """
                            INSERT INTO paragraph_fts(
                                rowid, doc_id, paragraph_no, original_text, normalized_text
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                row_id,
                                resolved_doc_id,
                                rec.paragraph_no,
                                rec.original_text,
                                rec.normalized_text,
                            ),
                        )
                    except sqlite3.Error:
                        pass

            for chunk in (doc.paragraph_chunks or []) + (doc.group_chunks or []):
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, doc_id, chunk_type, publication_no, title,
                        section, paragraph_no, paragraph_range_json, page_no,
                        page_range_json, original_text, normalized_text,
                        text_hash, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        resolved_doc_id,
                        chunk.chunk_type,
                        chunk.publication_no,
                        chunk.title,
                        chunk.section,
                        chunk.paragraph_no,
                        json.dumps(chunk.paragraph_range or [], ensure_ascii=False),
                        chunk.page_no,
                        json.dumps(chunk.page_range or [], ensure_ascii=False),
                        chunk.original_text,
                        chunk.normalized_text,
                        chunk.text_hash,
                        chunk.source,
                    ),
                )


def save_reference_entries_sqlite(case_dir: Path, entries: list[dict]) -> None:
    if not entries:
        return
    db_path = db_path_for_case(case_dir)
    init_db(db_path)
    with _connect(db_path) as conn:
        scopes = {
            (item.get("used_in_case", ""), int(item.get("claim_number", 0) or 0))
            for item in entries
        }
        for used_in_case, claim_number in scopes:
            conn.execute(
                "DELETE FROM reference_entries WHERE used_in_case = ? AND claim_number = ?",
                (used_in_case, claim_number),
            )
        for item in entries:
            conn.execute(
                """
                INSERT OR REPLACE INTO reference_entries (
                    publication_no, title, used_in_case, claim_number, role,
                    rejection_type, key_paragraphs_json, matched_features_json,
                    report_excerpt_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.get("publication_no", ""),
                    item.get("title", ""),
                    item.get("used_in_case", ""),
                    int(item.get("claim_number", 0) or 0),
                    item.get("role", ""),
                    item.get("rejection_type", ""),
                    json.dumps(item.get("key_paragraphs", []), ensure_ascii=False),
                    json.dumps(item.get("matched_features", []), ensure_ascii=False),
                    json.dumps(item.get("report_excerpt", []), ensure_ascii=False),
                ),
            )


def _fts_query_terms(text: str, max_terms: int = 24) -> str:
    tokens = re.findall(r"[A-Za-z0-9가-힣]{2,}", (text or "").lower())
    seen: set[str] = set()
    parts: list[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        safe = token.replace('"', '""')
        parts.append(f'"{safe}"')
        if len(parts) >= max_terms:
            break
    return " OR ".join(parts)


def search_paragraphs_fts5(
    case_dir: Path,
    doc_id: str,
    query: str,
    limit: int = 20,
) -> Optional[list[dict]]:
    db_path = db_path_for_case(case_dir)
    if not db_path.exists():
        return None
    match_query = _fts_query_terms(query)
    if not match_query:
        return []
    try:
        with _connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    p.doc_id,
                    p.paragraph_no,
                    p.original_text,
                    bm25(paragraph_fts) AS bm25_score
                FROM paragraph_fts
                JOIN paragraphs p ON p.id = paragraph_fts.rowid
                WHERE paragraph_fts MATCH ? AND p.doc_id = ?
                ORDER BY bm25_score ASC
                LIMIT ?
                """,
                (match_query, doc_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        logger.debug(f"SQLite FTS5 search skipped: {exc}")
        return None

