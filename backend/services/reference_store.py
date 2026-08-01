"""
SQLite reference store.

사건별 원문 단락·메타데이터·청크·보고서 인용 항목을 보관한다. 판단 근거를
사후에 원문과 대조·재현하기 위한 감사용 저장소이며, 비교 시점의 문헌 선별은
`citation_extractor`가 메모리 상의 청크로 수행한다(검색 인덱스를 두지 않는다).
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import logging
import sqlite3
from pathlib import Path
from typing import List

from backend.models.schemas import ExtractedDocument

logger = logging.getLogger(__name__)

DB_NAME = "reference.sqlite"


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
                conn.execute(
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
