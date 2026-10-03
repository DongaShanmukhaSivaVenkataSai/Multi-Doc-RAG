"""Local persistent sparse storage and BM25 retrieval using SQLite."""

import os
import re
import sqlite3
import logging
from typing import Optional, List, Dict
from rank_bm25 import BM25Okapi
from app.config import settings

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(settings.upload_dir, "sparse_store.db")


def get_db_path(custom_path: Optional[str] = None) -> str:
    path = custom_path or DEFAULT_DB_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def init_db(db_path: Optional[str] = None):
    """Initialize SQLite table for local chunk metadata and text."""
    path = get_db_path(db_path)
    with sqlite3.connect(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                doc_name TEXT NOT NULL,
                page INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_doc_name ON chunks(doc_name)")
        conn.commit()


def upsert_sparse_chunks(
    chunks: list[dict], doc_name: str, db_path: Optional[str] = None
) -> int:
    """Store chunks in local SQLite for sub-5ms BM25 sparse retrieval."""
    init_db(db_path)
    path = get_db_path(db_path)
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM chunks WHERE doc_name = ?", (doc_name,))
        rows = [
            (
                f"{doc_name}_{chunk.get('chunk_index', idx)}",
                doc_name,
                chunk.get("page", 1),
                chunk.get("chunk_index", idx),
                chunk.get("text", ""),
            )
            for idx, chunk in enumerate(chunks)
        ]
        conn.executemany(
            "INSERT INTO chunks (id, doc_name, page, chunk_index, text) VALUES (?, ?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    logger.info(f"Indexed {len(chunks)} chunks into local sparse store for {doc_name}")
    return len(chunks)


def delete_sparse_document(doc_name: str, db_path: Optional[str] = None) -> int:
    """Delete chunks belonging to doc_name from local SQLite."""
    init_db(db_path)
    path = get_db_path(db_path)
    with sqlite3.connect(path) as conn:
        cursor = conn.execute("DELETE FROM chunks WHERE doc_name = ?", (doc_name,))
        deleted = cursor.rowcount
        conn.commit()
    logger.info(f"Deleted {deleted} chunks from sparse store for {doc_name}")
    return deleted


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens."""
    return re.findall(r"\w+", text.lower())


def query_sparse(
    query: str,
    top_k: int = 20,
    doc_filter: Optional[List[str] | str] = None,
    db_path: Optional[str] = None,
) -> list[dict]:
    """
    Perform high-speed BM25 sparse retrieval over local chunks without network calls.
    Supports filtering by a single document name or list of document names.
    """
    init_db(db_path)
    path = get_db_path(db_path)

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        if doc_filter:
            if isinstance(doc_filter, str):
                doc_filter = [doc_filter]
            placeholders = ",".join(["?"] * len(doc_filter))
            cursor = conn.execute(
                f"SELECT id, doc_name, page, chunk_index, text FROM chunks WHERE doc_name IN ({placeholders})",
                doc_filter,
            )
        else:
            cursor = conn.execute("SELECT id, doc_name, page, chunk_index, text FROM chunks")
        rows = cursor.fetchall()

    if not rows:
        return []

    chunks = [dict(row) for row in rows]
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        if score > 0:
            chunk_copy = chunk.copy()
            chunk_copy["bm25_score"] = float(score)
            scored_chunks.append(chunk_copy)

    scored_chunks.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored_chunks[:top_k]


def get_all_chunks(doc_filter: Optional[List[str] | str] = None, db_path: Optional[str] = None) -> list[dict]:
    """Fetch all chunks from local database (used for evaluation & testing)."""
    init_db(db_path)
    path = get_db_path(db_path)
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        if doc_filter:
            if isinstance(doc_filter, str):
                doc_filter = [doc_filter]
            placeholders = ",".join(["?"] * len(doc_filter))
            cursor = conn.execute(
                f"SELECT id, doc_name, page, chunk_index, text FROM chunks WHERE doc_name IN ({placeholders})",
                doc_filter,
            )
        else:
            cursor = conn.execute("SELECT id, doc_name, page, chunk_index, text FROM chunks")
        return [dict(r) for r in cursor.fetchall()]
