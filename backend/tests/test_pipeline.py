"""Unit and integration test suite for Multi-Doc RAG pipeline components."""

import os
import sys
import pytest
from llama_index.core.schema import TextNode

# Ensure backend root is on path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.sparse_store import tokenize, upsert_sparse_chunks, query_sparse, delete_sparse_document
from app.retrieval_service import reciprocal_rank_fusion
from app.document_processor import _clamp_node_length, _format_docx_table


def test_tokenize():
    """Test tokenization handles punctuation and casing cleanly."""
    tokens = tokenize("FastAPI, Pinecone & Python 3.11!")
    assert "fastapi" in tokens
    assert "pinecone" in tokens
    assert "python" in tokens
    assert "3" in tokens or "11" in tokens
    assert "," not in tokens
    assert "&" not in tokens


def test_sparse_store_upsert_query_delete(tmp_path):
    """Test SQLite sparse store CRUD operations in temporary database."""
    test_db = str(tmp_path / "test_sparse.db")

    chunks_doc1 = [
        {"text": "Python is a popular programming language for AI and machine learning.", "page": 1, "chunk_index": 0},
        {"text": "FastAPI is a modern web framework for building APIs with Python.", "page": 1, "chunk_index": 1},
    ]
    chunks_doc2 = [
        {"text": "Pinecone is a managed cloud vector database for semantic search.", "page": 2, "chunk_index": 0},
    ]

    # Test upsert
    n1 = upsert_sparse_chunks(chunks_doc1, "doc1.pdf", db_path=test_db)
    n2 = upsert_sparse_chunks(chunks_doc2, "doc2.pdf", db_path=test_db)
    assert n1 == 2
    assert n2 == 1

    # Test query all
    results = query_sparse("FastAPI web framework", top_k=5, db_path=test_db)
    assert len(results) > 0
    assert results[0]["doc_name"] == "doc1.pdf"
    assert "FastAPI" in results[0]["text"]

    # Test scoped query by doc_filter
    doc2_results = query_sparse("Python", top_k=5, doc_filter="doc2.pdf", db_path=test_db)
    assert len(doc2_results) == 0  # doc2 does not contain Python

    # Test delete
    deleted = delete_sparse_document("doc1.pdf", db_path=test_db)
    assert deleted == 2
    after_del = query_sparse("FastAPI", top_k=5, db_path=test_db)
    assert len(after_del) == 0


def test_reciprocal_rank_fusion():
    """Test RRF fusion scoring and rank ordering."""
    dense = [
        {"id": "chunk_A", "score": 0.95},
        {"id": "chunk_B", "score": 0.85},
    ]
    sparse = [
        {"id": "chunk_B", "score": 4.5},
        {"id": "chunk_C", "score": 3.2},
    ]

    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    assert len(fused) == 3

    # chunk_B appears in both lists (dense rank 2, sparse rank 1)
    # dense rank 2 -> 1 / (60 + 2) = 1/62 = 0.016129
    # sparse rank 1 -> 1 / (60 + 1) = 1/61 = 0.016393
    # Total for B = ~0.032522
    # chunk_A appears only in dense (rank 1) -> 1/61 = 0.016393
    # chunk_C appears only in sparse (rank 2) -> 1/62 = 0.016129
    # chunk_B MUST be ranked #1
    assert fused[0]["id"] == "chunk_B"
    assert fused[0]["rrf_score"] > fused[1]["rrf_score"]


def test_clamp_node_length():
    """Test that oversized chunks are sub-chunked under character ceiling."""
    long_text = "Sentence number " * 150  # ~2400 characters
    node = TextNode(text=long_text, metadata={"page": 1, "doc_name": "test.txt"})

    clamped = _clamp_node_length([node], max_chars=1000, overlap=100)
    assert len(clamped) > 1
    for sub in clamped:
        assert len(sub.get_content()) <= 1000
        assert sub.metadata["doc_name"] == "test.txt"


def test_short_node_not_clamped():
    """Test that nodes already under max_chars are unchanged."""
    short_text = "This is a short sentence."
    node = TextNode(text=short_text, metadata={"page": 1})
    clamped = _clamp_node_length([node], max_chars=1000)
    assert len(clamped) == 1
    assert clamped[0].get_content() == short_text
