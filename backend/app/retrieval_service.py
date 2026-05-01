"""Hybrid retrieval: Dense (Pinecone) + Sparse (BM25) with Reciprocal Rank Fusion."""

import logging
from rank_bm25 import BM25Okapi
from app.pinecone_service import query_dense, get_all_chunks_text
from app.config import settings

logger = logging.getLogger(__name__)


def bm25_retrieval(
    query: str,
    chunks: list[dict],
    top_k: int = 20,
) -> list[dict]:
    """
    Sparse retrieval using BM25 over chunk texts.
    Returns chunks ranked by BM25 score.
    """
    if not chunks:
        return []

    # Tokenize
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)

    # Add BM25 scores to chunks
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        scored_chunks.append({**chunk, "bm25_score": float(scores[i])})

    # Sort by score descending
    scored_chunks.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored_chunks[:top_k]


def reciprocal_rank_fusion(
    dense_results: list[dict],
    sparse_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Reciprocal Rank Fusion (RRF) to merge dense and sparse retrieval results.

    Formula: score(d) = Σ 1 / (k + rank(d))

    This balances keyword-based and semantic matches, giving robust retrieval
    that handles both exact term matching and meaning-based similarity.
    """
    fused_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    # Score from dense retrieval
    for rank, result in enumerate(dense_results):
        doc_id = result["id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        chunk_data[doc_id] = result

    # Score from sparse retrieval
    for rank, result in enumerate(sparse_results):
        doc_id = result["id"]
        fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
        if doc_id not in chunk_data:
            chunk_data[doc_id] = result

    # Sort by fused score
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    fused_results = []
    for doc_id in sorted_ids:
        result = chunk_data[doc_id].copy()
        result["rrf_score"] = fused_scores[doc_id]
        fused_results.append(result)

    return fused_results


def hybrid_retrieve(
    query: str,
    dense_top_k: int | None = None,
    sparse_top_k: int | None = None,
    rrf_k: int | None = None,
) -> list[dict]:
    """
    Perform hybrid retrieval combining dense and sparse search with RRF.

    Pipeline:
    1. Dense retrieval via Pinecone (semantic similarity)
    2. Sparse retrieval via BM25 (keyword matching)
    3. Reciprocal Rank Fusion to merge both ranked lists
    """
    dense_top_k = dense_top_k or settings.dense_top_k
    sparse_top_k = sparse_top_k or settings.sparse_top_k
    rrf_k = rrf_k or settings.rrf_k

    logger.info(f"Hybrid retrieval for: '{query[:100]}...'")

    # 1. Dense retrieval from Pinecone
    dense_results = query_dense(query, top_k=dense_top_k)
    logger.info(f"Dense retrieval returned {len(dense_results)} results")

    # 2. Sparse retrieval via BM25
    # Fetch all chunks for BM25 scoring
    all_chunks = get_all_chunks_text()
    sparse_results = bm25_retrieval(query, all_chunks, top_k=sparse_top_k)
    logger.info(f"Sparse (BM25) retrieval returned {len(sparse_results)} results")

    # 3. RRF Fusion
    fused_results = reciprocal_rank_fusion(
        dense_results, sparse_results, k=rrf_k
    )
    logger.info(f"RRF fusion produced {len(fused_results)} results")

    return fused_results
