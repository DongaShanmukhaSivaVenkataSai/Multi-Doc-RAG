"""Cross-encoder reranking for precision refinement."""

import logging
from sentence_transformers import CrossEncoder
from app.config import settings

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker() -> CrossEncoder:
    """Get or initialize the cross-encoder reranker (singleton)."""
    global _reranker
    if _reranker is None:
        logger.info(f"Loading reranker model: {settings.reranker_model}")
        _reranker = CrossEncoder(settings.reranker_model, max_length=512)
        logger.info("Reranker model loaded successfully")
    return _reranker


def rerank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """
    Rerank retrieved chunks using a cross-encoder model.

    The cross-encoder scores each (query, chunk) pair directly,
    providing more accurate relevance scores than bi-encoder similarity.
    This is the precision stage of the retrieval pipeline.

    Args:
        query: The user's question
        chunks: Retrieved chunks from hybrid retrieval
        top_k: Number of top chunks to return after reranking

    Returns:
        Reranked chunks with cross-encoder scores, sorted by relevance
    """
    top_k = top_k or settings.rerank_top_k

    if not chunks:
        return []

    reranker = get_reranker()

    # Create query-document pairs
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # Score all pairs
    scores = reranker.predict(pairs)

    # The ms-marco cross-encoder outputs raw logits.
    # Apply sigmoid to convert to a 0-1 probability scale for the UI.
    import math
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    # Attach scores to chunks
    reranked = []
    for chunk, score in zip(chunks, scores):
        reranked_chunk = chunk.copy()
        # Cap the raw score if it's too extreme to prevent math.exp overflow, though unlikely
        clamped_score = max(-100.0, min(100.0, float(score)))
        normalized_score = sigmoid(clamped_score)
        reranked_chunk["rerank_score"] = normalized_score
        reranked.append(reranked_chunk)

    # Sort by rerank score descending
    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

    logger.info(
        f"Reranked {len(chunks)} chunks → returning top {top_k} "
        f"(best score: {reranked[0]['rerank_score']:.4f})"
    )

    return reranked[:top_k]
