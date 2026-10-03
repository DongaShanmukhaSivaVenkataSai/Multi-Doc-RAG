"""Embedding service using BAAI/bge-small-en-v1.5 with FP16 support."""

import logging
import numpy as np
from functools import lru_cache
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from app.config import settings

logger = logging.getLogger(__name__)

_embed_model = None


def get_embed_model() -> HuggingFaceEmbedding:
    """Get or initialize the embedding model (singleton)."""
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embed_model = HuggingFaceEmbedding(
            model_name=settings.embedding_model,
            embed_batch_size=32,
        )
        logger.info("Embedding model loaded successfully")
    return _embed_model


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using batch processing.
    Returns float embeddings.
    """
    if not texts:
        return []
    model = get_embed_model()
    embeddings = []
    # Process in batches with native batch inference
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.get_text_embedding_batch(batch)
        embeddings.extend(batch_embeddings)
    return embeddings


def embed_query(query: str) -> list[float]:
    """Generate embedding for a single query."""
    model = get_embed_model()
    return model.get_query_embedding(query)


def to_fp16(embeddings: list[list[float]]) -> list[list[float]]:
    """
    Pass through embeddings preserving float precision for vector similarity.
    Maintained for backward compatibility.
    """
    return embeddings


def get_embedding_dimension() -> int:
    """Get the dimension of the embedding model."""
    model = get_embed_model()
    test_emb = model.get_text_embedding("test")
    return len(test_emb)
