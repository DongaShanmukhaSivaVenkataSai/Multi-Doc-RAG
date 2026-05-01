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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    Returns raw float embeddings.
    """
    model = get_embed_model()
    embeddings = []
    # Process in batches for memory efficiency
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = [model.get_text_embedding(text) for text in batch]
        embeddings.extend(batch_embeddings)
    return embeddings


def embed_query(query: str) -> list[float]:
    """Generate embedding for a single query."""
    model = get_embed_model()
    return model.get_query_embedding(query)


def to_fp16(embeddings: list[list[float]]) -> list[list[float]]:
    """
    Convert embeddings to FP16 precision for storage efficiency.
    Pinecone stores these as FP16 internally when possible.
    """
    fp16_embeddings = []
    for emb in embeddings:
        arr = np.array(emb, dtype=np.float16)
        fp16_embeddings.append(arr.astype(np.float32).tolist())  # Convert back for API
    return fp16_embeddings


def get_embedding_dimension() -> int:
    """Get the dimension of the embedding model."""
    model = get_embed_model()
    test_emb = model.get_text_embedding("test")
    return len(test_emb)
