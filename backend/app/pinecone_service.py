"""Pinecone vector database service."""

import logging
import time
from typing import Optional
from pinecone import Pinecone
from app.config import settings
from app.embedding_service import embed_texts, embed_query, to_fp16

logger = logging.getLogger(__name__)

_pinecone_index = None


def get_pinecone_index():
    """Get or initialize the Pinecone index (singleton)."""
    global _pinecone_index
    if _pinecone_index is None:
        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=settings.pinecone_api_key)
        _pinecone_index = pc.Index(settings.pinecone_index_name)
        logger.info(f"Connected to Pinecone index: {settings.pinecone_index_name}")
    return _pinecone_index


def upsert_chunks(
    chunks: list[dict],
    doc_name: str,
    batch_size: int = 100,
) -> int:
    """
    Upsert document chunks into Pinecone.

    Each chunk dict should have: text, page, chunk_index
    Embeddings are converted to FP16 before storage.
    """
    index = get_pinecone_index()

    # Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    # Convert to FP16 for storage efficiency
    fp16_embeddings = to_fp16(embeddings)

    # Prepare vectors for upsert
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, fp16_embeddings)):
        vector_id = f"{doc_name}_{chunk['chunk_index']}"
        metadata = {
            "text": chunk["text"][:40000],  # Pinecone metadata limit
            "doc_name": doc_name,
            "page": chunk.get("page", 0),
            "chunk_index": chunk["chunk_index"],
        }
        vectors.append({
            "id": vector_id,
            "values": embedding,
            "metadata": metadata,
        })

    # Batch upsert
    total_upserted = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        total_upserted += len(batch)
        logger.info(f"Upserted batch {i // batch_size + 1}: {len(batch)} vectors")

    # Wait for index to be updated
    time.sleep(1)

    logger.info(f"Total vectors upserted for {doc_name}: {total_upserted}")
    return total_upserted


def query_dense(
    query: str,
    top_k: int = 20,
    doc_filter: Optional[str | list[str]] = None,
) -> list[dict]:
    """
    Dense retrieval: query Pinecone with embedding similarity.
    Supports filtering by single doc_name or list of doc_names.
    Returns list of {id, text, doc_name, page, chunk_index, score}.
    """
    index = get_pinecone_index()
    query_embedding = embed_query(query)

    # Build filter
    filter_dict = {}
    if doc_filter:
        if isinstance(doc_filter, str):
            filter_dict["doc_name"] = {"$eq": doc_filter}
        elif isinstance(doc_filter, list) and len(doc_filter) > 0:
            if len(doc_filter) == 1:
                filter_dict["doc_name"] = {"$eq": doc_filter[0]}
            else:
                filter_dict["doc_name"] = {"$in": doc_filter}

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None,
    )

    retrieved = []
    for match in results.get("matches", []):
        retrieved.append({
            "id": match["id"],
            "text": match["metadata"].get("text", ""),
            "doc_name": match["metadata"].get("doc_name", ""),
            "page": match["metadata"].get("page", 0),
            "chunk_index": match["metadata"].get("chunk_index", 0),
            "score": match["score"],
        })

    return retrieved


def delete_document(doc_name: str) -> bool:
    """Delete all vectors for a specific document."""
    index = get_pinecone_index()
    try:
        # Fetch all vector IDs for the document by prefix
        # Use list to get IDs by prefix
        ids_to_delete = []
        for id_batch in index.list(prefix=f"{doc_name}_"):
            ids_to_delete.extend(id_batch)

        if ids_to_delete:
            # Delete in batches
            batch_size = 1000
            for i in range(0, len(ids_to_delete), batch_size):
                batch = ids_to_delete[i : i + batch_size]
                index.delete(ids=batch)
            logger.info(f"Deleted {len(ids_to_delete)} vectors for {doc_name}")
        return True
    except Exception as e:
        logger.error(f"Error deleting document {doc_name}: {e}")
        return False


def get_all_chunks_text(doc_filter: Optional[str | list[str]] = None) -> list[dict]:
    """
    Fetch chunk texts from local sparse database.
    Eliminates high-latency Pinecone HTTP fetches and rate-limiting issues.
    """
    try:
        from app.sparse_store import get_all_chunks
        return get_all_chunks(doc_filter=doc_filter)
    except Exception as e:
        logger.error(f"Error fetching chunks from sparse store: {e}")
        return []


def check_connection() -> bool:
    """Check if Pinecone connection is healthy."""
    try:
        index = get_pinecone_index()
        index.describe_index_stats()
        return True
    except Exception:
        return False
