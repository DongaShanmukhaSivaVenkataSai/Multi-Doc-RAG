"""
Fast-RAG: Production-Grade Multi-Document Hybrid RAG Application

FastAPI backend with:
- Multi-document upload (PDF, DOCX, TXT, MD)
- Semantic chunking (LlamaIndex SemanticSplitterNodeParser)
- BAAI/bge-small-en-v1.5 embeddings (FP16)
- Pinecone vector storage
- Hybrid retrieval (Dense + BM25 + RRF)
- Cross-encoder reranking
- Groq LLM streaming generation
"""

import os
import json
import shutil
import logging
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.config import settings
from app.models import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    DocumentInfo,
    HealthResponse,
)
from app.document_processor import semantic_chunk_document, SUPPORTED_EXTENSIONS
from app.pinecone_service import upsert_chunks, delete_document, check_connection as check_pinecone
from app.sparse_store import upsert_sparse_chunks, delete_sparse_document, init_db as init_sparse_db
from app.retrieval_service import hybrid_retrieve
from app.reranker_service import rerank_chunks
from app.llm_service import (
    generate_streaming,
    generate_streaming_async,
    condense_query,
    check_connection as check_groq,
)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)

# --- Document registry (in-memory, persisted to disk) ---
DOCS_REGISTRY_PATH = os.path.join(settings.upload_dir, "_docs_registry.json")


def load_docs_registry() -> dict:
    """Load the document registry from disk."""
    if os.path.exists(DOCS_REGISTRY_PATH):
        with open(DOCS_REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def save_docs_registry(registry: dict):
    """Save the document registry to disk."""
    os.makedirs(os.path.dirname(DOCS_REGISTRY_PATH), exist_ok=True)
    with open(DOCS_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


docs_registry = load_docs_registry()


# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    logger.info("🚀 Starting Fast-RAG Backend...")
    os.makedirs(settings.upload_dir, exist_ok=True)
    init_sparse_db()
    logger.info("✅ Local sparse database initialized")

    # Pre-load models (optional, for faster first query)
    logger.info("Pre-loading models...")
    try:
        from app.embedding_service import get_embed_model
        get_embed_model()
        logger.info("✅ Embedding model loaded")
    except Exception as e:
        logger.warning(f"⚠️ Could not pre-load embedding model: {e}")

    try:
        from app.reranker_service import get_reranker
        get_reranker()
        logger.info("✅ Reranker model loaded")
    except Exception as e:
        logger.warning(f"⚠️ Could not pre-load reranker model: {e}")

    logger.info("✅ Fast-RAG Backend ready!")
    yield
    logger.info("👋 Shutting down Fast-RAG Backend...")


# --- FastAPI App ---
app = FastAPI(
    title="Fast-RAG API",
    description="Production-grade Multi-Document Hybrid RAG with Semantic Chunking",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== ROUTES ====================


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health: Pinecone connection and model status."""
    pinecone_ok = check_pinecone()

    models_ok = True
    try:
        from app.embedding_service import get_embed_model
        from app.reranker_service import get_reranker
        get_embed_model()
        get_reranker()
    except Exception:
        models_ok = False

    return HealthResponse(
        status="healthy" if (pinecone_ok and models_ok) else "degraded",
        pinecone_connected=pinecone_ok,
        models_loaded=models_ok,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document.

    Pipeline: Upload → Parse → Semantic Chunk → Embed (FP16) → Upsert to Pinecone
    """
    # Validate file type
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        )

    # Validate file size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Max: {settings.max_file_size_mb}MB",
        )

    # Save file
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        # Semantic chunking (offloaded to threadpool)
        logger.info(f"📄 Processing: {file.filename}")
        nodes = await asyncio.to_thread(semantic_chunk_document, file_path, file.filename)

        # Prepare chunks for upsert
        chunks = []
        for node in nodes:
            chunks.append({
                "text": node.get_content(),
                "page": node.metadata.get("page", 1),
                "chunk_index": node.metadata.get("chunk_index", 0),
            })

        # Upsert to Pinecone and local SQLite sparse store
        num_upserted = await asyncio.to_thread(upsert_chunks, chunks, file.filename)
        await asyncio.to_thread(upsert_sparse_chunks, chunks, file.filename)

        # Update registry
        docs_registry[file.filename] = {
            "num_chunks": len(chunks),
            "upload_time": datetime.now().isoformat(),
            "file_size_mb": round(size_mb, 2),
        }
        save_docs_registry(docs_registry)

        logger.info(f"✅ Processed {file.filename}: {len(chunks)} chunks indexed")

        return UploadResponse(
            status="success",
            filename=file.filename,
            num_chunks=len(chunks),
            message=f"Successfully processed and indexed {len(chunks)} semantic chunks",
        )

    except Exception as e:
        logger.error(f"❌ Error processing {file.filename}: {e}")
        # Clean up on failure
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/query")
async def query_documents(request: QueryRequest):
    """
    Query the RAG pipeline with streaming response.

    Pipeline: Query → Hybrid Retrieve (Dense + BM25 + RRF) → Rerank → Groq Stream
    Returns SSE stream with answer tokens and source metadata.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"🔍 Query: {question[:100]}...")

    try:
        # 1. Conversational Query Reformulation (if multi-turn)
        search_query = await condense_query(question, request.chat_history)

        # 2. Hybrid retrieval (offloaded to threadpool with document scoping)
        hybrid_results = await asyncio.to_thread(
            hybrid_retrieve,
            search_query,
            request.doc_ids,
        )

        if not hybrid_results:
            async def empty_stream():
                data = {
                    "type": "sources",
                    "sources": [],
                }
                yield f"data: {json.dumps(data)}\n\n"
                yield f"data: {json.dumps({'type': 'token', 'content': 'No relevant documents found. Please upload documents first.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            return StreamingResponse(
                empty_stream(),
                media_type="text/event-stream",
            )

        # 3. Rerank (offloaded to threadpool)
        reranked = await asyncio.to_thread(
            rerank_chunks,
            search_query,
            hybrid_results,
            top_k=request.top_k,
        )

        # 4. Stream response asynchronously
        async def event_stream():
            # Send sources first
            sources = []
            for chunk in reranked:
                sources.append({
                    "text": chunk["text"][:500],  # Truncate for display
                    "doc_name": chunk.get("doc_name", "Unknown"),
                    "page": chunk.get("page", 0),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "score": round(chunk.get("rerank_score", 0), 4),
                })

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Stream LLM response without blocking event loop
            async for token in generate_streaming_async(
                question, reranked, request.chat_history
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List all uploaded documents with metadata."""
    docs = []
    for doc_name, info in docs_registry.items():
        docs.append(DocumentInfo(
            doc_name=doc_name,
            num_chunks=info.get("num_chunks", 0),
            upload_time=info.get("upload_time", ""),
        ))
    return {"documents": docs}


@app.delete("/documents/{doc_name}")
async def remove_document(doc_name: str):
    """Delete a document and its vectors from Pinecone."""
    if doc_name not in docs_registry:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_name}")

    # Delete from Pinecone and local SQLite sparse store
    success = await asyncio.to_thread(delete_document, doc_name)
    await asyncio.to_thread(delete_sparse_document, doc_name)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete from vector store")

    # Delete local file
    file_path = os.path.join(settings.upload_dir, doc_name)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove from registry
    del docs_registry[doc_name]
    save_docs_registry(docs_registry)

    logger.info(f"🗑️ Deleted document: {doc_name}")
    return {"status": "success", "message": f"Deleted {doc_name}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
