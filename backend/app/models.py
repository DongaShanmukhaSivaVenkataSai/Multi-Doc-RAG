"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel
from typing import Optional


class UploadResponse(BaseModel):
    """Response after document upload and processing."""
    status: str
    filename: str
    num_chunks: int
    message: str


class QueryRequest(BaseModel):
    """User query request."""
    question: str
    chat_history: list[dict] = []
    top_k: int = 5
    doc_ids: Optional[list[str]] = None


class SourceChunk(BaseModel):
    """A retrieved source chunk with metadata."""
    text: str
    doc_name: str
    page: Optional[int] = None
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    """Full query response (non-streaming)."""
    answer: str
    sources: list[SourceChunk]


class DocumentInfo(BaseModel):
    """Info about an uploaded document."""
    doc_name: str
    num_chunks: int
    upload_time: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    pinecone_connected: bool
    models_loaded: bool
