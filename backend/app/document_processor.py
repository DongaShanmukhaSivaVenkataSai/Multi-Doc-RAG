"""Document processing: parsing and semantic chunking."""

import os
import logging
from typing import Optional
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from app.embedding_service import get_embed_model

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def parse_pdf(file_path: str) -> list[dict]:
    """Extract text from PDF using PyMuPDF, page by page."""
    pages = []
    doc = fitz.open(file_path)
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"text": text, "page": page_num + 1})
    doc.close()
    return pages


def parse_docx(file_path: str) -> list[dict]:
    """Extract text from DOCX file."""
    doc = DocxDocument(file_path)
    full_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    if full_text:
        return [{"text": full_text, "page": 1}]
    return []


def parse_text(file_path: str) -> list[dict]:
    """Extract text from TXT/MD files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    if text:
        return [{"text": text, "page": 1}]
    return []


def parse_document(file_path: str) -> list[dict]:
    """Parse a document based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    elif ext in (".txt", ".md"):
        return parse_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def semantic_chunk_document(
    file_path: str,
    doc_name: str,
    buffer_size: int = 1,
    breakpoint_percentile_threshold: int = 95,
) -> list[TextNode]:
    """
    Parse a document and split it into semantically coherent chunks.

    Uses LlamaIndex SemanticSplitterNodeParser which groups sentences
    by embedding similarity — creating chunks that preserve meaning.
    """
    logger.info(f"Processing document: {doc_name}")

    # Parse the document into pages
    pages = parse_document(file_path)
    if not pages:
        raise ValueError(f"No text could be extracted from {doc_name}")

    # Convert to LlamaIndex Documents
    llama_docs = []
    for page_info in pages:
        doc = Document(
            text=page_info["text"],
            metadata={"doc_name": doc_name, "page": page_info["page"]},
        )
        llama_docs.append(doc)

    # Semantic chunking
    embed_model = get_embed_model()
    splitter = SemanticSplitterNodeParser(
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        embed_model=embed_model,
    )

    nodes = splitter.get_nodes_from_documents(llama_docs)

    # Add chunk index metadata
    for idx, node in enumerate(nodes):
        node.metadata["chunk_index"] = idx
        node.metadata["doc_name"] = doc_name

    logger.info(f"Created {len(nodes)} semantic chunks from {doc_name}")
    return nodes
