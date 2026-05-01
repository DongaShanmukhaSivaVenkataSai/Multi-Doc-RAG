# Multi-Doc RAG

A production-grade multi-document Hybrid RAG (Retrieval-Augmented Generation) application with a FastAPI backend and React frontend.

## Features

- **Multi-document support** — Upload and query across PDF, DOCX, TXT, and Markdown files
- **Semantic chunking** — LlamaIndex `SemanticSplitterNodeParser` for context-aware document splitting
- **Hybrid retrieval** — Dense vector search + BM25 sparse retrieval fused with Reciprocal Rank Fusion (RRF)
- **Cross-encoder reranking** — `cross-encoder/ms-marco-MiniLM-L-6-v2` reranks retrieved chunks for precision
- **Streaming LLM responses** — Groq-hosted `llama-3.3-70b-versatile` with server-sent event streaming
- **Pinecone vector store** — Scalable cloud vector database for embeddings
- **Persistent document registry** — Tracks uploaded documents across server restarts

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | FastAPI + Uvicorn |
| Embeddings | `BAAI/bge-small-en-v1.5` (FP16) via `sentence-transformers` |
| Vector DB | Pinecone |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Document parsing | PyMuPDF, python-docx |
| Frontend | React 19 + Vite |
| HTTP client | Axios |

## Project Structure

```
multi-doc-rag/
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py               # FastAPI app, routes, lifespan
│       ├── config.py             # Pydantic settings (env vars)
│       ├── models.py             # Request/response schemas
│       ├── document_processor.py # Parsing & semantic chunking
│       ├── embedding_service.py  # BGE embedding model
│       ├── pinecone_service.py   # Vector upsert / delete / search
│       ├── retrieval_service.py  # Hybrid retrieval + RRF fusion
│       ├── reranker_service.py   # Cross-encoder reranking
│       └── llm_service.py        # Groq streaming generation
├── frontend/
│   ├── index.html
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx
│       ├── ChatPanel.jsx
│       ├── DocumentSidebar.jsx
│       ├── MessageBubble.jsx
│       └── SourceCard.jsx
└── .env                          # API keys (not committed)
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- A [Pinecone](https://www.pinecone.io/) account and index (dimension: `384`, metric: `dotproduct`)
- A [Groq](https://console.groq.com/) API key

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/multi-doc-rag.git
cd multi-doc-rag
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
```

### 3. Backend setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Start the server:

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 4. Frontend setup

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`.

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `GET` | `/documents` | List all uploaded documents |
| `POST` | `/upload` | Upload a document (PDF/DOCX/TXT/MD) |
| `DELETE` | `/documents/{doc_id}` | Delete a document and its vectors |
| `POST` | `/query` | Query across documents (streaming SSE) |

### Query request body

```json
{
  "question": "What are the key findings?",
  "doc_ids": ["doc-uuid-1", "doc-uuid-2"]
}
```

`doc_ids` is optional — omit it to query across all documents.

## Configuration

All settings can be overridden via environment variables or the `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `DENSE_TOP_K` | `20` | Dense retrieval candidates |
| `SPARSE_TOP_K` | `20` | BM25 retrieval candidates |
| `RRF_K` | `60` | RRF fusion constant |
| `RERANK_TOP_K` | `5` | Chunks kept after reranking |
| `MAX_FILE_SIZE_MB` | `50` | Maximum upload file size |

## How It Works

1. **Upload** — Documents are parsed and split into semantic chunks. Each chunk is embedded with BGE and upserted into Pinecone alongside BM25 index metadata.
2. **Query** — The question is embedded and used for dense retrieval from Pinecone. BM25 runs sparse retrieval in parallel. Results are fused with RRF.
3. **Rerank** — The fused candidates are reranked by the cross-encoder, keeping the top-5 most relevant chunks.
4. **Generate** — The reranked chunks are passed as context to Groq's LLM, which streams the answer back to the frontend via SSE.
