"""Application configuration loaded from environment variables."""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROOT_DIR = os.path.abspath(os.path.join(BACKEND_DIR, ".."))

ENV_FILE_BACKEND = os.path.join(BACKEND_DIR, ".env")
ENV_FILE_ROOT = os.path.join(ROOT_DIR, ".env")

if os.path.exists(ENV_FILE_BACKEND):
    load_dotenv(ENV_FILE_BACKEND)
if os.path.exists(ENV_FILE_ROOT):
    load_dotenv(ENV_FILE_ROOT, override=True)


class Settings(BaseSettings):
    """Central configuration for the RAG application."""

    groq_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "llama-3.3-70b-versatile"

    upload_dir: str = os.path.join(BACKEND_DIR, "uploads")
    max_file_size_mb: int = 50

    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    rerank_top_k: int = 5
    final_top_k: int = 5

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_BACKEND if os.path.exists(ENV_FILE_BACKEND) else ENV_FILE_ROOT,
        extra="ignore"
    )


settings = Settings()
