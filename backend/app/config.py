"""Application configuration loaded from environment variables."""

import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


class Settings(BaseSettings):
    """Central configuration for the RAG application."""

    # --- API Keys ---
    groq_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""

    # --- Model Settings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model: str = "llama-3.3-70b-versatile"

    # --- App Settings ---
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 50

    # --- Retrieval Settings ---
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60  # RRF constant
    rerank_top_k: int = 5  # Final chunks after reranking
    final_top_k: int = 5  # Chunks sent to LLM

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
