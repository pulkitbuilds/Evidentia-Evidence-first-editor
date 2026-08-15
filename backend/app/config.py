"""
Central configuration for RAGnarok backend.
Loads from environment variables / .env via pydantic-settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- NVIDIA NIM (OpenAI-compatible) ---
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.1-70b-instruct"

    # --- Embeddings ---
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"

    # --- Storage ---
    chroma_persist_dir: str = "./data/chroma"
    collection_name: str = "ragnarok_corpus"
    database_url: str = "sqlite:///./data/ragnarok.db"

    # --- Retrieval ---
    top_k: int = 5
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    # A claim is UNVERIFIED if the best merged retrieval score is below this
    min_relevance_score: float = 0.15

    # --- Chunking ---
    max_chunk_sentences: int = 3

    # --- Research assistant ---
    research_max_queries: int = 4        # how many search queries to expand a topic into
    research_results_per_query: int = 3  # web results to consider per query
    research_max_sources: int = 6        # total pages to actually fetch + ingest, across all queries
    research_fetch_timeout: int = 15     # seconds, per page fetch
    research_max_chars_per_source: int = 6000  # truncate long pages before chunking/indexing

    # --- Server ---
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]


settings = Settings()
