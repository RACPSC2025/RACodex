"""
Settings base — configuración común a TODOS los entornos.

Aquí van las opciones que no cambian entre development, staging, production o testing.
Los entornos específicos heredan de esta clase y solo sobreescriben lo necesario.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseSettings(BaseSettings):
    """
    Configuración base compartida por todos los entornos.

    Los entornos específicos (DevelopmentSettings, ProductionSettings, etc.)
    heredan de esta clase y sobrescriben solo los valores que difieren.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Entorno ───────────────────────────────────────────────────────────────

    app_env: Literal["development", "testing", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    root_dir: Path = Path(".")

    # ── AWS Bedrock ──────────────────────────────────────────────────────────

    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_session_token: str = ""

    bedrock_llm_model: str = "amazon.nova-pro-v1:0"
    bedrock_llm_large_ctx_model: str = "amazon.nova-lite-v1:0"
    bedrock_embeddings_model: str = "amazon.titan-embed-text-v2:0"

    # ── LLM Parámetros ───────────────────────────────────────────────────────

    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_max_tokens_large: int = 8192

    # ── Ingestion ─────────────────────────────────────────────────────────────

    chunk_size: int = 1100
    chunk_overlap: int = 220

    ocr_languages: str = "es,en"
    ocr_min_dpi: int = 150
    ocr_gpu: bool = False

    pdf_text_quality_threshold: int = 50

    llama_parse_api_key: str = ""

    docling_num_workers: int = 2

    # ── Vector Store (Chroma) ─────────────────────────────────────────────────

    chroma_persist_dir: str = "./storage/chroma"
    chroma_collection_name: str = "fenix_legal"

    # ── Base de datos relacional ──────────────────────────────────────────────

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/fenix_rag"
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # ── Retrieval ─────────────────────────────────────────────────────────────

    retrieval_top_k: int = 10
    retrieval_rerank_top_k: int = 5
    bm25_cache_dir: str = "./storage/bm25_cache"
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2"
    flashrank_cache_dir: str = "./storage/models"

    # ── API ───────────────────────────────────────────────────────────────────

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    # ─── Propiedades computadas ───────────────────────────────────────────────

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"

    @property
    def is_testing(self) -> bool:
        return self.app_env == "testing"

    @property
    def is_staging(self) -> bool:
        return self.app_env == "staging"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def upload_dir(self) -> Path:
        return self.root_dir / "storage" / "uploads"

    @property
    def storage_dir(self) -> Path:
        return self.root_dir / "storage"

    # ─── Validadores ──────────────────────────────────────────────────────────

    @field_validator("chunk_size")
    @classmethod
    def chunk_size_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("chunk_size debe ser mayor que 0")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def chunk_overlap_must_be_valid(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError("chunk_overlap no puede ser negativo")
        chunk_size = info.data.get("chunk_size", 1100)
        if v >= chunk_size:
            raise ValueError("chunk_overlap debe ser menor que chunk_size")
        return v

    @field_validator("retrieval_top_k")
    @classmethod
    def top_k_must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("retrieval_top_k debe ser mayor que 0")
        return v

    @field_validator("retrieval_rerank_top_k")
    @classmethod
    def rerank_top_k_must_be_valid(cls, v: int, info) -> int:
        if v <= 0:
            raise ValueError("retrieval_rerank_top_k debe ser mayor que 0")
        top_k = info.data.get("retrieval_top_k", 10)
        if v > top_k:
            raise ValueError("retrieval_rerank_top_k no puede ser mayor que retrieval_top_k")
        return v

    # ─── Métodos de utilidad ──────────────────────────────────────────────────

    def ensure_directories(self) -> None:
        """Crea todos los directorios necesarios si no existen."""
        dirs = [
            self.storage_dir,
            self.upload_dir,
            Path(self.chroma_persist_dir),
            Path(self.bm25_cache_dir),
            Path(self.flashrank_cache_dir),
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
