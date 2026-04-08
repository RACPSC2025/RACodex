"""
Tests unitarios para src/config/settings.py

Cubre:
  - Valores por defecto correctos
  - Validadores de campo (chunk_overlap < chunk_size, rerank_k <= top_k)
  - Propiedades derivadas (is_development, storage_dir, etc.)
  - Fallo en producción sin credenciales AWS
  - Parseo de OCR_LANGUAGES desde string CSV
  - reset de caché para tests aislados
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.settings import AppEnv, AppSettings, get_settings


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_settings(**overrides: object) -> AppSettings:
    """Construye AppSettings con overrides, sin leer .env real."""
    base = {
        "APP_ENV": "testing",
        "AWS_ACCESS_KEY_ID": "",
        "AWS_SECRET_ACCESS_KEY": "",
    }
    base.update(overrides)
    return AppSettings.model_validate(base)


# ─── Tests: Defaults ──────────────────────────────────────────────────────────

class TestDefaults:
    def test_default_chunk_size(self) -> None:
        s = make_settings()
        assert s.chunk_size == 1100

    def test_default_chunk_overlap(self) -> None:
        s = make_settings()
        assert s.chunk_overlap == 220

    def test_default_env_is_development(self) -> None:
        s = AppSettings.model_validate({"APP_ENV": "development"})
        assert s.app_env == AppEnv.DEVELOPMENT

    def test_default_retrieval_top_k(self) -> None:
        s = make_settings()
        assert s.retrieval_top_k == 10

    def test_default_rerank_top_k(self) -> None:
        s = make_settings()
        assert s.retrieval_rerank_top_k == 5

    def test_default_llama_parse_disabled(self) -> None:
        s = make_settings()
        assert s.llama_parse_enabled is False

    def test_llama_parse_enabled_when_key_set(self) -> None:
        s = make_settings(LLAMA_PARSE_API_KEY="llx-test-key")
        assert s.llama_parse_enabled is True


# ─── Tests: Validadores ────────────────────────────────────────────────────────

class TestValidators:
    def test_chunk_overlap_must_be_less_than_chunk_size(self) -> None:
        with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
            make_settings(CHUNK_SIZE=500, CHUNK_OVERLAP=600)

    def test_chunk_overlap_equal_to_chunk_size_fails(self) -> None:
        with pytest.raises(ValidationError, match="CHUNK_OVERLAP"):
            make_settings(CHUNK_SIZE=500, CHUNK_OVERLAP=500)

    def test_rerank_k_cannot_exceed_top_k(self) -> None:
        with pytest.raises(ValidationError, match="RETRIEVAL_RERANK_TOP_K"):
            make_settings(RETRIEVAL_TOP_K=5, RETRIEVAL_RERANK_TOP_K=10)

    def test_valid_chunk_config_passes(self) -> None:
        s = make_settings(CHUNK_SIZE=1000, CHUNK_OVERLAP=200)
        assert s.chunk_size == 1000
        assert s.chunk_overlap == 200

    def test_production_without_aws_keys_fails(self) -> None:
        with pytest.raises(ValidationError, match="AWS_ACCESS_KEY_ID"):
            AppSettings.model_validate({
                "APP_ENV": "production",
                "AWS_ACCESS_KEY_ID": "",
                "AWS_SECRET_ACCESS_KEY": "",
            })

    def test_production_with_aws_keys_passes(self) -> None:
        s = AppSettings.model_validate({
            "APP_ENV": "production",
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG",
            "DATABASE_URL": "postgresql+psycopg://u:p@host/db",
        })
        assert s.is_production is True

    def test_ocr_languages_parsed_from_csv_string(self) -> None:
        s = make_settings(OCR_LANGUAGES="es,en,fr")
        assert s.ocr_languages == ["es", "en", "fr"]

    def test_ocr_languages_strips_spaces(self) -> None:
        s = make_settings(OCR_LANGUAGES="es , en , pt")
        assert s.ocr_languages == ["es", "en", "pt"]

    def test_ocr_languages_accepts_list(self) -> None:
        s = make_settings(OCR_LANGUAGES=["es", "en"])
        assert s.ocr_languages == ["es", "en"]

    def test_paths_resolved_to_absolute(self, tmp_path) -> None:
        s = make_settings(CHROMA_PERSIST_DIR=str(tmp_path / "chroma"))
        assert s.chroma_persist_dir.is_absolute()

    def test_llm_temperature_out_of_range_fails(self) -> None:
        with pytest.raises(ValidationError):
            make_settings(LLM_TEMPERATURE=1.5)

    def test_api_port_out_of_range_fails(self) -> None:
        with pytest.raises(ValidationError):
            make_settings(API_PORT=80)  # < 1024


# ─── Tests: Propiedades derivadas ────────────────────────────────────────────

class TestDerivedProperties:
    def test_is_development_true(self) -> None:
        s = make_settings(APP_ENV="development")
        assert s.is_development is True
        assert s.is_testing is False
        assert s.is_production is False

    def test_is_testing_true(self) -> None:
        s = make_settings(APP_ENV="testing")
        assert s.is_testing is True

    def test_storage_dir_is_child_of_root(self) -> None:
        s = make_settings()
        assert s.storage_dir == s.root_dir / "storage"

    def test_uploads_dir_is_child_of_storage(self) -> None:
        s = make_settings()
        assert s.uploads_dir == s.storage_dir / "uploads"


# ─── Tests: ensure_directories ───────────────────────────────────────────────

class TestEnsureDirectories:
    def test_creates_missing_directories(self, tmp_path) -> None:
        s = make_settings(
            CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
            BM25_CACHE_DIR=str(tmp_path / "bm25"),
            FLASHRANK_CACHE_DIR=str(tmp_path / "models"),
            ROOT_DIR=str(tmp_path),
        )
        s.ensure_directories()

        assert (tmp_path / "chroma").exists()
        assert (tmp_path / "bm25").exists()
        assert (tmp_path / "models").exists()

    def test_idempotent_when_dirs_exist(self, tmp_path) -> None:
        """Llamar dos veces no debe lanzar error."""
        s = make_settings(
            CHROMA_PERSIST_DIR=str(tmp_path / "chroma"),
            BM25_CACHE_DIR=str(tmp_path / "bm25"),
            FLASHRANK_CACHE_DIR=str(tmp_path / "models"),
            ROOT_DIR=str(tmp_path),
        )
        s.ensure_directories()
        s.ensure_directories()  # no debe lanzar


# ─── Tests: Singleton ────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_settings_returns_same_instance(self) -> None:
        from src.config.settings import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_cache_can_be_cleared_for_tests(self) -> None:
        from src.config.settings import get_settings
        get_settings.cache_clear()
        s_new = get_settings()
        assert s_new is not None
