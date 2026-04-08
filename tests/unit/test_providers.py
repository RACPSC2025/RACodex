"""
Tests unitarios para src/config/providers.py

Toda la conectividad real a AWS se mockea — estos tests verifican
la lógica de construcción, caché y reset, no la integración con Bedrock.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_provider_cache():
    """Resetea el caché de providers antes y después de cada test."""
    from src.config.providers import reset_provider_cache
    reset_provider_cache()
    yield
    reset_provider_cache()


@pytest.fixture
def mock_boto3_session():
    """Mock completo de boto3.Session y su cliente bedrock-runtime."""
    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_session.client.return_value = mock_client

    with patch("src.config.providers.boto3.Session", return_value=mock_session) as mock_cls:
        yield mock_cls, mock_session, mock_client


@pytest.fixture
def mock_chat_bedrock():
    with patch("src.config.providers.ChatBedrock") as mock_cls:
        mock_cls.return_value = MagicMock()
        yield mock_cls


@pytest.fixture
def mock_bedrock_embeddings():
    with patch("src.config.providers.BedrockEmbeddings") as mock_cls:
        mock_cls.return_value = MagicMock()
        yield mock_cls


# ─── Tests: boto3 session ─────────────────────────────────────────────────────

class TestBoto3Session:
    def test_session_created_once(self, mock_boto3_session) -> None:
        mock_cls, _, _ = mock_boto3_session
        from src.config.providers import _get_boto3_session

        _get_boto3_session()
        _get_boto3_session()
        _get_boto3_session()

        # Solo se construye una vez (singleton)
        mock_cls.assert_called_once()

    def test_session_reset_creates_new_instance(self, mock_boto3_session) -> None:
        mock_cls, _, _ = mock_boto3_session
        from src.config.providers import _get_boto3_session, reset_provider_cache

        _get_boto3_session()
        reset_provider_cache()
        _get_boto3_session()

        assert mock_cls.call_count == 2

    def test_session_uses_settings_region(self, mock_boto3_session) -> None:
        mock_cls, _, _ = mock_boto3_session
        from src.config.providers import _get_boto3_session
        from src.config.settings import get_settings

        _get_boto3_session()

        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["region_name"] == get_settings().aws_region


# ─── Tests: provider inference ────────────────────────────────────────────────

class TestProviderInference:
    def test_amazon_model_inferred(self) -> None:
        from src.config.providers import _infer_provider
        assert _infer_provider("amazon.nova-pro-v1:0") == "amazon"

    def test_anthropic_model_inferred(self) -> None:
        from src.config.providers import _infer_provider
        assert _infer_provider("anthropic.claude-3-5-sonnet-20241022-v2:0") == "anthropic"

    def test_meta_model_inferred(self) -> None:
        from src.config.providers import _infer_provider
        assert _infer_provider("meta.llama3-70b-instruct-v1:0") == "meta"

    def test_unknown_prefix_defaults_to_amazon(self) -> None:
        from src.config.providers import _infer_provider
        assert _infer_provider("unknown.some-model") == "amazon"


# ─── Tests: LLM factory ───────────────────────────────────────────────────────

class TestGetLLM:
    def test_get_llm_returns_chat_bedrock(
        self, mock_boto3_session, mock_chat_bedrock
    ) -> None:
        from src.config.providers import get_llm

        llm = get_llm()
        assert llm is not None
        mock_chat_bedrock.assert_called_once()

    def test_get_llm_cached_on_second_call(
        self, mock_boto3_session, mock_chat_bedrock
    ) -> None:
        from src.config.providers import get_llm

        llm1 = get_llm()
        llm2 = get_llm()

        assert llm1 is llm2
        mock_chat_bedrock.assert_called_once()

    def test_get_large_context_llm_uses_different_model(
        self, mock_boto3_session, mock_chat_bedrock
    ) -> None:
        from src.config.providers import get_large_context_llm, get_llm
        from src.config.settings import get_settings

        get_llm()
        get_large_context_llm()

        # Deben haberse creado con model_ids distintos
        assert mock_chat_bedrock.call_count == 2
        calls = mock_chat_bedrock.call_args_list
        model_ids = [call[1].get("model_id") or call[0][0] for call in calls]
        settings = get_settings()
        assert settings.bedrock_llm_model in str(calls[0])
        assert settings.bedrock_llm_large_ctx_model in str(calls[1])

    def test_llm_uses_temperature_from_settings(
        self, mock_boto3_session, mock_chat_bedrock
    ) -> None:
        from src.config.providers import get_llm
        from src.config.settings import get_settings

        get_llm()
        call_kwargs = mock_chat_bedrock.call_args[1]
        assert call_kwargs["temperature"] == get_settings().llm_temperature


# ─── Tests: Embeddings factory ────────────────────────────────────────────────

class TestGetEmbeddings:
    def test_get_embeddings_returns_instance(
        self, mock_boto3_session, mock_bedrock_embeddings
    ) -> None:
        from src.config.providers import get_embeddings

        emb = get_embeddings()
        assert emb is not None
        mock_bedrock_embeddings.assert_called_once()

    def test_get_embeddings_cached(
        self, mock_boto3_session, mock_bedrock_embeddings
    ) -> None:
        from src.config.providers import get_embeddings

        e1 = get_embeddings()
        e2 = get_embeddings()

        assert e1 is e2
        mock_bedrock_embeddings.assert_called_once()


# ─── Tests: Health check ─────────────────────────────────────────────────────

class TestHealthCheck:
    def test_health_check_returns_true_on_success(self, mock_boto3_session) -> None:
        from src.config.providers import check_bedrock_connectivity

        _, mock_session, _ = mock_boto3_session
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.list_foundation_models.return_value = {"modelSummaries": []}
        mock_session.client.return_value = mock_bedrock_client

        result = check_bedrock_connectivity()
        assert result["bedrock"] is True

    def test_health_check_returns_false_on_failure(self, mock_boto3_session) -> None:
        from src.config.providers import check_bedrock_connectivity

        _, mock_session, _ = mock_boto3_session
        mock_bedrock_client = MagicMock()
        mock_bedrock_client.list_foundation_models.side_effect = Exception("Connection refused")
        mock_session.client.return_value = mock_bedrock_client

        result = check_bedrock_connectivity()
        assert result["bedrock"] is False
        assert "error" in result
