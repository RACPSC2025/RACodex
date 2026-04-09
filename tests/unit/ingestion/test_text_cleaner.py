"""
Tests del TextCleaner — reglas de limpieza de texto.
"""

from __future__ import annotations

import pytest

from src.ingestion.processors.text_cleaner import TextCleaner, get_cleaner


# ─── Reglas Universales ──────────────────────────────────────────────────────

class TestUniversalRules:
    """Reglas que aplican a todos los perfiles."""

    def test_normalize_line_endings(self) -> None:
        cleaner = TextCleaner(profile="default")
        result = cleaner.clean("line1\r\nline2\rline3")
        assert "\r" not in result
        assert result.count("\n") == 2

    def test_remove_multiple_blank_lines(self) -> None:
        cleaner = TextCleaner(profile="default")
        result = cleaner.clean("text\n\n\n\n\nmore")
        assert "\n\n\n" not in result

    def test_remove_trailing_whitespace(self) -> None:
        cleaner = TextCleaner(profile="default")
        result = cleaner.clean("text   \nmore\t")
        assert not result.endswith("   ")

    def test_join_broken_words(self) -> None:
        cleaner = TextCleaner(profile="default")
        result = cleaner.clean("palabra-\nrota")
        assert result == "palabrarota"

    def test_normalize_spaces(self) -> None:
        cleaner = TextCleaner(profile="default")
        result = cleaner.clean("text   with    spaces")
        assert "   " not in result

    def test_remove_control_chars(self) -> None:
        cleaner = TextCleaner(profile="default")
        result = cleaner.clean("text\x00\x01\x02more")
        assert "\x00" not in result


# ─── Perfiles ────────────────────────────────────────────────────────────────

class TestCleanerProfiles:
    """Tests de perfiles de limpieza."""

    def test_default_profile(self) -> None:
        cleaner = TextCleaner(profile="default")
        assert cleaner.profile == "default"

    def test_technical_profile(self) -> None:
        cleaner = TextCleaner(profile="technical")
        assert cleaner.profile == "technical"

    def test_ocr_output_profile(self) -> None:
        cleaner = TextCleaner(profile="ocr_output")
        assert cleaner.profile == "ocr_output"

    def test_contract_profile(self) -> None:
        cleaner = TextCleaner(profile="contract")
        assert cleaner.profile == "contract"

    def test_unknown_profile_uses_default(self) -> None:
        """Perfil desconocido solo aplica reglas universales."""
        cleaner = TextCleaner(profile="unknown")
        result = cleaner.clean("test\r\ntext")
        assert "\r" not in result


# ─── OCR Rules ───────────────────────────────────────────────────────────────

class TestOCRRules:
    """Reglas específicas de limpieza post-OCR."""

    def test_remove_ocr_artifacts(self) -> None:
        cleaner = TextCleaner(profile="ocr_output")
        result = cleaner.clean("text with © and § symbols")
        assert "©" not in result
        assert "§" not in result

    def test_remove_repeated_chars(self) -> None:
        cleaner = TextCleaner(profile="ocr_output")
        result = cleaner.clean("text with aaaaaa repeated")
        assert "aaaaa" not in result


# ─── Factory ─────────────────────────────────────────────────────────────────

class TestCleanerFactory:
    """Tests del factory de cleaners."""

    def test_get_cleaner_returns_instance(self) -> None:
        cleaner = get_cleaner("default")
        assert isinstance(cleaner, TextCleaner)

    def test_get_cleaner_caches(self) -> None:
        c1 = get_cleaner("default")
        c2 = get_cleaner("default")
        assert c1 is c2  # mismo objeto en cache

    def test_get_cleaner_different_profiles(self) -> None:
        c1 = get_cleaner("default")
        c2 = get_cleaner("technical")
        assert c1 is not c2  # diferentes perfiles
