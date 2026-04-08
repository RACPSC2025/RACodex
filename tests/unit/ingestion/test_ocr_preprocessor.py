"""
Tests del OCRPreprocessor.

Los tests que requieren OpenCV y numpy se skipean automáticamente
si las dependencias no están disponibles.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.base import LoaderUnavailableError
from src.ingestion.processors.ocr_preprocessor import OCRPreprocessor, PreprocessResult


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_fake_image(h: int = 1000, w: int = 800, channels: int = 3):
    """Crea un ndarray fake sin necesitar numpy real."""
    try:
        import numpy as np
        return np.ones((h, w, channels), dtype=np.uint8) * 200
    except ImportError:
        pytest.skip("numpy no disponible")


def make_gray_image(h: int = 1000, w: int = 800):
    try:
        import numpy as np
        return np.ones((h, w), dtype=np.uint8) * 200
    except ImportError:
        pytest.skip("numpy no disponible")


# ─── Tests: interfaz básica ───────────────────────────────────────────────────

class TestOCRPreprocessorInterface:
    def test_default_min_dpi_from_settings(self) -> None:
        preprocessor = OCRPreprocessor()
        from src.config.settings import get_settings
        assert preprocessor.min_dpi == get_settings().ocr_min_dpi

    def test_custom_min_dpi(self) -> None:
        preprocessor = OCRPreprocessor(min_dpi=200)
        assert preprocessor.min_dpi == 200

    def test_missing_opencv_raises_loader_unavailable(self, tmp_path: Path) -> None:
        preprocessor = OCRPreprocessor()
        preprocessor._magic_instance = None

        with patch("builtins.__import__", side_effect=lambda name, *a, **k: (
            (_ for _ in ()).throw(ImportError("cv2 not found"))
            if name == "cv2"
            else __import__(name, *a, **k)
        )):
            with pytest.raises((LoaderUnavailableError, ImportError)):
                preprocessor._get_cv2()

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        preprocessor = OCRPreprocessor()
        with pytest.raises(FileNotFoundError):
            preprocessor.preprocess_file(tmp_path / "no_existe.jpg")


# ─── Tests: con numpy/cv2 disponibles ────────────────────────────────────────

class TestOCRPreprocessorPipeline:
    def test_preprocess_returns_result_object(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("cv2/numpy no disponibles")

        preprocessor = OCRPreprocessor(min_dpi=72, binarize=True)
        image = make_fake_image(800, 600)
        result = preprocessor.preprocess(image)

        assert isinstance(result, PreprocessResult)
        assert result.image is not None
        assert result.original_shape == (800, 600)

    def test_grayscale_converted(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("cv2/numpy no disponibles")

        preprocessor = OCRPreprocessor(min_dpi=72, binarize=False)
        image = make_fake_image(800, 600, channels=3)
        result = preprocessor.preprocess(image)

        # El output debe ser 2D (grayscale) — no 3 canales
        assert len(result.image.shape) == 2

    def test_binarize_produces_binary_image(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            pytest.skip("cv2/numpy no disponibles")

        preprocessor = OCRPreprocessor(min_dpi=72, binarize=True)
        image = make_fake_image(800, 600)
        result = preprocessor.preprocess(image)

        unique_values = set(result.image.flatten().tolist())
        # Imagen binaria solo tiene 0 y 255
        assert unique_values.issubset({0, 255})

    def test_upscale_when_dpi_low(self) -> None:
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy no disponible")

        preprocessor = OCRPreprocessor(min_dpi=300, target_dpi=600, binarize=False)
        # Imagen pequeña que el estimador clasificará como DPI bajo
        small_image = make_fake_image(100, 80)
        result = preprocessor.preprocess(small_image)

        assert result.was_upscaled is True
        assert result.scale_factor > 1.0

    def test_no_upscale_for_large_image(self) -> None:
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy no disponible")

        preprocessor = OCRPreprocessor(min_dpi=72, binarize=False)
        # Imagen grande — DPI estimado será alto
        large_image = make_fake_image(3000, 2400)
        result = preprocessor.preprocess(large_image)

        assert result.was_upscaled is False

    def test_denoising_applied(self) -> None:
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy no disponible")

        preprocessor = OCRPreprocessor(min_dpi=72, binarize=False)
        image = make_fake_image(800, 600)
        result = preprocessor.preprocess(image)

        assert result.was_denoised is True

    def test_result_shape_recorded_correctly(self) -> None:
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy no disponible")

        preprocessor = OCRPreprocessor(min_dpi=72, binarize=False)
        image = make_fake_image(1000, 800)
        result = preprocessor.preprocess(image)

        assert result.original_shape == (1000, 800)
        assert len(result.final_shape) == 2

    def test_estimate_dpi_small_image_is_low(self) -> None:
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy no disponible")

        preprocessor = OCRPreprocessor()
        small = make_fake_image(200, 150)  # muy pequeña
        dpi = preprocessor._estimate_dpi(small)
        assert dpi < 150

    def test_estimate_dpi_large_image_is_high(self) -> None:
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy no disponible")

        preprocessor = OCRPreprocessor()
        large = make_fake_image(2500, 2000)
        dpi = preprocessor._estimate_dpi(large)
        assert dpi >= 150
