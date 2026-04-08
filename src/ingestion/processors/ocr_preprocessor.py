"""
Preprocesador de imágenes para OCR.

Pipeline de limpieza antes de pasar a EasyOCR:
  1. Upscale     — si DPI estimado < OCR_MIN_DPI, escalar para aumentar detalle
  2. Grayscale   — convertir a escala de grises (OCR no necesita color)
  3. Denoise     — eliminar ruido de escaneo (sal y pimienta, manchas leves)
  4. Deskew      — detectar y corregir inclinación de la página
  5. Binarize    — umbralización Otsu → imagen blanco/negro limpia para OCR

Por qué este orden importa:
  - Upscale primero: más píxeles = mejor detección de bordes en los pasos siguientes
  - Grayscale antes de denoise: el filtro trabaja en un solo canal (más rápido y efectivo)
  - Deskew antes de binarize: la rotación sobre imagen en grises es más suave
  - Binarize al final: imagen binaria es el input óptimo para Tesseract/EasyOCR

Requiere: opencv-python-headless, Pillow (numpy viene con opencv)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from src.config.logging import get_logger
from src.config.settings import get_settings
from src.ingestion.base import LoaderUnavailableError

if TYPE_CHECKING:
    pass  # evitar imports circulares en type hints

log = get_logger(__name__)


# ─── Resultado del preprocesado ────────────────────────────────────────────────

@dataclass
class PreprocessResult:
    """Resultado de preprocesar una imagen para OCR."""
    image: "np.ndarray"             # imagen preprocesada lista para OCR
    original_shape: tuple[int, int] # (alto, ancho) original
    final_shape: tuple[int, int]    # (alto, ancho) tras preprocesado
    skew_angle: float               # ángulo de inclinación detectado (grados)
    was_upscaled: bool
    was_deskewed: bool
    was_denoised: bool
    scale_factor: float = 1.0


# ─── Preprocessor ─────────────────────────────────────────────────────────────

class OCRPreprocessor:
    """
    Pipeline de preprocesamiento de imágenes para maximizar precisión OCR.

    Diseñado para documentos legales colombianos:
      - PDFs escaneados en oficinas (típicamente con ruido leve)
      - Fotos de documentos tomadas con celular (con inclinación y distorsión leve)
      - Fotocopias de baja calidad (contraste bajo, manchas)
    """

    def __init__(
        self,
        min_dpi: int | None = None,
        target_dpi: int = 300,
        denoise_strength: int = 10,
        deskew_max_angle: float = 15.0,
        binarize: bool = True,
    ) -> None:
        """
        Args:
            min_dpi: DPI mínimo estimado — por debajo de este valor se hace upscale.
                     None usa OCR_MIN_DPI de settings.
            target_dpi: DPI objetivo al hacer upscale (default 300 — estándar OCR).
            denoise_strength: Intensidad del filtro de ruido (5–15 recomendado).
                              Valores altos pueden borrar texto fino.
            deskew_max_angle: Ángulo máximo de inclinación a corregir (grados).
                              Más allá de este límite asumimos que no es inclinación
                              sino orientación intencional.
            binarize: Si True aplica umbralización Otsu (recomendado para OCR).
        """
        settings = get_settings()
        self.min_dpi = min_dpi or settings.ocr_min_dpi
        self.target_dpi = target_dpi
        self.denoise_strength = denoise_strength
        self.deskew_max_angle = deskew_max_angle
        self.binarize = binarize

    def _get_cv2(self):
        """Lazy import de OpenCV con error descriptivo."""
        try:
            import cv2  # noqa: PLC0415
            return cv2
        except ImportError as exc:
            raise LoaderUnavailableError(
                "opencv-python-headless no está instalado.\n"
                "Ejecuta: pip install opencv-python-headless"
            ) from exc

    def preprocess(self, image: "np.ndarray") -> PreprocessResult:
        """
        Aplica el pipeline completo de preprocesamiento.

        Args:
            image: Imagen como ndarray BGR (formato OpenCV estándar).
                   Puede venir de cv2.imread() o de fitz page.get_pixmap().

        Returns:
            PreprocessResult con la imagen procesada y métricas del proceso.
        """
        cv2 = self._get_cv2()

        original_shape = image.shape[:2]  # (alto, ancho)
        was_upscaled = False
        was_deskewed = False
        was_denoised = False
        scale_factor = 1.0
        skew_angle = 0.0

        # ── 1. Upscale si DPI estimado es bajo ────────────────────────────────
        estimated_dpi = self._estimate_dpi(image)
        if estimated_dpi < self.min_dpi:
            scale_factor = self.target_dpi / max(estimated_dpi, 1)
            image = self._upscale(cv2, image, scale_factor)
            was_upscaled = True
            log.debug(
                "image_upscaled",
                original_dpi=estimated_dpi,
                scale_factor=f"{scale_factor:.2f}",
                new_shape=image.shape[:2],
            )

        # ── 2. Convertir a escala de grises ───────────────────────────────────
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # ── 3. Denoising ──────────────────────────────────────────────────────
        denoised = cv2.fastNlMeansDenoising(
            gray,
            h=self.denoise_strength,
            templateWindowSize=7,
            searchWindowSize=21,
        )
        was_denoised = True

        # ── 4. Deskew ─────────────────────────────────────────────────────────
        skew_angle = self._detect_skew(cv2, denoised)
        if abs(skew_angle) > 0.3 and abs(skew_angle) <= self.deskew_max_angle:
            deskewed = self._rotate(cv2, denoised, -skew_angle)
            was_deskewed = True
            log.debug("image_deskewed", angle=f"{skew_angle:.2f}°")
        else:
            deskewed = denoised
            if abs(skew_angle) > self.deskew_max_angle:
                log.warning(
                    "deskew_skipped_large_angle",
                    angle=f"{skew_angle:.2f}°",
                    max=self.deskew_max_angle,
                )

        # ── 5. Binarización Otsu ─────────────────────────────────────────────
        if self.binarize:
            _, final = cv2.threshold(
                deskewed,
                0,     # umbral ignorado con THRESH_OTSU
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        else:
            final = deskewed

        final_shape = final.shape[:2]

        log.info(
            "ocr_preprocessing_complete",
            original_shape=original_shape,
            final_shape=final_shape,
            skew=f"{skew_angle:.2f}°",
            upscaled=was_upscaled,
            deskewed=was_deskewed,
        )

        return PreprocessResult(
            image=final,
            original_shape=original_shape,
            final_shape=final_shape,
            skew_angle=skew_angle,
            was_upscaled=was_upscaled,
            was_deskewed=was_deskewed,
            was_denoised=was_denoised,
            scale_factor=scale_factor,
        )

    def preprocess_file(self, image_path: Path) -> PreprocessResult:
        """
        Carga una imagen desde disco y aplica el pipeline completo.

        Args:
            image_path: Ruta a la imagen (JPEG, PNG, TIFF, BMP).

        Returns:
            PreprocessResult con imagen preprocesada.

        Raises:
            FileNotFoundError: Si el archivo no existe.
            ValueError: Si el archivo no es una imagen válida.
        """
        cv2 = self._get_cv2()

        path = Path(image_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Imagen no encontrada: {path}")

        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(
                f"No se pudo cargar la imagen: {path}. "
                "Verifica que sea un formato soportado (JPEG, PNG, TIFF, BMP)."
            )

        return self.preprocess(image)

    def pdf_page_to_image(self, page: object, dpi: int = 300) -> "np.ndarray":
        """
        Convierte una página de PyMuPDF a imagen ndarray para OpenCV.

        Args:
            page: fitz.Page object.
            dpi: DPI de renderizado. 300 es el estándar para OCR de calidad.

        Returns:
            Imagen BGR como ndarray (formato OpenCV).
        """
        cv2 = self._get_cv2()

        # zoom = dpi / 72 (DPI base de PDF)
        zoom = dpi / 72.0
        mat = page.get_matrix(zoom, zoom)  # fitz.Matrix
        pixmap = page.get_pixmap(matrix=mat, alpha=False)

        # Convertir de bytes RGB a ndarray BGR (formato OpenCV)
        img_array = np.frombuffer(pixmap.samples, dtype=np.uint8)
        img_array = img_array.reshape(pixmap.height, pixmap.width, 3)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # ── Métodos internos ──────────────────────────────────────────────────────

    def _estimate_dpi(self, image: "np.ndarray") -> int:
        """
        Estima el DPI de una imagen basándose en su tamaño en píxeles.

        Asume que un documento A4 estándar tiene 210×297mm.
        Si la imagen es más pequeña que A4@150dpi, el DPI es bajo.

        Esta es una heurística — no hay metadatos de DPI reales
        cuando la imagen viene de una fotografía.
        """
        height, width = image.shape[:2]
        # A4 a 150 DPI: 1240 × 1754 píxeles
        a4_150dpi_area = 1240 * 1754
        image_area = height * width

        if image_area >= a4_150dpi_area:
            # Asumir que al menos tiene 150 DPI
            return max(150, int(math.sqrt(image_area / a4_150dpi_area) * 150))
        else:
            # Imagen pequeña — DPI bajo
            return int(math.sqrt(image_area / a4_150dpi_area) * 150)

    def _upscale(
        self,
        cv2: object,
        image: "np.ndarray",
        scale_factor: float,
    ) -> "np.ndarray":
        """Escala la imagen con interpolación cúbica (mejor calidad para texto)."""
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_CUBIC,
        )

    def _detect_skew(self, cv2: object, gray: "np.ndarray") -> float:
        """
        Detecta el ángulo de inclinación de un documento usando la
        transformada de Hough en líneas horizontales del texto.

        Retorna el ángulo en grados (negativo = inclinado a la izquierda).
        """
        # Binarizar para detección de bordes
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Detectar líneas con la transformada de Hough probabilística
        edges = cv2.Canny(binary, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )

        if lines is None:
            return 0.0

        angles: list[float] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:
                continue  # línea vertical, ignorar
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Solo considerar ángulos pequeños (inclinación de texto horizontal)
            if abs(angle) <= self.deskew_max_angle:
                angles.append(angle)

        if not angles:
            return 0.0

        # Mediana para robustez ante outliers
        return float(np.median(angles))

    def _rotate(
        self,
        cv2: object,
        image: "np.ndarray",
        angle: float,
    ) -> "np.ndarray":
        """
        Rota la imagen el ángulo dado con expansión del canvas para
        no cortar texto en los bordes.
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

        # Calcular nuevo tamaño del canvas para no perder píxeles
        cos_a = abs(rotation_matrix[0, 0])
        sin_a = abs(rotation_matrix[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)

        # Ajustar el centro de la matriz de rotación
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2

        return cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )


# ─── Instancia singleton ──────────────────────────────────────────────────────

_default_preprocessor: OCRPreprocessor | None = None


def get_ocr_preprocessor() -> OCRPreprocessor:
    """Retorna la instancia singleton del preprocessor (lazy init)."""
    global _default_preprocessor  # noqa: PLW0603
    if _default_preprocessor is None:
        _default_preprocessor = OCRPreprocessor()
    return _default_preprocessor
